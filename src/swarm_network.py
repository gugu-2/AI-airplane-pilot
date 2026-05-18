"""
Aegis OS \u2014 Swarm Network
Fix #11: HMAC-SHA256 message authentication to prevent rogue swarm injection attacks
Fix #25: Non-blocking async UDP sendto using run_in_executor
"""
import asyncio
import json
import socket
import hashlib
import hmac
import os
import logging

# Fix #11: Pre-shared fleet secret (in production, load from env variable or HSM)
FLEET_SECRET = os.getenv("AEGIS_FLEET_SECRET", "aegis-default-fleet-key-CHANGE-IN-PRODUCTION").encode()

def _sign_message(payload: dict) -> str:
    """Generate an HMAC-SHA256 signature for a message payload."""
    msg_bytes = json.dumps(payload, sort_keys=True).encode('utf-8')
    return hmac.new(FLEET_SECRET, msg_bytes, hashlib.sha256).hexdigest()

def _verify_message(payload: dict, signature: str) -> bool:
    """Verify the HMAC-SHA256 signature of an incoming message."""
    expected = _sign_message(payload)
    return hmac.compare_digest(expected, signature)


class SwarmNetwork:
    """
    A decentralized UDP-based networking module for multi-drone swarm coordination.
    Fix #11: All messages are signed with HMAC-SHA256 using a pre-shared fleet key.
    Fix #25: sendto() uses run_in_executor() to avoid blocking the async event loop.
    """
    def __init__(self, drone_id="Alpha", port=5555):
        self.drone_id = drone_id
        self.port = port
        self.mapper = None
        self.on_swarm_telemetry = None
        self._loop = None
        
        # Setup UDP socket
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM, socket.IPPROTO_UDP)
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.sock.bind(("", self.port))
        self.sock.setblocking(False)

        print(f"[Swarm-{self.drone_id}] Initialized UDP Swarm Network on port {self.port}")
        print(f"[Swarm-{self.drone_id}] HMAC-SHA256 authentication active.")

    def attach_mapper(self, mapper):
        self.mapper = mapper

    def _build_signed_message(self, message: dict) -> bytes:
        """Attaches a signature field and serializes the message for sending."""
        # Sign only the payload (without the signature field itself)
        signature = _sign_message(message)
        message["sig"] = signature
        return json.dumps(message).encode('utf-8')

    def _send_udp(self, data: bytes):
        """Synchronous UDP send \u2014 called in executor thread pool (Fix #25)."""
        try:
            self.sock.sendto(data, ('<broadcast>', self.port))
        except Exception as e:
            print(f"[Swarm Error] Failed to broadcast: {e}")

    async def _async_send(self, data: bytes):
        """Fix #25: Wraps synchronous sendto in run_in_executor to avoid blocking the event loop."""
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, self._send_udp, data)

    def broadcast_obstacle(self, lat: float, lon: float):
        """Broadcasts a signed obstacle discovery to the swarm."""
        message = {
            "type": "OBSTACLE",
            "drone_id": self.drone_id,
            "lat": lat,
            "lon": lon
        }
        data = self._build_signed_message(message)
        # Fire-and-forget \u2014 swarm broadcasts are best-effort
        asyncio.ensure_future(self._async_send(data))
        print(f"[Swarm-{self.drone_id}] Signed obstacle broadcast: ({lat:.6f}, {lon:.6f})")

    def broadcast_telemetry(self, lat: float, lon: float, alt: float):
        """Broadcasts signed live position to the swarm."""
        message = {
            "type": "TELEMETRY",
            "drone_id": self.drone_id,
            "lat": lat,
            "lon": lon,
            "alt": alt
        }
        data = self._build_signed_message(message)
        asyncio.ensure_future(self._async_send(data))

    async def listen_for_swarm(self):
        """Background task that listens for incoming authenticated swarm broadcasts."""
        print(f"[Swarm-{self.drone_id}] Listening for authenticated Swarm data...")
        loop = asyncio.get_event_loop()
        
        while True:
            try:
                data, addr = await loop.sock_recvfrom(self.sock, 2048)
                full_message = json.loads(data.decode('utf-8'))
                
                # Fix #11: Extract and verify signature BEFORE trusting any data
                received_sig = full_message.pop("sig", None)
                if received_sig is None or not _verify_message(full_message, received_sig):
                    print(f"[Swarm-{self.drone_id}] WARNING: REJECTED unauthenticated message from {addr[0]}!")
                    continue
                
                # Ignore our own broadcasts
                if full_message.get("drone_id") == self.drone_id:
                    continue
                    
                if full_message.get("type") == "OBSTACLE" and self.mapper:
                    lat = full_message["lat"]
                    lon = full_message["lon"]
                    drone_src = full_message['drone_id']
                    print(f"\n[Swarm Intelligence] Authenticated obstacle from Drone {drone_src} at ({lat:.6f}, {lon:.6f})")
                    self.mapper.mark_obstacle(lat, lon)
                    
                elif full_message.get("type") == "TELEMETRY" and self.on_swarm_telemetry:
                    await self.on_swarm_telemetry(full_message)
                    
            except asyncio.CancelledError:
                print(f"[Swarm-{self.drone_id}] Shutting down Swarm listener.")
                break
            except BlockingIOError:
                pass
            except Exception:
                pass
                
            await asyncio.sleep(0.1)

import asyncio
import json
import socket
import logging

class SwarmNetwork:
    """
    A decentralized UDP-based networking module that allows multiple drone AI Brains
    to communicate with each other on the local network. 
    Uses UDP broadcast to share semantic map data (obstacles) instantly.
    """
    def __init__(self, drone_id="Alpha", port=5555):
        self.drone_id = drone_id
        self.port = port
        self.mapper = None # Will hold a reference to the local SemanticMap
        
        # Setup UDP socket for broadcasting
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM, socket.IPPROTO_UDP)
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        # Bind to port for receiving
        self.sock.bind(("", self.port))
        self.sock.setblocking(False)

        print(f"[Swarm-{self.drone_id}] Initialized UDP Swarm Network on port {self.port}")

    def attach_mapper(self, mapper):
        """Attaches the local SemanticMap so the swarm can inject external obstacles."""
        self.mapper = mapper

    def broadcast_obstacle(self, lat: float, lon: float):
        """Broadcasts a discovered obstacle to all other drones in the swarm."""
        message = {
            "type": "OBSTACLE",
            "drone_id": self.drone_id,
            "lat": lat,
            "lon": lon
        }
        data = json.dumps(message).encode('utf-8')
        try:
            # Broadcast to entire local subnet
            self.sock.sendto(data, ('<broadcast>', self.port))
            print(f"[Swarm-{self.drone_id}] Broadcasted obstacle at ({lat:.6f}, {lon:.6f}) to Swarm.")
        except Exception as e:
            print(f"[Swarm Error] Failed to broadcast: {e}")

    async def listen_for_swarm(self):
        """Background task that listens for incoming swarm broadcasts."""
        print(f"[Swarm-{self.drone_id}] Listening for Swarm intelligence data...")
        loop = asyncio.get_event_loop()
        
        while True:
            try:
                # Receive data asynchronously
                data, addr = await loop.sock_recvfrom(self.sock, 1024)
                message = json.loads(data.decode('utf-8'))
                
                # Ignore our own broadcasts
                if message.get("drone_id") == self.drone_id:
                    continue
                    
                if message.get("type") == "OBSTACLE" and self.mapper:
                    lat = message["lat"]
                    lon = message["lon"]
                    print(f"\n[Swarm Intelligence] Received obstacle data from Drone {message['drone_id']} at ({lat:.6f}, {lon:.6f})!")
                    
                    # Update our own local memory map with the Swarm's knowledge
                    self.mapper.mark_obstacle(lat, lon)
                    
            except BlockingIOError:
                pass
            except Exception as e:
                pass
                
            await asyncio.sleep(0.1) # Yield to event loop

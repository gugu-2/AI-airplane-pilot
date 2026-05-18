import asyncio
import json
import websockets
import sys
import os

# Ensure cognitive module is in path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '.')))
from cognitive.atc_agent import ATCNaturalLanguageProcessor

# R6 FIX: WebSocket API key. Set the AEGIS_WS_KEY env variable in production.
# Clients must send {"type": "AUTH", "key": "<key>"} as their first message.
AEGIS_WS_KEY = os.getenv("AEGIS_WS_KEY", "aegis-local-dev-key")

class TelemetryServer:
    def __init__(self, host="0.0.0.0", port=8765):
        self.host = host
        self.port = port
        self.clients = set()
        self.atc_nlp = ATCNaturalLanguageProcessor(callsign="Aegis 1")
        self.on_atc_intent = None
        # R14 FIX: Track last broadcast time to enforce a 50ms rate limit.
        # This prevents the mission loop from flooding browsers with hundreds of msgs/sec.
        import time as _time
        self._last_broadcast = 0.0
        self._time = _time

    async def register(self, websocket):
        """Handles a new WebSocket connection with authentication."""
        # R6 FIX: Require auth token as the first message within 2 seconds
        try:
            auth_raw = await asyncio.wait_for(websocket.recv(), timeout=2.0)
            auth_data = json.loads(auth_raw)
            if auth_data.get("type") != "AUTH" or auth_data.get("key") != AEGIS_WS_KEY:
                await websocket.send(json.dumps({"type": "ERROR", "msg": "Unauthorized"}))
                await websocket.close()
                print(f"[TelemetryServer] Rejected unauthorized connection.")
                return
        except (asyncio.TimeoutError, json.JSONDecodeError):
            await websocket.close()
            return

        self.clients.add(websocket)
        print(f"[TelemetryServer] Authenticated client connected. Total: {len(self.clients)}")
        try:
            async for message in websocket:
                try:
                    data = json.loads(message)
                    if data.get("type") == "VHF_RADIO":
                        transcript = data.get("text", "")
                        parsed = self.atc_nlp.process_audio_transcript(transcript)
                        
                        if parsed["intent"] != "IGNORE":
                            readback = self.atc_nlp.generate_readback(parsed)
                            # Broadcast readback to dashboard
                            await self.broadcast({
                                "type": "VHF_READBACK",
                                "transcript": transcript,
                                "readback": readback,
                                "intent": parsed["intent"]
                            })
                            
                            # Trigger physical override
                            if self.on_atc_intent:
                                await self.on_atc_intent(parsed)
                    
                    elif data.get("type") == "GET_FLIGHT_LOGS":
                        import sqlite3
                        # R2 FIX: Use absolute path so the DB is always found
                        # regardless of which directory the server was launched from.
                        _db_path = os.path.join(
                            os.path.dirname(os.path.abspath(__file__)),
                            '..', 'flight_data_recorder.db'
                        )
                        try:
                            conn = sqlite3.connect(_db_path)
                            cursor = conn.cursor()
                            cursor.execute("SELECT mission_id, start_time, end_time, status, total_distance_m, obstacles_encountered, evasions_triggered, errors_warnings FROM flight_summary ORDER BY start_time DESC LIMIT 50")
                            rows = cursor.fetchall()
                            
                            logs_data = []
                            for r in rows:
                                logs_data.append({
                                    "mission_id": r[0],
                                    "start_time": r[1],
                                    "end_time": r[2],
                                    "status": r[3],
                                    "distance": r[4],
                                    "obstacles": r[5],
                                    "evasions": r[6],
                                    "errors": r[7]
                                })
                            
                            conn.close()
                            
                            await websocket.send(json.dumps({
                                "type": "FLIGHT_LOGS_DATA",
                                "data": logs_data
                            }))
                        except Exception as e:
                            print(f"[TelemetryServer] DB Error: {e}")
                            
                except json.JSONDecodeError:
                    pass
        except websockets.exceptions.ConnectionClosed:
            pass
        finally:
            self.clients.remove(websocket)

    async def broadcast(self, data: dict):
        """Broadcasts a JSON message to all connected clients.
        R14 FIX: Throttled to max 20 messages/sec (50ms minimum between sends).
        Critical messages (level=critical) bypass the throttle.
        """
        if not self.clients:
            return
        
        # Bypass throttle for critical alerts
        is_critical = isinstance(data.get('data'), dict) and data['data'].get('level') == 'critical'
        now = self._time.time()
        if not is_critical and (now - self._last_broadcast) < 0.05:
            return  # Drop this message — too soon after last broadcast
        self._last_broadcast = now
            
        message = json.dumps(data)
        await asyncio.gather(*(client.send(message) for client in self.clients), return_exceptions=True)

    async def start_server(self):
        """Starts the WebSocket server in the background."""
        print(f"[TelemetryServer] Starting WebSocket server on ws://{self.host}:{self.port}")
        try:
            async with websockets.serve(self.register, self.host, self.port):
                # Run forever
                await asyncio.Future()
        except asyncio.CancelledError:
            print("[TelemetryServer] Shutting down WebSocket server.")

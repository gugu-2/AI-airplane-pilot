import asyncio
import json
import websockets

class TelemetryServer:
    def __init__(self, host="0.0.0.0", port=8765):
        self.host = host
        self.port = port
        self.clients = set()
        
    async def register(self, websocket):
        self.clients.add(websocket)
        try:
            await websocket.wait_closed()
        finally:
            self.clients.remove(websocket)

    async def broadcast(self, data: dict):
        """Broadcasts a JSON dictionary to all connected web clients."""
        if not self.clients:
            return
            
        message = json.dumps(data)
        # Use asyncio.gather to send to all clients concurrently
        await asyncio.gather(*(client.send(message) for client in self.clients), return_exceptions=True)

    async def start_server(self):
        """Starts the WebSocket server in the background."""
        print(f"[TelemetryServer] Starting WebSocket server on ws://{self.host}:{self.port}")
        async with websockets.serve(self.register, self.host, self.port):
            # Run forever
            await asyncio.Future()

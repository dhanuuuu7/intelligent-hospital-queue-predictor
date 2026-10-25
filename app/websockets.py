"""
Phase 12.2: Real-Time Event Bus & WebSockets Broadcast Engine
Provides instant, push-based queue updates to React clients without polling.
"""

import time
import json
from typing import List, Dict, Any
from fastapi import WebSocket, WebSocketDisconnect


class ConnectionManager:
    """
    Thread-safe FastAPI ConnectionManager for broadcasting queue events to web clients.
    """

    def __init__(self):
        self.active_connections: List[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket):
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)

    async def broadcast(self, event_type: str, data: Dict[str, Any]):
        """
        Broadcasts JSON event payload to all connected frontend clients.
        """
        message = {
            "event_type": event_type,
            "timestamp": int(time.time()),
            "data": data
        }
        
        dead_connections = []
        for connection in self.active_connections:
            try:
                await connection.send_json(message)
            except Exception:
                dead_connections.append(connection)

        for conn in dead_connections:
            self.disconnect(conn)


# Global WebSocket Connection Manager Instance
ws_manager = ConnectionManager()

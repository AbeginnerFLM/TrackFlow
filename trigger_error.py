import asyncio
import websockets
import json
import cv2
import numpy as np

async def test():
    uri = "ws://localhost:9001"
    async with websockets.connect(uri) as websocket:
        # Create image
        img = np.zeros((640, 640, 3), dtype=np.uint8)
        _, buffer = cv2.imencode('.jpg', img)
        binary_data = buffer.tobytes()
        
        print(f"Sending header...")
        # 1. Send Header
        header = {
            "type": "infer_header",
            "request_id": "test_bin_1",
            "frame_id": 1,
            "session_id": "test_session",
            "config": {"tracker": {"enabled": False}}
        }
        await websocket.send(json.dumps(header))
        
        print(f"Sending binary data ({len(binary_data)} bytes)...")
        # 2. Send Binary
        await websocket.send(binary_data)
        
        response = await websocket.recv()
        print(f"Received: {response}")

if __name__ == "__main__":
    asyncio.get_event_loop().run_until_complete(test())

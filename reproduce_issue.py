import asyncio
import websockets
import json
import numpy as np
import cv2
import base64
import sys

async def test_inference():
    uri = "ws://127.0.0.1:9001"
    print(f"Connecting to {uri}...")
    
    try:
        async with websockets.connect(uri) as websocket:
            print("Connected!")
            
            # 1. Create a dummy image
            # 464x261 is the size mentioned in user logs
            img = np.random.randint(0, 255, (261, 464, 3), dtype=np.uint8)
            
            # 2. Encode to JPEG (mimic browser compression)
            # Browser compressed 3561KB -> 290KB (approx 222KB binary)
            _, buffer = cv2.imencode('.jpg', img, [cv2.IMWRITE_JPEG_QUALITY, 90])
            binary_data = buffer.tobytes()
            print(f"Prepared binary data size: {len(binary_data)} bytes")
            
            # 3. Send binary data
            print("Sending binary data...")
            await websocket.send(binary_data)
            
            # 4. Wait for response
            print("Waiting for response...")
            response = await websocket.recv()
            print(f"Received response: {response}")
            
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    asyncio.get_event_loop().run_until_complete(test_inference())

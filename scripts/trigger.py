import asyncio
import websockets
import json
import sys
from pathlib import Path


async def hello():
    image_path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("frame_000000.jpg")
    if not image_path.exists():
        raise FileNotFoundError(f"Image file not found: {image_path}")

    binary_data = image_path.read_bytes()
    uri = "ws://localhost:9001"
    async with websockets.connect(uri) as websocket:
        header = {
            "type": "infer_header",
            "request_id": "test_cuda_1",
            "frame_id": 1,
            "session_id": "trigger_script",
            "config": {
                "yolo": {
                    "use_cuda": True
                }
            }
        }

        await websocket.send(json.dumps(header))
        await websocket.send(binary_data)
        print(f"> Sent header: {header}")
        print(f"> Sent binary image: {image_path} ({len(binary_data)} bytes)")

        resp = await websocket.recv()
        print(f"< Received: {resp}")


asyncio.run(hello())

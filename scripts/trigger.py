import asyncio
import websockets
import json

async def hello():
    uri = "ws://localhost:9001"
    async with websockets.connect(uri) as websocket:
        req = {
            "type": "infer",
            "request_id": "test_cuda_1",
            "image": "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKwAEQAAAABJRU5ErkJggg==", 
            "config": {
                "use_cuda": True
            }
        }
        await websocket.send(json.dumps(req))
        print(f"> Sent request: {req}")
        
        resp = await websocket.recv()
        print(f"< Received: {resp}")

asyncio.run(hello())

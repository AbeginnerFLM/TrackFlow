#!/usr/bin/env python3
"""
Automated test script for TrackFlow image upload debugging.
Simulates browser behavior: sends infer_header then binary image data.
"""

import asyncio
import websockets
import json
import argparse
import sys
from pathlib import Path


async def test_image_upload(server_url: str, image_path: str, verbose: bool = False):
    """
    Test image upload via WebSocket binary protocol.
    
    Protocol:
    1. Send JSON header with type="infer_header"
    2. Send binary image data
    3. Receive JSON response
    """
    print(f"[*] Connecting to {server_url}...")
    
    try:
        async with websockets.connect(
            server_url,
            max_size=100 * 1024 * 1024,  # 100MB
            ping_interval=30,
            ping_timeout=10
        ) as websocket:
            print(f"[+] Connected!")
            
            # Load image
            image_file = Path(image_path)
            if not image_file.exists():
                print(f"[!] Error: Image file not found: {image_path}")
                return False
            
            binary_data = image_file.read_bytes()
            print(f"[*] Loaded image: {image_path} ({len(binary_data)} bytes)")
            
            # Prepare header
            header = {
                "type": "infer_header",
                "request_id": "test_001",
                "frame_id": 1,
                "session_id": "test_session",
                "config": {
                    "yolo": {"model_path": "models/yolo11.onnx"}
                },
                "pipeline": ["decoder"]  # Only test decoder first
            }
            
            # Send header
            print(f"[*] Sending header: {json.dumps(header, indent=2) if verbose else 'infer_header'}")
            await websocket.send(json.dumps(header))
            
            # Send binary data
            print(f"[*] Sending binary image data ({len(binary_data)} bytes)...")
            await websocket.send(binary_data)
            
            # Wait for response
            print(f"[*] Waiting for response...")
            response = await asyncio.wait_for(websocket.recv(), timeout=30)
            
            # Parse response
            try:
                resp_json = json.loads(response)
                if resp_json.get("type") == "result":
                    print(f"[+] SUCCESS! Response:")
                    print(json.dumps(resp_json, indent=2))
                    return True
                else:
                    print(f"[!] FAILED! Error response:")
                    print(json.dumps(resp_json, indent=2))
                    return False
            except json.JSONDecodeError:
                print(f"[!] Invalid JSON response: {response[:200]}")
                return False
                
    except websockets.exceptions.ConnectionClosedError as e:
        print(f"[!] Connection closed: {e}")
        return False
    except asyncio.TimeoutError:
        print(f"[!] Timeout waiting for response")
        return False
    except Exception as e:
        print(f"[!] Error: {type(e).__name__}: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Test TrackFlow image upload via WebSocket"
    )
    parser.add_argument(
        "--server", "-s",
        default="ws://localhost:9001",
        help="WebSocket server URL (default: ws://localhost:9001)"
    )
    parser.add_argument(
        "--image", "-i",
        default="frame_000000.jpg",
        help="Image file to upload (default: frame_000000.jpg)"
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Verbose output"
    )
    parser.add_argument(
        "--loop", "-l",
        type=int,
        default=1,
        help="Number of test iterations (default: 1)"
    )
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("TrackFlow Image Upload Test")
    print("=" * 60)
    
    success_count = 0
    for i in range(args.loop):
        if args.loop > 1:
            print(f"\n--- Test {i+1}/{args.loop} ---")
        
        result = asyncio.run(test_image_upload(
            args.server,
            args.image,
            args.verbose
        ))
        
        if result:
            success_count += 1
    
    print("\n" + "=" * 60)
    print(f"Results: {success_count}/{args.loop} tests passed")
    print("=" * 60)
    
    return 0 if success_count == args.loop else 1


if __name__ == "__main__":
    sys.exit(main())

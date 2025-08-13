#!/usr/bin/env python3

import json
import os
import asyncio
import logging
import websockets
import concurrent.futures
import time
from vosk import Model, KaldiRecognizer, GpuInit

class Args:
    interface = "0.0.0.0"
    port = 2700
    sample_rate = 8000
    model_path = "/opt/vosk-server/model"
    log_level = "INFO"

args = Args()
ACTIVE_LOGGER = os.getenv("LOGGER", "").strip()

logger = logging.getLogger("asr_server")
logger.setLevel(getattr(logging, args.log_level.upper(), logging.INFO))
handler = logging.StreamHandler()
formatter = logging.Formatter("%(asctime)s %(levelname)s [%(name)s] %(message)s")
handler.setFormatter(formatter)
logger.addHandler(handler)

def llFull(msg):
    if ACTIVE_LOGGER == "llFull":
        logger.info(f"[llFull] {msg}")

def llExtLogic(msg):
    if ACTIVE_LOGGER in ("llExtLogic", "llFull"):
        logger.info(f"[llExtLogic] {msg}")

def llBaselogic(msg):
    if ACTIVE_LOGGER in ("llBaselogic", "llExtLogic", "llFull"):
        logger.info(f"[llBaselogic] {msg}")

GpuInit()
start_time = time.time()
llBaselogic(f"Loading model from: {args.model_path}")
model = Model(args.model_path)
llBaselogic(f"Model loaded in {time.time() - start_time:.2f} seconds")

pool = concurrent.futures.ThreadPoolExecutor(max_workers=os.cpu_count())

def process_chunk(rec, message):
    if isinstance(message, str) and message.strip() == '{"eof" : 1}':
        return rec.FinalResult(), True
    if isinstance(message, bytes) and message.strip() == b'{"eof" : 1}':
        return rec.FinalResult(), True
    elif rec.AcceptWaveform(message):
        return rec.Result(), False
    else:
        return rec.PartialResult(), False

async def recognize(websocket):
    llBaselogic(f"New connection from {websocket.remote_address}")
    rec = None
    sample_rate = args.sample_rate
    last_message_time = time.time()

    async def timeout_checker():
        while True:
            await asyncio.sleep(5)
            if time.time() - last_message_time > 15:
                llBaselogic(f"Closing idle connection {websocket.remote_address} due to 15s timeout")
                await websocket.close()
                break

    timeout_task = asyncio.create_task(timeout_checker())

    try:
        async for message in websocket:
            last_message_time = time.time()

            if isinstance(message, str) and "config" in message:
                try:
                    jobj = json.loads(message)["config"]
                    llExtLogic(f"Received config: {jobj}")
                    if "sample_rate" in jobj:
                        sample_rate = float(jobj["sample_rate"])
                except Exception as e:
                    llExtLogic(f"Invalid config: {e}")
                continue

            if not rec:
                rec = KaldiRecognizer(model, sample_rate)
                rec.SetWords(True)
                llBaselogic("Recognizer created")

            loop = asyncio.get_running_loop()
            result, stop = await loop.run_in_executor(pool, process_chunk, rec, message)
            llFull(result)
            await websocket.send(result)

            if stop:
                llBaselogic(f"EOF received from {websocket.remote_address}")
                break

    except websockets.exceptions.ConnectionClosedOK:
        llBaselogic(f"Client {websocket.remote_address} closed connection normally.")
    except websockets.exceptions.ConnectionClosedError as e:
        llBaselogic(f"Connection closed with error: {e}")
    except Exception as e:
        llBaselogic(f"Exception: {e}")
    finally:
        timeout_task.cancel()
        if rec:
            final = rec.FinalResult()
            if final and '"text"' in final and not '"text": ""' in final:
                llFull(f"Final result (forced): {final}")
                try:
                    await websocket.send(final)
                except Exception as e:
                    llBaselogic(f"Final result send failed: {e}")

        try:
            await websocket.close()
            llBaselogic(f"WebSocket explicitly closed for {websocket.remote_address}")
        except Exception as e:
            llBaselogic(f"WebSocket close failed: {e}")

        llBaselogic(f"Connection from {websocket.remote_address} closed")

async def main():
    llBaselogic(f"Starting server on {args.interface}:{args.port}")
    async with websockets.serve(recognize, args.interface, args.port, ping_timeout=None):
        llBaselogic(f"Server listening on {args.interface}:{args.port}")
        await asyncio.Future()

if __name__ == "__main__":
    asyncio.run(main())
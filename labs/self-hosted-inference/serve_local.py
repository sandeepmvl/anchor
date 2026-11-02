import argparse
import asyncio
import os
from pathlib import Path
import socket
import sys
import threading

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from anchor_ai.provider import Delta
from anchor_ai.serving_benchmark import LocalServing, WORKLOAD, configuration, frame


def real_generator(weights):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer
    tokenizer = AutoTokenizer.from_pretrained(weights, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(weights, local_files_only=True)
    model.eval()

    async def generate(message):
        inputs = tokenizer(str(message), return_tensors='pt')
        streamer = TextIteratorStreamer(tokenizer, skip_prompt=True,
                                       skip_special_tokens=True, timeout=30)
        errors = []

        def worker():
            try:
                with torch.inference_mode():
                    model.generate(**inputs, streamer=streamer, max_new_tokens=32,
                                   do_sample=False)
            except BaseException as error:
                errors.append(error)
                streamer.end()

        thread = threading.Thread(target=worker, daemon=True)
        thread.start()
        sentinel = object()

        def take():
            return next(streamer, sentinel)

        try:
            while True:
                text = await asyncio.to_thread(take)
                if text is sentinel:
                    break
                if text:
                    yield Delta(text=text)
            if errors:
                raise RuntimeError('model_generation_failed')
            yield Delta(stop='stop')
        finally:
            await asyncio.to_thread(thread.join, 31)
            if thread.is_alive():
                # Do not admit another generation on top of an abandoned worker.
                os._exit(1)
    return generate


async def smoke(server, c):
    with socket.socket() as sock:
        sock.bind((c['host'], 0))
        print('loopback-only bind verified:', sock.getsockname()[0])
    events = [event async for event in server.events(WORKLOAD)]
    assert any(e.get('text') for e in events) and events[-1] == {'done': True}
    print('execution_mode=offline; SSE generator verified; network delivery, weights and hardware NOT verified')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', choices=['tiny-llama'], default='tiny-llama')
    parser.add_argument('--smoke', action='store_true')
    args = parser.parse_args()
    c = configuration()
    mode = os.environ.get('ANCHOR_SELF_HOSTED_MODE', 'offline')
    if mode not in {'offline', 'transformers'}:
        raise SystemExit('invalid serving mode')
    if args.smoke and mode != 'offline':
        raise SystemExit('smoke requires offline mode')
    generate = None
    if mode == 'transformers':
        weights = os.environ.get('ANCHOR_LOCAL_WEIGHTS', '')
        if not weights or not Path(weights).is_dir():
            raise SystemExit('local model weights required; no fake fallback')
        generate = real_generator(weights)
    server = LocalServing(generate)
    if args.smoke:
        asyncio.run(smoke(server, c))
        return
    from fastapi import FastAPI, Request
    from fastapi.responses import StreamingResponse, JSONResponse
    import uvicorn
    app = FastAPI()

    @app.post('/generate')
    async def endpoint(request: Request):
        body = await request.body()
        if len(body) > 1024:
            return JSONResponse({'error': 'invalid_request'}, status_code=400)
        try:
            import json
            payload = json.loads(body)
            if payload != {'prompt': WORKLOAD, 'max_new_tokens': 32}:
                raise ValueError()
        except (ValueError, TypeError):
            return JSONResponse({'error': 'benchmark_workload_required'}, status_code=400)

        async def encoded():
            async for event in server.events(WORKLOAD):
                yield frame(event)
        return StreamingResponse(encoded(), media_type='text/event-stream')

    print('loopback-only teaching server; execution_mode=' + mode, flush=True)
    uvicorn.run(app, host=c['host'], port=c['port'], access_log=False)


if __name__ == '__main__':
    main()

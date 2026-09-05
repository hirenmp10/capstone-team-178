"""Resident LLM worker for Qwen 2.5-3B-Instruct.
Runs in its OWN interpreter (e.g. dedicated Python 3.11 virtualenv with torch/transformers),
never inside Isaac Sim.

Listens on 127.0.0.1:5557 by default.
Protocol:
  Request : JSON string terminated by newline: {"prompt": "<string>"}
  Response: JSON string terminated by newline: {"text": "<string>", "duration_s": <float>}
"""

import argparse
import json
import socket
import socketserver
import sys
import time
from pathlib import Path

DEFAULT_PORT = 5557
DEFAULT_MODEL = "Qwen/Qwen2.5-3B-Instruct"

class LlmService:
    def __init__(self, model_id: str, device: str = "cuda"):
        self.model_id = model_id
        self.device = device
        self.tokenizer = None
        self.model = None

    def load(self):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        print(f"  [LLM Worker] Loading tokenizer from {self.model_id}...")
        t0 = time.time()
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)

        print(f"  [LLM Worker] Loading model on {self.device}...")
        try:
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_id,
                dtype=torch.float16 if self.device == "cuda" else torch.float32,
                device_map=self.device
            )
            self.model.eval()
            print(f"  [LLM Worker] Ready on {self.device} in {time.time() - t0:.2f}s")
        except Exception as exc:
            if self.device == "cuda":
                print(f"  [LLM Worker] CUDA load failed ({exc}); falling back to CPU...")
                self.device = "cpu"
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.model_id,
                    dtype=torch.float32,
                    device_map="cpu"
                )
                self.model.eval()
                print(f"  [LLM Worker] Ready on CPU in {time.time() - t0:.2f}s")
            else:
                raise

    def complete(self, prompt: str) -> str:
        import torch

        messages = [
            {"role": "system", "content": "You are a precise robot command parser. Reply only with a JSON object."},
            {"role": "user", "content": prompt}
        ]
        text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.tokenizer([text], return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=128,
                temperature=0.01,
                pad_token_id=self.tokenizer.eos_token_id
            )
        generated_ids = outputs[0][inputs["input_ids"].shape[1]:]
        return self.tokenizer.decode(generated_ids, skip_special_tokens=True).strip()

_service = None

class _Handler(socketserver.StreamRequestHandler):
    def handle(self):
        global _service
        for raw_line in self.rfile:
            line = raw_line.decode("utf-8").strip()
            if not line:
                continue
            try:
                data = json.loads(line)
                prompt = data.get("prompt", "")
                t0 = time.time()
                completion = _service.complete(prompt)
                duration = time.time() - t0
                response = {"ok": True, "text": completion, "duration_s": duration}
            except Exception as exc:
                response = {"ok": False, "error": str(exc), "text": ""}
            out = json.dumps(response) + "\n"
            self.wfile.write(out.encode("utf-8"))
            self.wfile.flush()

class _ThreadedServer(socketserver.ThreadingMixIn, socketserver.TCPServer):
    allow_reuse_address = True

def main():
    global _service
    parser = argparse.ArgumentParser(description="Resident Qwen LLM Server")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    args = parser.parse_args()

    _service = LlmService(model_id=args.model, device=args.device)
    _service.load()

    server = _ThreadedServer((args.host, args.port), _Handler)
    print(f"  [LLM Worker] Serving at {args.host}:{args.port}")
    sys.stdout.flush()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n  [LLM Worker] Shutting down...")
    finally:
        server.server_close()

if __name__ == "__main__":
    main()

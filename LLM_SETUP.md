# Optional: Local LLM Setup (Question Shortage Fallback)

This is **completely optional**. The backend works fine without it — if no LLM server is detected on startup, it automatically falls back to static, difficulty-matched questions with zero crashes or delays.

Set this up only if you want AI-generated fallback questions when the local question pool runs out for a given difficulty tier.

## What's used

- **Model:** Llama 3.2, 1B parameters, Q4_K_M quantization, GGUF format
- **Runtime:** `llama-cpp-python` with its built-in OpenAI-compatible server
- **Port:** 3000

## 1. Install llama-cpp-python with server support

```bash
pip install llama-cpp-python[server]
```

If you want GPU acceleration (CUDA), reinstall with the relevant build flag instead:

```bash
CMAKE_ARGS="-DGGML_CUDA=on" pip install llama-cpp-python[server] --force-reinstall --no-cache-dir
```

CPU-only is fine for a 1B Q4_K_M model — it's small enough to run comfortably without a GPU.

## 2. Get the model

Download a Llama 3.2 1B Instruct GGUF, Q4_K_M quant, from Hugging Face. Any repo distributing this exact quant works (search `Llama-3.2-1B-Instruct-GGUF`). You want the file named something like:

```
Llama-3.2-1B-Instruct-Q4_K_M.gguf
```

Place it somewhere convenient, e.g. `./models/Llama-3.2-1B-Instruct-Q4_K_M.gguf`.

Alternatively, skip the manual download and let `llama-cpp-python` pull it directly from Hugging Face:

```python
from llama_cpp import Llama

llm = Llama.from_pretrained(
    repo_id="<hf-repo-with-this-gguf>",
    filename="*Q4_K_M.gguf"
)
```

(Requires `pip install huggingface-hub` as well.)

## 3. Run the server on port 3000

```bash
python3 -m llama_cpp.server \
  --model ./models/Llama-3.2-1B-Instruct-Q4_K_M.gguf \
  --port 3000 \
  --n_ctx 2048 \
  --chat_format chatml
```

Adjust `--chat_format` if the specific GGUF you downloaded expects a different template (check the model card — Llama 3.2 instruct models are usually fine with the `llama-3` or `chatml` chat format depending on quant source).

Once running, you should be able to confirm it's live with:

```bash
curl http://localhost:3000/v1/models
```

This is exactly the `base_url="http://localhost:3000/v1"` the backend's `generate_shortage_question` function expects — no code changes needed on the backend side once the server is up.

## 4. Verify the backend picks it up

On backend startup, you should see either:

```
(no message — LLM ping succeeded silently)
```

or, if the server isn't running:

```
No local LLM detected at startup — fallback question generation disabled, using static fallback instead.
```

If you see the second message but expect the LLM to be available, double check the server is actually listening on port 3000 and that nothing else (firewall, another process) is blocking it.

## Notes

- This is checked **once** at backend startup, not on every request — so starting the LLM server after the backend has already booted won't be picked up until the backend restarts.
- Lower-end machines: 1B Q4_K_M should run fine on CPU with a few threads; you generally won't need a GPU for this scale.
- If you swap in a different model size or quant, response time and quality will change accordingly — 1B is chosen here specifically because it's small and fast enough to act as a quick fallback, not as the primary scoring mechanism.

# Local Docker setup for HiDream-O1-Image

This local setup runs the official HiDream-O1 Flask app separately from the existing ComfyUI / HiDream I1 app.

## Model choice

The machine has an RTX 5090 with 32 GB VRAM. The default local service uses:

- `HiDream-ai/HiDream-O1-Image-Dev`
- `model_type=dev`
- 28 inference steps from the official app
- Port `7861`

The full model is also supported by the repo, but the dev model is the safer first target on 32 GB VRAM.

## Verified locally

- Container: `hidream-o1-image`
- URL: `http://127.0.0.1:7861`
- Smoke tests: text-to-image completed successfully through the Flask API.
- Output size: the official pipeline snapped 512/1024 requests to `2048x2048`.
- Timing observed: about 35 seconds for a 28-step `2048x2048` Dev generation while the existing ComfyUI container was still running.
- Prompt optimization: Japanese prompts are automatically rewritten through local Ollama into O1-friendly English before generation.

The official O1 code only snaps to predefined large resolutions such as `2048x2048`, `2304x1728`, and `1728x2304`. Treat `2048x2048` as the practical local baseline for this environment.

## Automatic prompt optimization

The Flask app optimizes the user's raw prompt before calling HiDream-O1:

1. The web UI sends the original prompt to `/api/generate/start`.
2. The server first calls the OpenAI-compatible refiner configured by `OPENAI_BASE_URL`, `OPENAI_MODEL`, and `OPENAI_API_KEY`.
3. If the OpenAI-compatible refiner is unavailable or rate-limited, the server falls back to Ollama at `OLLAMA_URL` with `OLLAMA_MODEL`.
4. The optimizer translates non-English prompts and rewrites them into a single English paragraph for HiDream-O1.
5. Anime, manga, bishoujo game, visual novel, and illustration requests are explicitly anchored as `2D Japanese visual novel game CG illustration, not photorealistic, not a real-life photo`.
6. The optimized English prompt is sent to O1, and the UI shows it under `Prompt sent to HiDream-O1`.

During generation, the UI shows the active process:

- Japanese prompt analysis and English refinement
- Send optimized prompt to HiDream-O1
- Generate image
- Return result to web app

The UI also includes style preference sliders. They are passed to the automatic prompt optimizer before each generation:

- Anime / visual novel strength
- Clean line art and detail
- Color vividness
- Background mood
- Photorealism avoidance

These sliders guide the English prompt sent to HiDream-O1, rather than changing the model checkpoint itself.

## Image editing

After an image is generated, use the `AIに修正を依頼` panel to edit the current image in a conversational flow. The latest generated image is sent back to HiDream-O1 as a reference image with `mode=edit`, and the user's Japanese edit instruction is automatically refined into an O1-friendly English edit prompt.

The edited image becomes the new current image, so it can be edited again or saved to R2.

## Cloudflare R2 storage

The app can save generated images to Cloudflare R2 with the `R2に画像を保存` button. It uploads:

- The PNG image
- A JSON metadata file containing the original prompt, optimized prompt, refiner source, style settings, and saved object keys

Bucket:

```text
hidream-o1-generated-images
```

Object key format:

```text
generated/YYYY/MM/DD/<uuid>.png
generated/YYYY/MM/DD/<uuid>.json
```

R2 credentials are stored only in the local `.env` file used by Docker Compose. They are not copied into the Docker image, and they are not rendered into the web page.

Environment defaults:

```text
AUTO_PROMPT_OPTIMIZE=1
OLLAMA_URL=http://host.docker.internal:11434
OLLAMA_MODEL=qwen3.5:9b
```

## Prompt Refiner via OpenRouter

The `Prompt Refiner` panel's `OpenAI-compatible API` backend is configured to use OpenRouter:

```text
OPENAI_BASE_URL=https://openrouter.ai/api/v1
OPENAI_MODEL=google/gemma-4-31b-it:free
OPENAI_API_KEY=${OPENROUTER_API_KEY}
```

The API key is stored only in the local `.env` file used by Docker Compose. It is not copied into the Docker image, and the web page does not render the key into the password field. Leave the API key field blank in the UI to use the server-side OpenRouter setting.

The same OpenRouter setting is also used automatically by the main `Generate` button before every O1 generation. OpenRouter free models can return HTTP 429 when rate-limited; in that case the app automatically falls back to local Ollama prompt optimization so generation can continue.

## Build

```powershell
cd C:\Users\GT-1096D\Documents\Codex\2026-06-30\d\work\HiDream-O1-Image
docker compose -f docker-compose.o1.yml build
```

## Download model

```powershell
docker run --rm `
  -v "${PWD}\models-cache:/models" `
  hidream-o1-image:local `
  huggingface-cli download HiDream-ai/HiDream-O1-Image-Dev `
    --local-dir /models/HiDream-O1-Image-Dev
```

## Start

```powershell
docker compose -f docker-compose.o1.yml up -d
```

Open:

```text
http://127.0.0.1:7861
```

## Notes

- `models/pipeline.py` is patched to set `use_flash_attn=False`, matching the official README fallback when flash-attn is not installed.
- The official O1 app is independent from ComfyUI because O1 is a unified raw-pixel model and is not packaged as ComfyUI split diffusion/text-encoder/VAE files.
- ComfyUI / HiDream I1 and O1 can both stay up, but together they leave little free VRAM. Stop `comfyhidream-comfyui-1` before heavier O1 tests or before trying the full model.
- If you later want the full model, download `HiDream-ai/HiDream-O1-Image`, change `HIDREAM_MODEL_PATH` to `/models/HiDream-O1-Image`, and set `HIDREAM_MODEL_TYPE=full`.

## Useful commands

```powershell
docker ps --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}"
docker logs --tail 120 hidream-o1-image
docker compose -f docker-compose.o1.yml restart
docker stop comfyhidream-comfyui-1
```

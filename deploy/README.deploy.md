# HiDream-O1 Vast.ai deployment package

This package prepares the current HiDream-O1 app for:

- GitHub Container Registry (GHCR)
- Vast.ai GPU instances
- Cloudflare Worker reverse proxy
- Model download on each Vast.ai instance

The Docker image does not include HiDream model weights. The entrypoint downloads the model into `/models` at container startup when missing.

## Recommended GPU target

Default model:

```text
HiDream-ai/HiDream-O1-Image-Dev
```

Recommended instance class:

- RTX PRO 4500 32GB
- A40 48GB
- RTX 5090 32GB
- Tesla V100 32GB, with `HIDREAM_TORCH_DTYPE=float16` or `auto`

Default settings are tuned for the Dev model and 32GB-ish VRAM. Full model is not the default for Vast.ai.

## Files

```text
Dockerfile
requirements-docker.txt
deploy/vast/entrypoint.sh
deploy/vast/env.vast.example
deploy/vast/run-vast.sh
deploy/vast/docker-compose.vast.yml
deploy/ghcr/build-and-push.ps1
deploy/cloudflare-worker/wrangler.toml.example
deploy/cloudflare-worker/src/index.ts
```

## Build and push to GHCR

From this directory:

```powershell
cd C:\Users\GT-1096D\Documents\Codex\2026-06-30\d\work\HiDream-O1-Image
```

Login:

```powershell
$env:GHCR_PAT = "github_pat_xxx"
$env:GHCR_PAT | docker login ghcr.io -u YOUR_GITHUB_USER --password-stdin
```

Build and push:

```powershell
.\deploy\ghcr\build-and-push.ps1 -Image ghcr.io/YOUR_GITHUB_OWNER/hidream-o1-image -Tag latest
```

The image is built from the current local base image and app files. Model files are excluded by `.dockerignore`.

## Vast.ai startup

On the Vast.ai machine:

```bash
mkdir -p /workspace/hidream-o1
cd /workspace/hidream-o1
```

Copy `deploy/vast/env.vast.example` to `env.vast` and fill secrets:

```bash
cp env.vast.example env.vast
nano env.vast
```

Run:

```bash
chmod +x run-vast.sh
./run-vast.sh ghcr.io/YOUR_GITHUB_OWNER/hidream-o1-image:latest
```

The first run downloads the model:

```text
HiDream-ai/HiDream-O1-Image-Dev -> /workspace/hidream-o1-models/HiDream-O1-Image-Dev
```

Watch logs:

```bash
docker logs -f hidream-o1-image
```

## Vast.ai Docker launch template

If using a Vast.ai launch template directly, use:

```text
Docker image: ghcr.io/YOUR_GITHUB_OWNER/hidream-o1-image:latest
Docker options:
  --gpus all -p 7861:7861 -v /workspace/hidream-o1-models:/models --env-file /workspace/hidream-o1/env.vast
```

Expose port:

```text
7861/tcp
```

## Environment variables

Important:

```text
HIDREAM_MODEL_REPO=HiDream-ai/HiDream-O1-Image-Dev
HIDREAM_MODEL_PATH=/models/HiDream-O1-Image-Dev
HIDREAM_MODEL_TYPE=dev
HIDREAM_TORCH_DTYPE=auto
```

For Tesla V100 32GB, use:

```text
HIDREAM_TORCH_DTYPE=float16
```

For RTX 5090 / RTX PRO 4500 / A40, use:

```text
HIDREAM_TORCH_DTYPE=auto
```

## Cloudflare Worker

The Worker is a reverse proxy in front of the Vast.ai backend.

Prepare:

```bash
cd deploy/cloudflare-worker
cp wrangler.toml.example wrangler.toml
```

Set:

```toml
BACKEND_URL = "https://YOUR_VAST_PUBLIC_URL"
```

Deploy:

```bash
npm install -g wrangler
wrangler deploy
```

When the Vast.ai instance changes, update `BACKEND_URL` and redeploy or use Wrangler environment variables.

## R2 storage

R2 bucket:

```text
hidream-o1-generated-images
```

The app uploads:

```text
generated/YYYY/MM/DD/<uuid>.png
generated/YYYY/MM/DD/<uuid>.json
```

Do not commit `.env`, `env.vast`, API keys, R2 access keys, or model files.

### R2 model cache for JANKU

For the `sdxl_janku` workflow, Civitai downloads can be unstable. Put the JANKU
checkpoint in a private R2 bucket and point Vast.ai at that object:

```text
JANKU_MODEL_PATH=/models/checkpoints/jankuV60.safetensors
JANKU_MODEL_URL=
JANKU_R2_BUCKET=hidream-o1-model-cache
JANKU_R2_KEY=models/jankuV60.safetensors
R2_ENDPOINT_URL=https://<account-id>.r2.cloudflarestorage.com
R2_ACCESS_KEY_ID=<r2-access-key>
R2_SECRET_ACCESS_KEY=<r2-secret-key>
```

When `JANKU_R2_BUCKET` and `JANKU_R2_KEY` are set, the entrypoint downloads JANKU
from R2 instead of Civitai. Keep the bucket in Standard storage if you want it to
count toward the R2 free tier.

## Notes

- OpenRouter free models may return HTTP 429. The app falls back to local Ollama if configured.
- If no Ollama exists on Vast.ai, OpenRouter should be the primary prompt refiner.
- Model download is about 35GB for the Dev model.
- V100 may need lower concurrency and `float16`.

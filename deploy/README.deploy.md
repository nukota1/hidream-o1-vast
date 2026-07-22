# Deployment

## Vast.ai template

```text
Image: nukota0615/hidream-o1-image:latest
Launch mode: Docker ENTRYPOINT
Container disk: 120GB or more for the default editor
Internal port: 7861 (Vast.ai assigns the external port after startup)
```

Set the variables from `vast/env.vast.example`. Required secret values are:

```text
R2_ENDPOINT_URL
R2_ACCESS_KEY_ID
R2_SECRET_ACCESS_KEY
```

The same R2 credentials currently access both `ai-model-cache` and the generated
image bucket. Separate least-privilege credentials can be introduced before public
production launch.

## Model lifecycle

The published image extends Vast.ai's pre-cached CUDA base image. Only the
application source is pulled when renting a new instance. On a fresh Vast.ai
disk, the entrypoint then installs CUDA PyTorch and Python dependencies. This
can take several minutes, but runs after the container has started and is
visible in the instance logs. It is skipped after the dependencies are
installed on the disk.

The container does not contain model weights. After the Python setup, it begins
background prefetch for:

1. JANKU v7.77 from private R2
2. Waifu-Inpaint-XL, the default editor, from Hugging Face
3. Qwen3.5 prompt refiner from Hugging Face

FLUX.1-Kontext-dev and HiDream-O1-Image are downloaded only after a user
selects the corresponding editor in the web UI. Allocate additional disk before
using them. The Hugging Face account must accept required model conditions and
the instance must have a read token in `HF_TOKEN`.

Models are stored under `/models`. Mount a persistent volume there when the Vast
host supports it; otherwise they are downloaded for every new instance.

## Cloudflare Worker

`cloudflare-worker/src/index.ts` proxies requests to the current Vast public URL.
Set `BACKEND_URL` to the URL without its query string. If Vast issues a browser
URL containing `?token=...`, store only that value in the Worker secret
`BACKEND_TOKEN`; the Worker adds it to upstream requests without exposing it to
the browser. Set `ALLOWED_ORIGIN` to the Worker URL or production custom domain.

R2 access remains in the Vast container through its S3-compatible environment
variables. The current Worker does not have an R2 or D1 binding.

For a Worker created through the Cloudflare dashboard's single-file editor, copy
`cloudflare-worker/worker.js` into the dashboard's `worker.js` file instead of
the TypeScript source file.

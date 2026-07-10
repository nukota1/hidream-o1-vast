# Deployment

## Vast.ai template

```text
Image: ghcr.io/nukota1/hidream-o1-image:latest
Launch mode: Docker ENTRYPOINT
Container disk: 120GB or more
Port: 7861
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

The container does not contain model weights. At startup it begins background
prefetch for:

1. JANKU v7.77 from private R2
2. Qwen Image Edit from Hugging Face
3. Qwen3.5 prompt refiner from Hugging Face

Models are stored under `/models`. Mount a persistent volume there when the Vast
host supports it; otherwise they are downloaded for every new instance.

## Cloudflare Worker

`cloudflare-worker/src/index.ts` proxies requests to the current Vast public URL.
Set `BACKEND_URL` in Wrangler and deploy the Worker. User authentication and rate
limiting are not implemented yet.

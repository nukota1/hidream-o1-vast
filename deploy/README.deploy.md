# Deployment

## Vast.ai template

```text
Image: nukota0615/hidream-o1-image:<full-git-commit-sha>
Launch mode: Docker ENTRYPOINT
Container disk: 120GB
Internal port: 7861 (Vast.ai assigns the external port after startup)
```

Use a full SHA tag instead of `latest`, then set the variables from
`vast/env.vast.example`. The only secret required for the Worker-to-Vast path is:

```text
BACKEND_SHARED_SECRET
```

Set the same value as a Cloudflare Worker secret. The Worker stores generated
images through its R2 binding, so the Vast container does not need a Cloudflare
API token or R2 S3 credentials. `R2_*` is optional and only enables the backend's
manual `R2に保存` endpoint.

## Model lifecycle

The published image extends Vast.ai's pre-cached CUDA base image. Only the
application source is pulled when renting a new instance. On a fresh Vast.ai
disk, the entrypoint then installs CUDA PyTorch and Python dependencies. This
can take several minutes, but runs after the container has started and is
visible in the instance logs. It is skipped after the dependencies are
installed on the disk.

The container does not contain model weights. After the Python setup, it begins
background prefetch only for the models used by the current workflow:

1. Animagine XL 4.0 Zero from Hugging Face
2. SDXL IP-Adapter Plus and its image encoder from Hugging Face
3. Qwen3.5-9B prompt refiner from Hugging Face

The standard template does not configure JANKU, Waifu-Inpaint-XL, anime
segmentation, FLUX.1-Kontext-dev, Qwen-Image-Edit, or HiDream-O1-Image.
HiDream source and its extra Python dependencies are also skipped unless
`HIDREAM_RUNTIME_SETUP_ON_START=1` is explicitly set. `HF_TOKEN` is optional for
higher Hugging Face download limits.

Models are stored under `/models`. Mount a persistent volume there when the Vast
host supports it; otherwise they are downloaded for every new instance.

Character and Style LoRA files are user assets rather than base models. They are
not embedded in the public image or downloaded by the standard template. Copy or
train compatible `sdxl-animagine-zero` LoRAs under `LORA_ROOT` separately.

## Cloudflare Worker and authentication

The Worker owns the gallery storage API. Generated PNG files are stored in
Cloudflare R2 and gallery records, folders, and favorite prompt groups are stored
in D1. Records are scoped by the authenticated Cloudflare Access user ID.
The Worker proxies image-generation requests to Vast only after Access JWT
validation and adds a private `X-Backend-Key` header.

Create the resources once from `deploy/cloudflare-worker`:

```powershell
npm install
npx wrangler r2 bucket create hidream-o1-generated-images
npx wrangler d1 create janku-image-studio
```

Copy the returned D1 `database_id` into `wrangler.toml`. Keep `BACKEND_URL` empty
until the Vast endpoint is known. Apply the schema and deploy:

```powershell
npx wrangler d1 migrations apply janku-image-studio --remote
npx wrangler deploy
```

The R2/D1 bindings are named `GALLERY_BUCKET` and `DB`. Do not put API keys,
`BACKEND_SHARED_SECRET`, Google OAuth secrets, or Access credentials in
`wrangler.toml` or GitHub. Once the bindings exist, the Worker runtime does not
need the Cloudflare API token or the R2 S3 secret.

### Google account login

Configure Cloudflare Access on the Worker before enabling its `workers.dev`
route. Add Google as an identity provider, create an external Google OAuth
client, and use an Access Allow policy with `Include: Everyone` plus
`Require: Login methods -> Google`. This permits any Google account to sign in;
it does not expose the Worker before authentication. Set the resulting
`ACCESS_TEAM_DOMAIN` and `ACCESS_AUD` values as Worker variables. The Worker
fails closed with HTTP 503 until both values are present.

After Access is configured, create a long random `BACKEND_SHARED_SECRET` as a
Worker secret and set the same value as the Vast `BACKEND_SHARED_SECRET`
environment variable. The Flask backend rejects direct requests without this
header. The current project intentionally leaves `BACKEND_URL` empty until the
Vast URL is provided.

When the UI is opened directly at `127.0.0.1`, it keeps using IndexedDB as a
development fallback. When opened through the deployed Worker after Access
login, the UI detects `/api/gallery/health` and automatically uses the
authenticated user's R2/D1 data. Images are uploaded after each successful
generation, so the gallery no longer depends on the manual save button.

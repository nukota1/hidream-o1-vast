# Deployment

## Vast.ai template

```text
Image: nukota0615/hidream-o1-image:<full-git-commit-sha>
Launch mode: Docker ENTRYPOINT
Container disk: 120GB
Internal port: 7861 (Vast.ai assigns the external port after startup)
```

Use a full SHA tag instead of `latest`, then set the variables from
`vast/env.vast.example`. The Worker-to-Vast proxy itself uses:

```text
BACKEND_SHARED_SECRET
```

Set the same value as a Cloudflare Worker secret. The Worker stores generated
gallery images through its own R2 binding, so that path does not need R2 S3
credentials in Vast. Per-user LoRA persistence is different: the Vast backend
uploads and restores large LoRA artifacts directly and therefore needs the
bucket-scoped `LORA_R2_*` S3 credentials from `vast/env.vast.example`.

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
not embedded in the public image. With `LORA_R2_SYNC_ENABLED=1`, the backend
stores ready LoRAs under the authenticated user's pseudonymous R2 prefix after
training and lazily restores only that user's assets when the LoRA list or a
generation request needs them. `/models/loras` remains a local cache.

The default remote objects are `metadata.json`, the inference weights, and the
small training configuration. Intermediate checkpoints are also excluded unless
`LORA_R2_INCLUDE_CHECKPOINTS=1` is selected. Raw source images and captions remain local unless
`LORA_R2_INCLUDE_TRAINING_DATA=1` is explicitly selected. Use the matching
restore flag only when retraining data must be recovered on a new instance.

To publish an existing local compatible LoRA after local verification:

```powershell
python scripts/sync_lora_r2.py publish --owner-id local --model-id <model-id>
```

For a one-time migration from the local development owner to a signed-in cloud
user, read that user's `remote_storage.owner_key` from `GET /api/lora/models`,
then add `--remote-owner-key <owner-key>`. The command never places the raw
Cloudflare user ID in an R2 object key.

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

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

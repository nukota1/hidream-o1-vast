# Animagine Image Studio

Japanese web application for anime image generation. It runs locally with Docker
for development and on a rented GPU server for production.

## Runtime

- Local and Vast.ai text-to-image: Animagine XL 4.0 Zero / SDXL
- Character reference: h94/IP-Adapter Plus for SDXL
- Prompt refinement: Qwen/Qwen3.5-9B on GPU, optional per request
- Character and style training: separate SDXL LoRAs on Animagine XL 4.0 Zero
- Generated image storage: private Cloudflare R2
- Backend and UI: Flask
- Public gateway: Cloudflare Worker reverse proxy
- Container registry: GitHub Container Registry

The standard local and Vast startup does not provision JANKU, Waifu-Inpaint-XL,
anime segmentation, FLUX.1-Kontext-dev, Qwen-Image-Edit, or HiDream-O1-Image.
Their legacy/optional code paths remain available for explicit future opt-in.

## Local development (RTX 5090)

Use the local Docker Compose configuration for normal development and testing.
It uses the same application image as production, exposes the app at
`http://127.0.0.1:7861`, and keeps all downloaded models in the named Docker
volume `janku-models-local`. Recreating the container does not download models
again. Python/CUDA packages persist in `janku-python-local`, so dependency
installation is also skipped after the first successful setup.

IP-Adapter Plus is loaded into the active SDXL pipeline only when a reference
image is used. Reserve 120GB of disk for the base image, Python/CUDA runtime,
Animagine checkpoint, prompt refiner, reference adapter, generated files, and
operational headroom.

1. Start Docker Desktop.
2. Create `.env` from `.env.example` and set the R2 credentials.
3. Create `.env.local` from `.env.local.example` to change non-secret local
   settings when needed.
4. Run the following in PowerShell:

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\scripts\local.ps1 start
```

The first local startup downloads Animagine XL 4.0 Zero, Qwen3.5-9B, and the
SDXL IP-Adapter Plus reference model. It starts the web app
immediately while these models are cached in the background. Check readiness
and download progress with:

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\scripts\local.ps1 logs
```

Other commands are `stop`, `restart`, `status`, and `build`.

## Animagine XL 4.0 Zero defaults

Animagine XL 4.0 Zero is the LoRA/fine-tuning base published by CagliostroLab.
Local and Vast.ai use the same exact model profile so a LoRA is never silently
applied to a different checkpoint. It uses ordered Danbooru tags and the
application uses the officially recommended settings:

```text
Sampler: Euler A
Steps: 28
CFG: 5.0
Clip Skip: 2
```

The local Qwen3.5-9B refiner translates Japanese into Animagine-compatible
ordered tags and appends `masterpiece, high score, great score, absurdres`.
Free-form text is refined first and kept ahead of prompt-catalog tags. The
standard PyTorch execution path is used; optional Hub activation kernels are
disabled because their current builds are not ABI-compatible with this
project's Torch 2.7.1 runtime.

The web UI provides presets for bishoujo visual-novel CG, anime illustration,
manga, light-novel illustration, and custom settings. Every value can be changed
under `詳細設定`.

Japanese input is always translated into compact Illustrious/Danbooru tags.
Turning off `構図と表現をAIで補完` disables inferred embellishment, but does
not disable the required Japanese-to-English tag conversion.

## Vast.ai

Use `Docker ENTRYPOINT` launch mode and expose port `7861`.

Use 120GB for Vast.ai `Disk Space (Container + Volume)` with the current model
set and use a full Git SHA image tag.

```text
nukota0615/hidream-o1-image:<full-git-commit-sha>
```

Copy the variables from `deploy/vast/env.vast.example` into the Vast.ai template.
Do not put model weights or secrets in Git.

Only Animagine XL 4.0 Zero, IP-Adapter Plus, its image encoder, and the Qwen
prompt refiner are prefetched. HiDream source/dependency setup is disabled by
default as well as all unused model prefetches. Cloudflare API and R2 S3
credentials are not required in Vast when gallery storage goes through the
Worker binding.

## Image editing models

The standard story-illustration workflow separates `キャラクターの要素` from
`背景・シーン`, refines them independently, and then places identity tags ahead
of scene and style tags in one full-frame SDXL generation. It never creates a
silhouette mask, so there is no chroma fringe or segmentation seam.

The composition panel offers two workflows:

1. **Consistent full-frame regeneration**: the default ADV/event-CG workflow.
   The source image is passed through SDXL IP-Adapter Plus. A selected Character
   LoRA supplies repeatable identity, a separate Style LoRA supplies reusable
   line and rendering style, and the character definition is ordered
   before the requested expression, pose, camera, and background. The reference
   strength is adjustable: lower values permit larger pose/composition changes;
   higher values retain more of the source appearance.
2. **Manual-mask localized correction**: Waifu-Inpaint-XL edits only the white
   area of a user-supplied mask. Automatic silhouette extraction is deliberately
   not used. This is an advanced repair tool, not the background-change path.

Waifu-Inpaint-XL, FLUX.1-Kontext-dev, and HiDream-O1-Image remain optional
backends in the server code, but the current Vast template does not provision
them. Enabling one later requires an explicit environment/configuration change
and additional disk planning.

## Character and Style LoRA training

Open the gallery and right-click, long-press, or use a card's menu button, then
choose `このイラストで追加学習する`. The training dialog accepts:

- LoRA name and trigger word
- a fixed character definition and fixed negative traits for immutable face,
  hair, eye, and body traits
- character, style, pose, or background category
- the active SDXL base-model type
- selected gallery images and additional uploaded images
- automatic or explicit training steps

One-image training is available for experiments, but ten or more images of the
same character with varied angles, expressions, poses, and lighting are
recommended. Training runs as an exclusive GPU job and reports progress over
SSE. Datasets, metadata, and final weights are stored per user under
`/models/loras`. A completed LoRA appears in its matching Character or Style
selector. Both can be active with independent weights; their trigger words,
exact base-model profile, and weights are recorded with the generated asset.
Former Animagine Opt and JANKU LoRAs are marked incompatible instead of being
loaded on Zero.

The fixed character definition accepts Japanese and is stored with the LoRA.
When that LoRA is selected, the definition is automatically merged ahead of
the scene and optional style tags so users do not need to repeat identity
details for every image.

Per-image sidecar captions or `captions.json` separate clothing, background,
pose, and expression from character identity. Portrait inputs use
aspect-ratio-preserving 64-pixel buckets instead of a square center crop, and
character training does not randomly mirror asymmetric features. The automatic
step recommendation is `image count * 20`, clamped to 200-800 steps.

Style LoRA uses a separate policy: prepare multiple characters, clothes, poses,
backgrounds, and compositions that share one visual style. The UI recommends
at least 50 images, uses rank 32, learning rate `5e-5`, horizontal flip, and a
starting inference weight of 0.6. A reusable Style LoRA should not be trained
from only one character because identity and style would become entangled.

If a dataset constant such as a white background leaks into generations, record
it in the model's `training_leakage_tags`; the app adds unrequested constants to
the negative prompt. Use `scripts/evaluate_character_lora.py` for deterministic
baseline/checkpoint comparisons before registering a weight.

LoRA training and instant reference conditioning remain separate assets but are
combined at inference time: Character LoRA provides long-term identity, Style
LoRA provides reusable rendering style, and IP-Adapter uses the selected source
image as appearance guidance for the current generation. Gallery images can be
sent directly to the consistency-regeneration screen.

## Storage

When accessed through the Cloudflare Worker, generated images are saved
automatically to private R2 and the gallery metadata is saved to D1:

```text
R2: gallery/<uuid>.png
D1: gallery_records, gallery_folders, favorite_groups
```

The D1 record contains the original prompt, final prompt, selected preset,
refinement state, generation settings, workflow, and folder. The PNG is streamed
through the Worker at `/api/gallery/image/<uuid>`. Direct local access keeps an
IndexedDB fallback for development.

## Build

Pull requests to `main` trigger a container build without publishing. Pushing
`main` triggers the same workflow and publishes both `latest` and commit-SHA
tags to GHCR and Docker Hub.

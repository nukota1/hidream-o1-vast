import base64
import hashlib
import io
import json
import os
import re
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path

from PIL import Image


LORA_CATEGORIES = {"character", "style", "pose", "background"}
LORA_MODEL_TYPES = {
    "sdxl-animagine-zero",
    "sdxl-animagine-opt",
    "sdxl-janku-v777",
}
LEGACY_MODEL_TYPES = {
    "sdxl-animagine": "sdxl-animagine-opt",
    "sdxl-janku": "sdxl-janku-v777",
}
MODEL_TYPE_LABELS = {
    "sdxl-animagine-zero": "SDXL / Animagine XL 4.0 Zero",
    "sdxl-animagine-opt": "SDXL / Animagine XL 4.0 Opt",
    "sdxl-janku-v777": "SDXL / JANKU v7.77",
}
TRIGGER_PATTERN = re.compile(r"^[A-Za-z][A-Za-z0-9_-]{2,63}$")
MAX_LORA_IMAGES_BY_CATEGORY = {
    "character": 100,
    "style": 150,
    "pose": 100,
    "background": 150,
}
RECOMMENDED_IMAGE_COUNTS = {
    "character": 10,
    "style": 50,
    "pose": 20,
    "background": 30,
}
MAX_IMAGE_BYTES = 12 * 1024 * 1024


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def current_model_type():
    configured = os.environ.get("IMAGE_MODEL_PROFILE", "").strip().lower()
    if configured:
        canonical = LEGACY_MODEL_TYPES.get(configured, configured)
        if canonical not in LORA_MODEL_TYPES:
            raise RuntimeError(f"Unsupported IMAGE_MODEL_PROFILE: {configured}")
        return canonical
    family = os.environ.get("IMAGE_MODEL_FAMILY", "janku").strip().lower()
    if family == "animagine":
        identity = " ".join([
            os.environ.get("ANIMAGINE_MODEL_REPO", ""),
            os.environ.get("ANIMAGINE_MODEL_PATH", ""),
        ]).lower()
        return (
            "sdxl-animagine-zero"
            if "zero" in identity
            else "sdxl-animagine-opt"
        )
    return "sdxl-janku-v777"


def canonical_model_type(model_type):
    value = str(model_type or "").strip().lower()
    return LEGACY_MODEL_TYPES.get(value, value)


def model_type_label(model_type=None):
    canonical = canonical_model_type(model_type or current_model_type())
    return MODEL_TYPE_LABELS.get(canonical, canonical)


def is_lora_compatible(metadata):
    return canonical_model_type(metadata.get("model_type")) == current_model_type()


def owner_storage_key(owner_id):
    owner = (owner_id or "local").strip() or "local"
    if owner == "local":
        return "local"
    return hashlib.sha256(owner.encode("utf-8")).hexdigest()[:32]


def _atomic_json_write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_suffix(f"{path.suffix}.{uuid.uuid4().hex}.tmp")
    temp_path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    temp_path.replace(path)


def _decode_image(value):
    encoded = str(value or "").strip()
    if encoded.startswith("data:image"):
        encoded = encoded.split(",", 1)[1]
    try:
        raw = base64.b64decode(encoded, validate=True)
    except Exception as exc:
        raise ValueError("学習画像のBase64データが壊れています。") from exc
    if not raw or len(raw) > MAX_IMAGE_BYTES:
        raise ValueError("学習画像は1枚あたり12MB以下にしてください。")
    try:
        loaded = Image.open(io.BytesIO(raw))
        loaded.load()
    except Exception as exc:
        raise ValueError("学習画像を読み込めませんでした。") from exc
    if min(loaded.size) < 384:
        raise ValueError("学習画像の短辺は384px以上にしてください。")
    return loaded


def _flatten_training_image(image):
    rgba = image.convert("RGBA")
    background = Image.new("RGBA", rgba.size, (240, 240, 240, 255))
    background.alpha_composite(rgba)
    return background.convert("RGB")


def validate_training_request(data):
    name = str(data.get("name") or "").strip()
    trigger_word = str(data.get("trigger_word") or "").strip()
    identity_prompt = str(data.get("identity_prompt") or "").strip()
    identity_negative_prompt = str(
        data.get("identity_negative_prompt") or ""
    ).strip()
    category = str(data.get("category") or "character").strip().lower()
    model_type = canonical_model_type(
        data.get("model_type") or current_model_type()
    )
    images = data.get("images")
    if not name or len(name) > 80:
        raise ValueError("LoRA名は1文字以上80文字以下で入力してください。")
    if not TRIGGER_PATTERN.fullmatch(trigger_word):
        raise ValueError(
            "トリガーワードは英字で始まる3〜64文字の英数字、_、-で入力してください。"
        )
    if len(identity_prompt) > 2000:
        raise ValueError("人物の固定定義は2000文字以下で入力してください。")
    if len(identity_negative_prompt) > 1000:
        raise ValueError("人物の固定除外条件は1000文字以下で入力してください。")
    if category not in LORA_CATEGORIES:
        raise ValueError("未対応のLoRAカテゴリです。")
    if model_type not in LORA_MODEL_TYPES:
        raise ValueError("未対応のモデル種別です。")
    if model_type != current_model_type():
        raise ValueError(
            "現在起動中の基盤モデルと異なる種別では学習できません。"
        )
    if not isinstance(images, list) or not images:
        raise ValueError("学習画像を1枚以上選択してください。")
    max_images = MAX_LORA_IMAGES_BY_CATEGORY[category]
    if len(images) > max_images:
        raise ValueError(
            f"{category} LoRAの学習画像は最大{max_images}枚です。"
        )
    default_rank = 32 if category == "style" else 16
    rank = max(4, min(64, int(data.get("rank") or default_rank)))
    requested_steps = int(data.get("steps") or 0)
    if category == "style":
        recommended_steps = max(400, min(1600, len(images) * 10))
    else:
        recommended_steps = max(200, min(800, len(images) * 20))
    steps = max(50, min(2000, requested_steps or recommended_steps))
    resolution = max(512, min(1024, int(data.get("resolution") or 768)))
    resolution = int(round(resolution / 64.0) * 64)
    learning_rate = float(
        data.get("learning_rate")
        or (5e-5 if category == "style" else 1e-4)
    )
    learning_rate = max(1e-6, min(5e-4, learning_rate))
    return {
        "name": name,
        "trigger_word": trigger_word,
        "identity_prompt": identity_prompt,
        "identity_negative_prompt": identity_negative_prompt,
        "category": category,
        "model_type": model_type,
        "images": images,
        "rank": rank,
        "steps": steps,
        "resolution": resolution,
        "learning_rate": learning_rate,
        "random_flip": category == "style",
        "recommended_weight": 0.6 if category == "style" else 0.8,
    }


class LoraStore:
    def __init__(self, root=None):
        self.root = Path(
            root or os.environ.get("LORA_ROOT", "/models/loras")
        ).resolve()

    def _owner_root(self, owner_id):
        return self.root / owner_storage_key(owner_id)

    def _model_root(self, owner_id, model_id):
        if not re.fullmatch(r"[a-f0-9]{32}", str(model_id or "")):
            raise KeyError("Unknown LoRA")
        return self._owner_root(owner_id) / model_id

    def model_root(self, owner_id, model_id):
        return self._model_root(owner_id, model_id)

    def create(self, owner_id, request_data):
        validated = validate_training_request(request_data)
        model_id = uuid.uuid4().hex
        model_root = self._model_root(owner_id, model_id)
        dataset_root = model_root / "dataset"
        dataset_root.mkdir(parents=True, exist_ok=False)
        captions = []
        for index, item in enumerate(validated["images"], start=1):
            image_data = item.get("image") if isinstance(item, dict) else item
            image = _flatten_training_image(_decode_image(image_data))
            image_path = dataset_root / f"{index:03d}.png"
            image.save(image_path, format="PNG", optimize=True)
            prompt = ""
            training_caption = ""
            source_id = ""
            if isinstance(item, dict):
                prompt = str(item.get("prompt") or "")[:2000]
                training_caption = str(
                    item.get("caption")
                    or item.get("optimized_prompt")
                    or prompt
                    or ""
                ).strip()[:2000]
                source_id = str(item.get("source_id") or "")[:200]
            if training_caption:
                image_path.with_suffix(".txt").write_text(
                    training_caption,
                    encoding="utf-8",
                )
            captions.append({
                "file": image_path.name,
                "source_id": source_id,
                "source_prompt": prompt,
                "training_caption": training_caption,
            })

        metadata = {
            "id": model_id,
            "name": validated["name"],
            "trigger_word": validated["trigger_word"],
            "identity_prompt": validated["identity_prompt"],
            "identity_negative_prompt": validated["identity_negative_prompt"],
            "category": validated["category"],
            "model_type": validated["model_type"],
            "status": "queued",
            "image_count": len(validated["images"]),
            "steps": validated["steps"],
            "rank": validated["rank"],
            "resolution": validated["resolution"],
            "learning_rate": validated["learning_rate"],
            "random_flip": validated["random_flip"],
            "recommended_weight": validated["recommended_weight"],
            "created_at": utc_now(),
            "updated_at": utc_now(),
            "progress": 0,
            "error": "",
            "captions": captions,
        }
        _atomic_json_write(model_root / "metadata.json", metadata)
        return metadata

    def read(self, owner_id, model_id):
        path = self._model_root(owner_id, model_id) / "metadata.json"
        if not path.is_file():
            raise KeyError("Unknown LoRA")
        return json.loads(path.read_text(encoding="utf-8"))

    def update(self, owner_id, model_id, **changes):
        metadata = self.read(owner_id, model_id)
        metadata.update(changes)
        metadata["updated_at"] = utc_now()
        _atomic_json_write(
            self._model_root(owner_id, model_id) / "metadata.json",
            metadata,
        )
        return metadata

    def write_metadata(self, owner_id, model_id, metadata):
        if str(metadata.get("id") or "") != model_id:
            raise ValueError("LoRA metadata ID does not match its storage path.")
        _atomic_json_write(
            self._model_root(owner_id, model_id) / "metadata.json",
            metadata,
        )
        return metadata

    def list(self, owner_id):
        owner_root = self._owner_root(owner_id)
        if not owner_root.is_dir():
            return []
        models = []
        for metadata_path in owner_root.glob("*/metadata.json"):
            try:
                models.append(json.loads(metadata_path.read_text(encoding="utf-8")))
            except (OSError, ValueError):
                continue
        models.sort(key=lambda item: item.get("created_at", ""), reverse=True)
        return models

    def dataset_path(self, owner_id, model_id):
        path = self._model_root(owner_id, model_id) / "dataset"
        if not path.is_dir():
            raise KeyError("Unknown LoRA dataset")
        return path

    def output_path(self, owner_id, model_id):
        return self._model_root(owner_id, model_id) / "output"

    def weight_path(self, owner_id, model_id):
        return self.output_path(owner_id, model_id) / "pytorch_lora_weights.safetensors"

    def public(self, metadata):
        return {
            key: value
            for key, value in metadata.items()
            if key not in {"captions", "remote_storage"}
        }

    def training_command(self, owner_id, model_id):
        metadata = self.read(owner_id, model_id)
        script_path = Path(__file__).with_name("scripts") / "train_character_lora.py"
        base_model = (
            os.environ.get(
                "ANIMAGINE_MODEL_PATH",
                "/models/checkpoints/animagine-xl-4.0-zero.safetensors",
            )
            if metadata["model_type"].startswith("sdxl-animagine-")
            else os.environ.get("JANKU_MODEL_PATH", "")
        )
        if not base_model:
            raise RuntimeError("LoRA学習用の基盤モデルが設定されていません。")
        command = [
            sys.executable,
            str(script_path),
            "--base-model",
            base_model,
            "--dataset",
            str(self.dataset_path(owner_id, model_id)),
            "--output",
            str(self.output_path(owner_id, model_id)),
            "--trigger-word",
            metadata["trigger_word"],
            "--category",
            metadata["category"],
            "--steps",
            str(metadata["steps"]),
            "--rank",
            str(metadata["rank"]),
            "--resolution",
            str(metadata["resolution"]),
            "--learning-rate",
            str(metadata.get("learning_rate") or 1e-4),
        ]
        if metadata.get("random_flip"):
            command.append("--random-flip")
        if metadata["model_type"].startswith("sdxl-animagine-"):
            command.extend([
                "--config",
                os.environ.get(
                    "ANIMAGINE_MODEL_CONFIG",
                    "cagliostrolab/animagine-xl-4.0-zero",
                ),
            ])
        return command

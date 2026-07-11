import json
import os
import re
import gc

import torch


SYSTEM_PROMPT = """
You are a prompt director for the JANKU v7.77 Illustrious XL anime image model.
Convert the user's Japanese request into one concise English prompt suitable for
anime image generation or image editing.

Rules:
- Preserve every requested subject, appearance detail, outfit, pose, camera angle,
  composition, weather, background, color, material, expression, and mood.
- Never invent unrelated characters, props, text, logos, brands, or locations.
- Put user-requested visual facts first. Put generic style terms last so they
  are discarded before requested appearance, outfit, pose, and background.
- Order the prompt by coverage, not by the order of the user's sentences:
  subject and composition first; then outfit and important props; then setting,
  weather, and background; then appearance details; generic quality/style last.
- Return canonical Danbooru/Illustrious XL tags only, never prose sentences.
  Use concrete tags such as `1girl`, `solo`, `full body`, `blonde hair`,
  `red eyes`, `schoolyard`, and `rain`; use one tag for each requested object,
  color, clothing item, pose, camera direction, and background element.
- Add `masterpiece, best quality, very aesthetic` only after all user-requested
  details, never before them.
- Japanese clothing glossary: `ワンピース` and `ロングワンピース` mean
  `dress` and `long dress`, never `swimsuit` unless the user explicitly says
  `水着`. `紐付きショートブーツ` means `lace-up ankle boots`.
- Return no more than 24 short comma-separated tags. Target 60 CLIP tokens or
  fewer so the image model does not truncate the request.
- For edits, preserve the same character identity, face, hairstyle, body
  proportions, pose, camera framing, and composition unless the user explicitly
  requests a change. State only the requested changes as changes.
- For edits, never add style, quality, rendering, line-art, color-treatment, or
  preset tags unless the user explicitly requests a style change. Preserve the
  source image's existing visual style.
- Honor the selected style preset and numerical style preferences.
- Return JSON only with keys "prompt" and "intent_notes".
""".strip()

RETRY_SYSTEM_PROMPT = """
Convert the request into one compact English SDXL image prompt.
Reply with one comma-separated line of canonical Danbooru/Illustrious XL tags
only: no JSON, no explanation, no markdown, and no prose sentences.
Keep the primary subject, requested appearance, outfit, pose, composition, and
background. Use at most 24 short English tags and 380 characters. For image
editing, retain the same character identity, face, hairstyle, body proportions,
pose, camera framing, and composition unless explicitly changed.
""".strip()

TRANSLATE_ONLY_SYSTEM_PROMPT = """
Translate the user's request into canonical English Danbooru/Illustrious XL
tags without adding new visual ideas. Preserve every requested subject,
appearance, color, outfit, pose, camera direction, object, weather condition,
and background element. Put user-requested facts first and the selected style
last. Return JSON only with keys "prompt" and "intent_notes". The prompt must
contain at most 24 short comma-separated tags and 380 characters.
Japanese glossary: `ワンピース` means `dress`, never `swimsuit`, unless `水着`
is explicitly present. `ロングワンピース` means `long dress`.
""".strip()

EDIT_SYSTEM_PROMPT = """
Convert the user's image-edit request into canonical English
Danbooru/Illustrious XL tags describing only the requested changed content.
Do not output source attributes that the user asks to preserve. Do not add
generic subject, pose, quality, style, rendering, or background tags unless
they are themselves being changed. For example, changing a dress should return
only tags such as `long dress, cyan dress, red and white floral pattern`.
Japanese glossary: `ワンピース` means `dress`, never `swimsuit`, unless `水着`
is explicitly present. Return JSON only with keys "prompt" and "intent_notes".
Use at most 16 short comma-separated tags.
""".strip()


class LocalPromptRefiner:
    def __init__(self, model_id=None):
        self.model_id = model_id or os.environ.get("PROMPT_REFINER_MODEL", "Qwen/Qwen3.5-9B")
        self.processor = None
        self.model = None
        self.device = None

    def _load(self):
        if self.model is not None:
            return

        from transformers import AutoModelForCausalLM, AutoProcessor

        print(f"[refine] Loading local prompt refiner: {self.model_id}")
        self.processor = AutoProcessor.from_pretrained(self.model_id, trust_remote_code=True)
        device = os.environ.get("PROMPT_REFINER_DEVICE", "cuda").lower()
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        model_kwargs = {"dtype": "auto", "trust_remote_code": True}
        if device == "cpu":
            model_kwargs["device_map"] = {"": "cpu"}
        elif device in {"cuda", "gpu"}:
            model_kwargs["device_map"] = "cuda"
            model_kwargs["dtype"] = torch.float16
            device = "cuda"
        else:
            model_kwargs["device_map"] = "auto"
        self.model = AutoModelForCausalLM.from_pretrained(self.model_id, **model_kwargs).eval()
        self.device = device

    def unload_if_cuda(self):
        if self.device != "cuda" or self.model is None:
            return
        model = self.model
        self.model = None
        self.processor = None
        self.device = None
        del model
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()

    @staticmethod
    def _tag_priority(tag):
        """Keep scene-defining facts ahead of fine details and style filler."""
        value = tag.lower()
        category_terms = (
            (0, ("1girl", "1boy", "girl", "boy", "solo", "full body", "upper body", "multiple girls")),
            (1, ("looking at viewer", "toward viewer", "through umbrella", "camera", "from ", "standing", "sitting", "holding", "pose")),
            (2, ("dress", "skirt", "shirt", "coat", "jacket", "boots", "shoes", "umbrella", "uniform", "outfit")),
            (3, ("rain", "snow", "schoolyard", "school yard", "puddle", "background", "outdoors", "indoors", "street", "classroom", "nostalgic", "wet")),
            (4, ("hair", "eyes", "eye", "skin", "legs", "body", "blush", "lips", "smile", "expression", "ribbon", "hair tie")),
            (6, ("masterpiece", "best quality", "aesthetic", "anime", "illustration", "line art", "colors", "photorealistic")),
        )
        for priority, terms in category_terms:
            if any(term in value for term in terms):
                return priority
        return 5

    @staticmethod
    def _within_category_priority(tag, category):
        value = tag.lower()
        preferred_terms = {
            0: ("1girl", "girl", "solo", "full body"),
            1: ("toward viewer", "facing camera", "through umbrella", "looking at viewer", "holding"),
            2: ("dress", "umbrella", "boots", "shoes"),
            3: ("schoolyard", "school yard", "rain", "puddle", "nostalgic", "wet"),
            4: ("blonde hair", "red eyes", "crimson", "wavy hair", "ribbon", "hair tie", "skin", "legs", "blush", "lips"),
        }
        for index, term in enumerate(preferred_terms.get(category, ())):
            if term in value:
                return index
        return 99

    @staticmethod
    def _compact_prompt(prompt):
        compact_replacements = (
            (r"delicate 6-heads tall girl", "1girl, slender"),
            (r"umbrella top facing (?:the )?camera", "umbrella toward viewer"),
            (r"light blue pastel long dress with red and white floral chirimashi pattern", "pastel blue floral long dress, red and white pattern"),
            (r"full body wet with dripping water droplets", "full body, soaking wet"),
            (r"rural school courtyard with puddles", "rural schoolyard, puddles"),
            (r"2d japanese bishoujo visual novel cg", "bishoujo game CG"),
            (r"crimson red eyes with glassy depth and multiple highlights", "glassy crimson eyes"),
            (r"detailed fluffy wavy hair", "fluffy wavy hair"),
        )
        for pattern, replacement in compact_replacements:
            prompt = re.sub(pattern, replacement, prompt, flags=re.IGNORECASE)
        raw_tags = [tag.strip() for tag in re.split(r"[,\n]", prompt) if tag.strip()]
        buckets = {priority: [] for priority in range(7)}
        limits = {0: 2, 1: 3, 2: 4, 3: 4, 4: 5, 5: 2, 6: 1}
        for tag in raw_tags:
            priority = LocalPromptRefiner._tag_priority(tag)
            buckets[priority].append(tag)
        for priority in range(7):
            buckets[priority].sort(
                key=lambda tag: LocalPromptRefiner._within_category_priority(tag, priority)
            )
            buckets[priority] = buckets[priority][:limits[priority]]

        # Interleave requested categories so a verbose group cannot consume the
        # entire CLIP budget. Generic quality/style tags remain strictly last.
        tags = []
        max_bucket = max((len(buckets[p]) for p in range(6)), default=0)
        for index in range(max_bucket):
            for priority in range(6):
                if index < len(buckets[priority]):
                    tags.append(buckets[priority][index])
        tags.extend(buckets[6])
        accepted = []
        for tag in tags[:24]:
            candidate = ", ".join([*accepted, tag])
            if len(candidate) > 380:
                continue
            accepted.append(tag)
        return ", ".join(accepted)

    @staticmethod
    def _parse_json(text):
        text = text.strip()
        if "<think>" in text and "</think>" in text:
            text = text.split("</think>", 1)[1].strip()
        if "```" in text:
            for block in text.split("```"):
                candidate = block.strip()
                if candidate.startswith("json"):
                    candidate = candidate[4:].strip()
                if candidate.startswith("{") and candidate.endswith("}"):
                    text = candidate
                    break
        start = text.find("{")
        end = text.rfind("}")
        if start < 0 or end <= start:
            raise ValueError("Prompt refiner did not return JSON.")
        result = json.loads(text[start:end + 1])
        prompt = LocalPromptRefiner._compact_prompt(str(result.get("prompt", "")).strip())
        if len(prompt) < 20:
            raise ValueError("Prompt refiner returned an empty or very short prompt.")
        return {
            "prompt": prompt,
            "intent_notes": str(result.get("intent_notes", "")).strip(),
        }

    @staticmethod
    def _parse_plain_prompt(text):
        text = text.strip()
        if "<think>" in text and "</think>" in text:
            text = text.split("</think>", 1)[1].strip()
        text = re.sub(r"^\s*(?:prompt|final prompt)\s*[:：]\s*", "", text, flags=re.IGNORECASE)
        text = text.replace("```", "").strip()
        prompt = LocalPromptRefiner._compact_prompt(text)
        if len(prompt) < 20:
            raise ValueError("Prompt refiner retry returned an empty or very short prompt.")
        # A retry must be an English SDXL prompt. Do not silently send another
        # long Japanese request into CLIP if the model ignored the instruction.
        latin_chars = sum(char.isascii() and char.isalpha() for char in prompt)
        if latin_chars < 12:
            raise ValueError("Prompt refiner retry did not return an English prompt.")
        return {"prompt": prompt, "intent_notes": "Used compact plain-text retry."}

    @staticmethod
    def _apply_japanese_glossary(result, user_prompt, mode):
        prompt = result["prompt"]
        if "ワンピース" in user_prompt and "水着" not in user_prompt:
            replacement = "long dress" if "ロングワンピース" in user_prompt or "ロング丈" in user_prompt else "dress"
            prompt = re.sub(r"\b(?:one[- ]piece\s+)?swimsuit\b", replacement, prompt, flags=re.IGNORECASE)
            prompt = re.sub(r"\bone[- ]piece dress\b", replacement, prompt, flags=re.IGNORECASE)
        tags = []
        edit_style_tags = {
            "anime", "visual-novel", "visual novel", "2d", "line-art", "line art",
            "vivid colors", "atmospheric background", "non-photorealistic",
            "masterpiece", "best quality", "very aesthetic",
        }
        for tag in (item.strip() for item in prompt.split(",") if item.strip()):
            if re.search(r"\b(?:unchanged|do not change|keep unchanged)\b", tag, flags=re.IGNORECASE):
                continue
            if mode == "edit" and tag.lower() in edit_style_tags:
                continue
            if mode == "edit" and "制服" not in user_prompt and re.search(r"\bschool uniform\b", tag, flags=re.IGNORECASE):
                continue
            if tag.lower() not in {item.lower() for item in tags}:
                tags.append(tag)
        result["prompt"] = LocalPromptRefiner._compact_prompt(", ".join(tags))
        return result

    def _generate(self, messages, max_new_tokens):
        text = self.processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        raw_inputs = self.processor(text=text, return_tensors="pt")
        inputs = {
            name: raw_inputs[name]
            for name in ("input_ids", "attention_mask")
            if name in raw_inputs
        }
        if "input_ids" not in inputs:
            raise RuntimeError("Prompt refiner did not return token ids.")
        target_device = getattr(self.model, "device", torch.device("cpu"))
        inputs = {name: value.to(target_device) for name, value in inputs.items()}
        input_len = inputs["input_ids"].shape[-1]
        with torch.inference_mode():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
            )
        return self.processor.decode(outputs[0][input_len:], skip_special_tokens=True)

    def refine(self, user_prompt, mode, preset_label, preset_hint, style_description, enhance=True):
        self._load()
        if mode == "edit":
            system_prompt = EDIT_SYSTEM_PROMPT
        elif enhance:
            system_prompt = SYSTEM_PROMPT
        else:
            system_prompt = TRANSLATE_ONLY_SYSTEM_PROMPT
        messages = [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": (
                    f"Mode: {mode}\n"
                    f"Style preset: {preset_label}\n"
                    f"Preset visual direction: {preset_hint or 'none'}\n"
                    f"Style preferences:\n{style_description}\n\n"
                    f"User request:\n{user_prompt}"
                ),
            },
        ]
        max_new_tokens = int(os.environ.get("PROMPT_REFINER_MAX_NEW_TOKENS", "220"))
        raw = self._generate(messages, max_new_tokens)
        try:
            result = self._parse_json(raw)
            result["source"] = self.model_id
            return self._apply_japanese_glossary(result, user_prompt, mode)
        except (ValueError, json.JSONDecodeError) as exc:
            print(f"[refine] JSON response was unusable ({exc}); retrying with plain-text output")

        retry_system = RETRY_SYSTEM_PROMPT
        if mode == "edit":
            retry_system = EDIT_SYSTEM_PROMPT.replace("Return JSON only with keys \"prompt\" and \"intent_notes\".", "Reply with one comma-separated line only.")
        elif not enhance:
            retry_system += "\nTranslate only. Do not infer or add visual details."
        retry_messages = [
            {"role": "system", "content": retry_system},
            {
                "role": "user",
                "content": (
                    f"Mode: {mode}\n"
                    f"Style direction: {preset_hint or preset_label}\n"
                    f"User request:\n{user_prompt}"
                ),
            },
        ]
        result = self._parse_plain_prompt(self._generate(retry_messages, max_new_tokens))
        suffix = "plain-retry" if enhance else "translation-only-retry"
        result["source"] = f"{self.model_id}:{suffix}"
        return self._apply_japanese_glossary(result, user_prompt, mode)

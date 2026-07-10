import json
import os

import torch


SYSTEM_PROMPT = """
You are a prompt director for the JANKU v7.77 Illustrious XL anime image model.
Convert the user's Japanese request into one concise English prompt suitable for
anime image generation or image editing.

Rules:
- Preserve every requested subject, appearance detail, outfit, pose, camera angle,
  composition, weather, background, color, material, expression, and mood.
- Never invent unrelated characters, props, text, logos, brands, or locations.
- Put the most important subject and style terms first.
- Use a compact combination of natural English and useful booru-style tags.
- For edits, describe the desired final image instead of explaining editing steps.
- Honor the selected style preset and numerical style preferences.
- Return JSON only with keys "prompt" and "intent_notes".
""".strip()


class LocalPromptRefiner:
    def __init__(self, model_id=None):
        self.model_id = model_id or os.environ.get("PROMPT_REFINER_MODEL", "Qwen/Qwen3.5-9B")
        self.processor = None
        self.model = None

    def _load(self):
        if self.model is not None:
            return

        from transformers import AutoModelForCausalLM, AutoProcessor

        print(f"[refine] Loading local prompt refiner: {self.model_id}")
        self.processor = AutoProcessor.from_pretrained(self.model_id, trust_remote_code=True)
        device = os.environ.get("PROMPT_REFINER_DEVICE", "cpu").lower()
        model_kwargs = {"dtype": "auto", "trust_remote_code": True}
        if device == "cpu":
            model_kwargs["device_map"] = {"": "cpu"}
        elif device in {"cuda", "gpu"}:
            model_kwargs["device_map"] = "cuda"
        else:
            model_kwargs["device_map"] = "auto"
        self.model = AutoModelForCausalLM.from_pretrained(self.model_id, **model_kwargs).eval()

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
        prompt = str(result.get("prompt", "")).strip()
        if len(prompt) < 20:
            raise ValueError("Prompt refiner returned an empty or very short prompt.")
        return {
            "prompt": prompt,
            "intent_notes": str(result.get("intent_notes", "")).strip(),
        }

    def refine(self, user_prompt, mode, preset_label, preset_hint, style_description):
        self._load()
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
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
        text = self.processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        inputs = self.processor(text=text, return_tensors="pt")
        target_device = getattr(self.model, "device", torch.device("cpu"))
        inputs = inputs.to(target_device)
        input_len = inputs["input_ids"].shape[-1]
        max_new_tokens = int(os.environ.get("PROMPT_REFINER_MAX_NEW_TOKENS", "900"))

        with torch.inference_mode():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
            )
        raw = self.processor.decode(outputs[0][input_len:], skip_special_tokens=True)
        result = self._parse_json(raw)
        result["source"] = self.model_id
        return result

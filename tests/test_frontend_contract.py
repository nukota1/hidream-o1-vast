import re
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class FrontendContractTests(unittest.TestCase):
    def test_character_only_wrapper_contains_identity_prompt_not_lora_name(self):
        template = (ROOT / "templates" / "index.html").read_text(encoding="utf-8")
        wrapper = re.search(
            r'<div class="field" id="lora-identity-field">(.*?)</div>',
            template,
            flags=re.DOTALL,
        )

        self.assertIsNotNone(wrapper)
        self.assertIn('id="lora-identity-prompt"', wrapper.group(1))
        self.assertNotIn('id="lora-name"', wrapper.group(1))

    def test_batch_generation_controls_and_result_grid_are_present(self):
        template = (ROOT / "templates" / "index.html").read_text(encoding="utf-8")
        script = (ROOT / "static" / "app.js").read_text(encoding="utf-8")

        self.assertIn('id="character-batch-count"', template)
        self.assertIn('id="compose-batch-count"', template)
        self.assertIn('id="result-batch-grid"', template)
        self.assertIn('batch_count: batchCountForWorkflow("character")', script)
        self.assertIn('data.type === "batch_image"', script)
        self.assertIn("showGenerationResults(data, rootPrompt)", script)

    def test_style_strength_is_cached_and_restored_from_history(self):
        script = (ROOT / "static" / "app.js").read_text(encoding="utf-8")

        self.assertIn('STYLE_CACHE_KEY = "illustration-style-strength-v1"', script)
        self.assertIn("localStorage.setItem(STYLE_CACHE_KEY", script)
        self.assertIn("restoreCachedStyleSettings();", script)
        self.assertIn("record.metadata?.style_settings", script)
        self.assertIn("generationSettings.style", script)
        self.assertIn("style_settings: data.settings?.style", script)


if __name__ == "__main__":
    unittest.main()

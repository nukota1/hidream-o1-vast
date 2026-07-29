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


if __name__ == "__main__":
    unittest.main()

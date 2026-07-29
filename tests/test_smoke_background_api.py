import os
import unittest
from unittest.mock import patch

from scripts.smoke_background_api import backend_headers


class SmokeBackgroundApiAuthTests(unittest.TestCase):
    def test_backend_header_is_omitted_without_shared_secret(self):
        with patch.dict(os.environ, {}, clear=True):
            self.assertEqual(backend_headers(), {})

    def test_backend_header_uses_shared_secret_from_environment(self):
        with patch.dict(
            os.environ,
            {"BACKEND_SHARED_SECRET": "test-shared-secret"},
            clear=True,
        ):
            self.assertEqual(
                backend_headers(),
                {"X-Backend-Key": "test-shared-secret"},
            )


if __name__ == "__main__":
    unittest.main()

"""Offline check for the repo-local Modal index download command."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from scripts.download_index import INDEX_VOLUME_NAME, INDEX_VOLUME_PATH, download_index


class DownloadIndexTests(unittest.TestCase):
    def test_uses_project_path_and_volume(self):
        with tempfile.TemporaryDirectory() as directory, patch("scripts.download_index.subprocess.run") as run:
            root = Path(directory)
            result = download_index(root)
            destination = root / "rag" / "vector_db"
            self.assertTrue(destination.is_dir())
            self.assertEqual(result, destination / INDEX_VOLUME_PATH.lstrip("/"))
            run.assert_called_once_with(
                ["modal", "volume", "get", INDEX_VOLUME_NAME, INDEX_VOLUME_PATH, str(destination)],
                check=True,
            )


if __name__ == "__main__":
    unittest.main()

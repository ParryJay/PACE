"""Check reference preprocessing and strict training/test data separation."""
from __future__ import annotations

import json
from pathlib import Path
import sys
import tempfile
import unittest

import cv2
import numpy as np
from torch.utils.data import DataLoader

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))
from data import PairedDataset


class PairedDataTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="pace_data_", dir="/tmp")
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.image = np.arange(7 * 9 * 3, dtype=np.uint8).reshape(7, 9, 3)
        self.mask = np.where(np.indices((7, 9)).sum(0) % 3 == 0, 255, 0).astype(np.uint8)

    def _fixture(self, dataset, *, split="train", labels=True):
        root = self.root / dataset
        for modality in ("nbi", "wli"):
            image_dir = root / modality / "img" if dataset == "FAHSYSU" else root / "img" / modality.upper()
            image_dir.mkdir(parents=True, exist_ok=True)
            self.assertTrue(cv2.imwrite(str(image_dir / f"{modality}.png"), self.image))
            if labels:
                label_dir = root / modality / "lbl" if dataset == "FAHSYSU" else root / "lbl"
                label_dir.mkdir(parents=True, exist_ok=True)
                self.assertTrue(cv2.imwrite(str(label_dir / f"{modality}.png"), self.mask))
        if dataset == "FAHSYSU":
            pair = {"nbi_img": "nbi.png", "wli_img": "wli.png",
                    "nbi_lbl": "nbi.png", "wli_lbl": "wli.png"}
            manifest = root / f"pairs_{split}.json"
        else:
            pair = {"nbi": "nbi.png", "wht_img": "wli.png"}
            manifest = root / f"{split}.jsonl"
        if not labels:
            pair.update(nbi_lbl={"invalid": "must be ignored"}, wli_lbl="missing.png")
        manifest.write_text(json.dumps([pair]) if dataset == "FAHSYSU" else json.dumps(pair) + "\n", encoding="utf-8")
        return root

    def test_preprocessing_matches_reference_for_both_datasets(self):
        # The legacy code reads labels as color, resizes with bilinear
        # interpolation, converts to grayscale, then thresholds at 100.
        expected_image = cv2.cvtColor(cv2.resize(self.image, (16, 16)), cv2.COLOR_BGR2RGB).transpose(2, 0, 1)
        label_color = cv2.cvtColor(self.mask, cv2.COLOR_GRAY2BGR)
        expected_label = (cv2.cvtColor(cv2.resize(label_color, (16, 16)), cv2.COLOR_BGR2GRAY) > 100).astype(np.uint8)
        for dataset in ("FAHSYSU", "PICCOLO"):
            with self.subTest(dataset=dataset):
                root = self._fixture(dataset)
                data = PairedDataset(dataset, root, img_size=16)
                sample = data[0]
                for modality in ("nbi", "wli"):
                    np.testing.assert_array_equal(sample[modality + "_img"], expected_image)
                    np.testing.assert_array_equal(sample[modality + "_lbl"], expected_label)
                    self.assertEqual(sample[modality + "_img"].dtype, np.uint8)
                    self.assertEqual(sample[modality + "_lbl"].dtype, np.uint8)
                self.assertIs(data.train_pairs, data.pairs)
                batch = next(iter(DataLoader(data, batch_size=1)))
                self.assertEqual(tuple(batch["nbi_img"].shape), (1, 3, 16, 16))
                self.assertEqual(tuple(batch["wli_lbl"].shape), (1, 16, 16))

    def test_training_never_opens_test_manifest(self):
        for dataset in ("FAHSYSU", "PICCOLO"):
            with self.subTest(dataset=dataset):
                root = self._fixture(dataset)
                test_name = "pairs_test.json" if dataset == "FAHSYSU" else "test.jsonl"
                (root / test_name).write_text("deliberately invalid JSON", encoding="utf-8")
                data = PairedDataset(dataset, root, split="train", img_size=16)
                self.assertEqual(len(data), 1)
                self.assertIn("nbi_lbl", data[0])

    def test_inference_does_not_require_or_expose_labels(self):
        for dataset in ("FAHSYSU", "PICCOLO"):
            with self.subTest(dataset=dataset):
                root = self._fixture(dataset, split="test", labels=False)
                data = PairedDataset(dataset, root, split="test", img_size=16)
                sample = data[0]
                self.assertEqual(sample["pair_index"], 0)
                self.assertFalse(any("lbl" in key for key in sample))
                self.assertFalse(any("lbl" in key for key in sample["info"]))
                self.assertFalse(list(root.rglob("lbl")))
                batch = next(iter(DataLoader(data, batch_size=1)))
                self.assertEqual(tuple(batch["wli_img"].shape), (1, 3, 16, 16))
                with self.assertRaisesRegex(ValueError, "inference-only"):
                    data.label_path(data.pairs[0], "nbi")

    def test_explicit_manifest_resolves_relative_to_data_root(self):
        root = self._fixture("FAHSYSU")
        original = root / "pairs_train.json"
        manifest = root / "splits" / "custom.json"
        manifest.parent.mkdir()
        original.replace(manifest)
        relative = PairedDataset("FAHSYSU", root, "splits/custom.json", img_size=16)
        absolute = PairedDataset("FAHSYSU", root, manifest, img_size=16)
        self.assertEqual(relative.pairs, absolute.pairs)
        np.testing.assert_array_equal(relative[0]["nbi_lbl"], absolute[0]["nbi_lbl"])


if __name__ == "__main__":
    unittest.main()

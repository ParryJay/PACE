"""Independent paired-image loading for FAHSYSU and PICCOLO.

Training preserves the original RGB image and binary-label preprocessing.
The test split loads only images: label files are neither needed nor opened.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

import cv2
import numpy as np
from torch.utils.data import Dataset


def read_image(
    path: str | Path,
    format: str = "RGB",
    resize: tuple[int, int] | None = None,
) -> np.ndarray:
    """Match the original color-read -> linear-resize -> convert pipeline.

    In particular, labels are resized as color images before grayscale
    conversion and thresholding at 100; this intentionally preserves the
    reference training targets.
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Image file does not exist: {path}")
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"OpenCV could not decode image: {path}")
    if resize is not None:
        image = cv2.resize(image, resize, interpolation=cv2.INTER_LINEAR)
    image_format = format.upper()
    if image_format == "RGB":
        return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    if image_format == "BGR":
        return image
    if image_format in {"GRAY", "01", "LABEL_01"}:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        return (image > 100).astype(np.uint8) if image_format != "GRAY" else image
    raise ValueError(f"Unsupported image format: {format!r}")


def _extract_frame(filename: str) -> str:
    base = Path(filename).stem
    frame_index = base.lower().rfind("frame")
    if frame_index != -1:
        return base[frame_index + len("frame"):]
    parts = base.split("_")
    return parts[1] if len(parts) >= 2 and parts[1].isdigit() else ""


def _filename(value: Any, *, field: str, index: int) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"Pair {index}: {field} must be a nonempty filename string.")
    return value


def _normalize_pair(
    record: Mapping[str, Any], dataset: str, split: str, index: int,
) -> dict[str, str]:
    nbi = _filename(record.get("nbi_img") or record.get("nbi"), field="nbi_img", index=index)
    wli = _filename(
        record.get("wli_img") or record.get("wli") or record.get("wht_img"),
        field="wli_img", index=index,
    )
    pair = {
        "id": str(record.get("id") or record.get("patient_id") or Path(nbi).stem.split("_")[0]),
        "nbi_frame": str(record.get("nbi_frame") or _extract_frame(nbi)),
        "wht_frame": str(record.get("wht_frame") or record.get("wli_frame") or _extract_frame(wli)),
        "nbi_img": nbi,
        "wli_img": wli,
    }
    # Do not inspect label fields at all for test/inference manifests.
    if split == "train":
        for modality, image in (("nbi", nbi), ("wli", wli)):
            label = record.get(modality + "_lbl")
            if dataset == "PICCOLO":
                label = label or image
            pair[modality + "_lbl"] = _filename(label, field=modality + "_lbl", index=index)
    return pair


def _read_manifest(path: Path) -> list[Mapping[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(
            f"Pair manifest does not exist: {path}. "
            "Provide --train_manifest/--test_manifest if stored outside the dataset root."
        )
    with path.open("r", encoding="utf-8") as handle:
        if path.suffix.lower() == ".jsonl":
            records = []
            for line_number, line in enumerate(handle, 1):
                if not line.strip():
                    continue
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError as error:
                    raise ValueError(f"Invalid JSON at {path}:{line_number}: {error.msg}") from error
        else:
            records = json.load(handle)
    if not isinstance(records, list):
        raise ValueError(f"Pair manifest must contain a JSON array or JSONL records: {path}")
    if not records:
        raise ValueError(f"Pair manifest is empty: {path}")
    for index, record in enumerate(records):
        if not isinstance(record, Mapping):
            raise ValueError(f"Pair {index} in {path} must be a JSON object.")
    return records


class PairedDataset(Dataset):
    """Dataset-specific paths with shared paired-image preprocessing.

    Relative manifest paths are resolved against ``root``. Image and label
    fields are filenames relative to each dataset's modality directory; absolute
    paths also work. Instantiating ``split='train'`` never opens the test list.
    """

    def __init__(
        self,
        dataset: str,
        root: str | Path,
        manifest: str | Path | None = None,
        *,
        split: str = "train",
        img_size: int = 512,
    ) -> None:
        if dataset not in {"FAHSYSU", "PICCOLO"}:
            raise ValueError("dataset must be FAHSYSU or PICCOLO.")
        if split not in {"train", "test"}:
            raise ValueError("split must be train or test.")
        if not isinstance(img_size, int) or img_size < 1:
            raise ValueError("img_size must be a positive integer.")
        self.dataset, self.split, self.img_size = dataset, split, img_size
        self.root = Path(root).expanduser().resolve()
        default_name = f"pairs_{split}.json" if dataset == "FAHSYSU" else f"{split}.jsonl"
        manifest_path = Path(manifest).expanduser() if manifest is not None else Path(default_name)
        self.manifest = manifest_path if manifest_path.is_absolute() else self.root / manifest_path
        self.pairs = [
            _normalize_pair(pair, dataset, split, index)
            for index, pair in enumerate(_read_manifest(self.manifest))
        ]
        # Compatibility names for the training prior and checkpoint metadata.
        self.train_pairs = self.pairs if split == "train" else []
        self.test_pairs = self.pairs if split == "test" else []

    def __len__(self) -> int:
        return len(self.pairs)

    @staticmethod
    def _check_modality(modality: str) -> None:
        if modality not in {"nbi", "wli"}:
            raise ValueError("modality must be nbi or wli.")

    def image_path(self, pair: Mapping[str, str], modality: str) -> Path:
        self._check_modality(modality)
        directory = self.root / modality / "img" if self.dataset == "FAHSYSU" else self.root / "img" / modality.upper()
        return directory / pair[modality + "_img"]

    def label_path(self, pair: Mapping[str, str], modality: str) -> Path:
        if self.split != "train":
            raise ValueError("The test split is inference-only and does not expose label files.")
        self._check_modality(modality)
        directory = self.root / modality / "lbl" if self.dataset == "FAHSYSU" else self.root / "lbl"
        return directory / pair[modality + "_lbl"]

    def read_label(self, pair: Mapping[str, str], modality: str) -> np.ndarray:
        return read_image(self.label_path(pair, modality), format="01", resize=(self.img_size, self.img_size))

    def __getitem__(self, index: int) -> dict[str, Any]:
        pair = self.pairs[index]
        result: dict[str, Any] = {"pair_index": index, "info": dict(pair)}
        for modality in ("nbi", "wli"):
            image = read_image(
                self.image_path(pair, modality), format="RGB", resize=(self.img_size, self.img_size),
            )
            result[modality + "_img"] = np.ascontiguousarray(image.transpose(2, 0, 1))
            if self.split == "train":
                result[modality + "_lbl"] = self.read_label(pair, modality)
        return result

"""Reproducible synthetic observation-quality augmentation used during training."""
from __future__ import annotations

import math
from numbers import Integral
from typing import Any, Mapping, Sequence

import torch
import torch.nn.functional as F

class SyntheticQualityDegrader:
    """Torch-only, reproducible synthetic observation-quality degradation.

    Images are expected *after* geometric augmentation but *before* ImageNet
    normalization.  The public call accepts ``[C,H,W]`` or ``[B,C,H,W]`` in
    ``[0,1]`` and never receives or changes a segmentation target.  All random
    draws use a private/explicit :class:`torch.Generator`; Python, NumPy and the
    global torch RNG states are therefore untouched.
    """

    SUPPORTED_OPERATIONS = (
        "gaussian_blur",
        "motion_blur",
        "defocus_blur",
        "occlusion",
        "overexposure",
        "low_illumination",
        "specular_highlight",
        "black_border",
        "color_corruption",
    )
    _ALIASES = {
        "local_occlusion": "occlusion",
        "endoscopic_black_border": "black_border",
        "local_color_corruption": "color_corruption",
        "specular_highlight_simulation": "specular_highlight",
    }

    def __init__(
        self,
        apply_probability: float = 0.8,
        max_operations: int = 2,
        min_severity: float = 0.2,
        max_severity: float = 1.0,
        enabled_operations: Mapping[str, bool] | Sequence[str] | None = None,
        operation_probabilities: Mapping[str, float] | None = None,
        seed: int = 0,
        local_probability: float = 0.8,
    ) -> None:
        self.apply_probability = self._probability(
            apply_probability, "apply_probability"
        )
        if isinstance(max_operations, bool) or not isinstance(max_operations, Integral):
            raise TypeError("max_operations must be a non-negative integer.")
        if int(max_operations) < 0:
            raise ValueError("max_operations must be non-negative.")
        self.max_operations = int(max_operations)
        self.min_severity = self._probability(min_severity, "min_severity")
        self.max_severity = self._probability(max_severity, "max_severity")
        if self.min_severity > self.max_severity:
            raise ValueError("min_severity cannot exceed max_severity.")
        self.local_probability = self._probability(
            local_probability, "local_probability"
        )
        if isinstance(seed, bool) or not isinstance(seed, Integral):
            raise TypeError("seed must be an integer (bool is not accepted).")
        self.seed = int(seed)
        self._call_index = 0

        if enabled_operations is None:
            enabled = {name: True for name in self.SUPPORTED_OPERATIONS}
        elif isinstance(enabled_operations, Mapping):
            enabled = {name: False for name in self.SUPPORTED_OPERATIONS}
            for raw_name, value in enabled_operations.items():
                name = self._canonical_name(raw_name)
                if not isinstance(value, bool):
                    raise TypeError(f"enabled_operations[{raw_name!r}] must be bool.")
                enabled[name] = value
        elif isinstance(enabled_operations, Sequence) and not isinstance(
            enabled_operations, (str, bytes)
        ):
            enabled = {name: False for name in self.SUPPORTED_OPERATIONS}
            for raw_name in enabled_operations:
                enabled[self._canonical_name(raw_name)] = True
        else:
            raise TypeError(
                "enabled_operations must be a mapping, a sequence of names, or None."
            )
        self.enabled_operations = enabled

        probabilities = {name: 1.0 for name in self.SUPPORTED_OPERATIONS}
        if operation_probabilities is not None:
            if not isinstance(operation_probabilities, Mapping):
                raise TypeError("operation_probabilities must be a mapping or None.")
            for raw_name, probability in operation_probabilities.items():
                name = self._canonical_name(raw_name)
                probabilities[name] = self._probability(
                    probability, f"operation_probabilities[{raw_name!r}]"
                )
        self.operation_probabilities = probabilities
        self._registry = {
            "gaussian_blur": self._gaussian_blur,
            "motion_blur": self._motion_blur,
            "defocus_blur": self._defocus_blur,
            "occlusion": self._occlusion,
            "overexposure": self._overexposure,
            "low_illumination": self._low_illumination,
            "specular_highlight": self._specular_highlight,
            "black_border": self._black_border,
            "color_corruption": self._color_corruption,
        }

    @staticmethod
    def _probability(value: object, name: str) -> float:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise TypeError(f"{name} must be a real number.")
        result = float(value)
        if not math.isfinite(result) or not 0.0 <= result <= 1.0:
            raise ValueError(f"{name} must be finite and in [0,1], got {value}.")
        return result

    @classmethod
    def _canonical_name(cls, operation: object) -> str:
        if not isinstance(operation, str):
            raise TypeError("Degradation operation names must be strings.")
        name = cls._ALIASES.get(operation, operation)
        if name not in cls.SUPPORTED_OPERATIONS:
            raise ValueError(
                f"Unsupported quality degradation {operation!r}; expected one of "
                f"{cls.SUPPORTED_OPERATIONS}."
            )
        return name

    def reset_seed(self, seed: int | None = None) -> None:
        """Reset the private deterministic sequence without touching global RNGs."""

        if seed is not None:
            if isinstance(seed, bool) or not isinstance(seed, Integral):
                raise TypeError("seed must be an integer.")
            self.seed = int(seed)
        self._call_index = 0

    @staticmethod
    def _generator(device: torch.device, seed: int) -> torch.Generator:
        generator = torch.Generator(device=device)
        generator.manual_seed(int(seed) % (2**63 - 1))
        return generator

    @staticmethod
    def _rand(
        generator: torch.Generator,
        device: torch.device,
        *shape: int,
    ) -> torch.Tensor:
        return torch.rand(shape or (), generator=generator, device=device)

    @classmethod
    def _uniform(
        cls,
        generator: torch.Generator,
        device: torch.device,
        low: float,
        high: float,
    ) -> float:
        if high == low:
            return float(low)
        return float(
            (low + (high - low) * cls._rand(generator, device)).item()
        )

    @classmethod
    def _randint(
        cls,
        generator: torch.Generator,
        device: torch.device,
        low: int,
        high: int,
    ) -> int:
        if high <= low:
            return int(low)
        return int(
            torch.randint(low, high, (), generator=generator, device=device).item()
        )

    @staticmethod
    def _coordinate_grid(
        height: int, width: int, device: torch.device
    ) -> tuple[torch.Tensor, torch.Tensor]:
        y = torch.linspace(-1.0, 1.0, height, device=device, dtype=torch.float32)
        x = torch.linspace(-1.0, 1.0, width, device=device, dtype=torch.float32)
        return torch.meshgrid(y, x, indexing="ij")

    def _region_mask(
        self,
        height: int,
        width: int,
        generator: torch.Generator,
        device: torch.device,
        *,
        allow_global: bool = True,
        shape: str | None = None,
        soft: bool = True,
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        if allow_global and bool(
            (self._rand(generator, device) >= self.local_probability).item()
        ):
            return torch.ones((1, height, width), device=device), {"scope": "global"}

        grid_y, grid_x = self._coordinate_grid(height, width, device)
        center_x = self._uniform(generator, device, -0.55, 0.55)
        center_y = self._uniform(generator, device, -0.55, 0.55)
        radius_x = self._uniform(generator, device, 0.18, 0.55)
        radius_y = self._uniform(generator, device, 0.18, 0.55)
        if shape is None:
            shape = ("ellipse", "rectangle", "irregular")[
                self._randint(generator, device, 0, 3)
            ]
        softness = 0.12 if soft else 1.0e-6

        def primitive(
            cx: float, cy: float, rx: float, ry: float, kind: str
        ) -> torch.Tensor:
            dx = (grid_x - cx).abs() / max(rx, 1.0e-4)
            dy = (grid_y - cy).abs() / max(ry, 1.0e-4)
            distance = torch.maximum(dx, dy) if kind == "rectangle" else (
                dx.square() + dy.square()
            ).sqrt()
            return ((1.0 + softness - distance) / softness).clamp(0.0, 1.0)

        base_kind = "rectangle" if shape == "rectangle" else "ellipse"
        mask = primitive(center_x, center_y, radius_x, radius_y, base_kind)
        if shape == "irregular":
            offset_x = self._uniform(generator, device, -0.25, 0.25)
            offset_y = self._uniform(generator, device, -0.25, 0.25)
            second = primitive(
                center_x + offset_x,
                center_y + offset_y,
                radius_x * self._uniform(generator, device, 0.55, 0.9),
                radius_y * self._uniform(generator, device, 0.55, 0.9),
                "ellipse",
            )
            mask = torch.maximum(mask, second)
        metadata = {
            "scope": "local",
            "shape": shape,
            "center": (center_x, center_y),
            "radius": (radius_x, radius_y),
        }
        return mask.unsqueeze(0), metadata

    @staticmethod
    def _odd_kernel_size(severity: float, height: int, width: int) -> int:
        requested = 3 + 2 * int(round(4.0 * severity))
        maximum = max(1, min(19, 2 * min(height, width) - 1))
        if maximum % 2 == 0:
            maximum -= 1
        return max(1, min(requested, maximum))

    @staticmethod
    def _filter(image: torch.Tensor, kernel: torch.Tensor) -> torch.Tensor:
        channels, _, _ = image.shape
        kernel_size = int(kernel.shape[0])
        if kernel.shape != (kernel_size, kernel_size) or kernel_size % 2 != 1:
            raise ValueError("Filtering kernel must be square with an odd size.")
        kernel = kernel.to(device=image.device, dtype=torch.float32)
        kernel = kernel / kernel.sum().clamp_min(1.0e-12)
        weight = kernel.view(1, 1, kernel_size, kernel_size).expand(
            channels, 1, -1, -1
        )
        padded = F.pad(
            image.float().unsqueeze(0),
            (kernel_size // 2,) * 4,
            mode="replicate",
        )
        return F.conv2d(padded, weight, groups=channels).squeeze(0)

    @staticmethod
    def _blend(
        image: torch.Tensor, degraded: torch.Tensor, alpha: torch.Tensor
    ) -> torch.Tensor:
        return image * (1.0 - alpha) + degraded * alpha

    def _gaussian_blur(
        self, image: torch.Tensor, severity: float, generator: torch.Generator
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
        _, height, width = image.shape
        kernel_size = self._odd_kernel_size(severity, height, width)
        radius = kernel_size // 2
        coordinate = torch.arange(
            -radius, radius + 1, device=image.device, dtype=torch.float32
        )
        sigma = 0.35 + 2.65 * severity
        kernel_1d = torch.exp(-coordinate.square() / (2.0 * sigma * sigma))
        kernel = kernel_1d[:, None] * kernel_1d[None, :]
        blurred = self._filter(image, kernel)
        mask, metadata = self._region_mask(
            height, width, generator, image.device, allow_global=True
        )
        metadata.update({"kernel_size": kernel_size, "sigma": sigma})
        return self._blend(image, blurred, mask), mask, metadata

    def _motion_blur(
        self, image: torch.Tensor, severity: float, generator: torch.Generator
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
        _, height, width = image.shape
        kernel_size = self._odd_kernel_size(severity, height, width)
        radius = kernel_size // 2
        axis = torch.arange(
            -radius, radius + 1, device=image.device, dtype=torch.float32
        )
        yy, xx = torch.meshgrid(axis, axis, indexing="ij")
        angle = self._uniform(generator, image.device, 0.0, math.pi)
        perpendicular = (-math.sin(angle) * xx + math.cos(angle) * yy).abs()
        along = (math.cos(angle) * xx + math.sin(angle) * yy).abs()
        kernel = torch.exp(-0.5 * (perpendicular / 0.45).square())
        kernel = kernel * (along <= radius + 0.25)
        blurred = self._filter(image, kernel)
        mask, metadata = self._region_mask(
            height, width, generator, image.device, allow_global=True
        )
        metadata.update({"kernel_size": kernel_size, "angle_radians": angle})
        return self._blend(image, blurred, mask), mask, metadata

    def _defocus_blur(
        self, image: torch.Tensor, severity: float, generator: torch.Generator
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
        _, height, width = image.shape
        kernel_size = self._odd_kernel_size(severity, height, width)
        radius = kernel_size // 2
        axis = torch.arange(
            -radius, radius + 1, device=image.device, dtype=torch.float32
        )
        yy, xx = torch.meshgrid(axis, axis, indexing="ij")
        disk_radius = max(1.0, radius * (0.6 + 0.4 * severity))
        kernel = (xx.square() + yy.square() <= disk_radius**2).float()
        blurred = self._filter(image, kernel)
        mask, metadata = self._region_mask(
            height, width, generator, image.device, allow_global=True
        )
        metadata.update({"kernel_size": kernel_size, "disk_radius": disk_radius})
        return self._blend(image, blurred, mask), mask, metadata

    def _occlusion(
        self, image: torch.Tensor, severity: float, generator: torch.Generator
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
        _, height, width = image.shape
        mask, metadata = self._region_mask(
            height, width, generator, image.device, allow_global=False, soft=True
        )
        fill_kind = ("black", "mean", "noise")[
            self._randint(generator, image.device, 0, 3)
        ]
        if fill_kind == "black":
            fill = torch.zeros_like(image)
        elif fill_kind == "mean":
            fill = image.mean(dim=(1, 2), keepdim=True).expand_as(image)
        else:
            fill = self._rand(generator, image.device, *image.shape)
        # Occlusion opacity grows monotonically while its label uses the same
        # severity and exact effective alpha mask.
        effective_mask = (mask * (0.35 + 0.65 * severity)).clamp(0.0, 1.0)
        metadata["fill"] = fill_kind
        return self._blend(image, fill, effective_mask), effective_mask, metadata

    def _overexposure(
        self, image: torch.Tensor, severity: float, generator: torch.Generator
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
        _, height, width = image.shape
        mask, metadata = self._region_mask(
            height, width, generator, image.device, allow_global=True
        )
        exposed = (image * (1.0 + 2.5 * severity) + 0.45 * severity).clamp(0.0, 1.0)
        return self._blend(image, exposed, mask), mask, metadata

    def _low_illumination(
        self, image: torch.Tensor, severity: float, generator: torch.Generator
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
        _, height, width = image.shape
        mask, metadata = self._region_mask(
            height, width, generator, image.device, allow_global=True
        )
        gamma = 1.0 + 2.5 * severity
        multiplier = max(0.03, 1.0 - 0.9 * severity)
        dark = image.clamp(0.0, 1.0).pow(gamma) * multiplier
        metadata.update({"gamma": gamma, "multiplier": multiplier})
        return self._blend(image, dark, mask), mask, metadata

    def _specular_highlight(
        self, image: torch.Tensor, severity: float, generator: torch.Generator
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
        _, height, width = image.shape
        blob_count = self._randint(generator, image.device, 1, 4)
        mask = torch.zeros((1, height, width), device=image.device)
        blob_metadata = []
        for _ in range(blob_count):
            blob, metadata = self._region_mask(
                height,
                width,
                generator,
                image.device,
                allow_global=False,
                shape="ellipse",
                soft=True,
            )
            # Specular spots are smaller than generic degradation regions.
            blob = F.avg_pool2d(blob.unsqueeze(0), 3, 1, 1).squeeze(0)
            mask = torch.maximum(mask, blob)
            blob_metadata.append(metadata)
        highlight = torch.ones_like(image)
        effective_mask = (mask * (0.45 + 0.55 * severity)).clamp(0.0, 1.0)
        return (
            self._blend(image, highlight, effective_mask),
            effective_mask,
            {"blob_count": blob_count, "blobs": blob_metadata},
        )

    def _black_border(
        self, image: torch.Tensor, severity: float, generator: torch.Generator
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
        _, height, width = image.shape
        style = ("circular", "bands")[
            self._randint(generator, image.device, 0, 2)
        ]
        if style == "circular":
            grid_y, grid_x = self._coordinate_grid(height, width, image.device)
            center_x = self._uniform(generator, image.device, -0.08, 0.08)
            center_y = self._uniform(generator, image.device, -0.08, 0.08)
            radius = 1.08 - 0.33 * severity
            distance = ((grid_x - center_x).square() + (grid_y - center_y).square()).sqrt()
            mask_2d = ((distance - radius) / 0.06).clamp(0.0, 1.0)
            metadata = {"style": style, "radius": radius}
        else:
            mask_2d = torch.zeros((height, width), device=image.device)
            band_h = max(1, int(round(height * (0.025 + 0.18 * severity))))
            band_w = max(1, int(round(width * (0.025 + 0.18 * severity))))
            sides = torch.randperm(4, generator=generator, device=image.device)[
                : self._randint(generator, image.device, 1, 4)
            ].tolist()
            for side in sides:
                if side == 0:
                    mask_2d[:band_h] = 1.0
                elif side == 1:
                    mask_2d[-band_h:] = 1.0
                elif side == 2:
                    mask_2d[:, :band_w] = 1.0
                else:
                    mask_2d[:, -band_w:] = 1.0
            metadata = {"style": style, "sides": sides}
        mask = mask_2d.unsqueeze(0)
        return self._blend(image, torch.zeros_like(image), mask), mask, metadata

    def _color_corruption(
        self, image: torch.Tensor, severity: float, generator: torch.Generator
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
        channels, height, width = image.shape
        mask, metadata = self._region_mask(
            height, width, generator, image.device, allow_global=False
        )
        gains = 1.0 + (
            self._rand(generator, image.device, channels, 1, 1) * 2.0 - 1.0
        ) * (0.9 * severity)
        offsets = (
            self._rand(generator, image.device, channels, 1, 1) * 2.0 - 1.0
        ) * (0.35 * severity)
        corrupted = image * gains + offsets
        if channels >= 3 and bool((self._rand(generator, image.device) < 0.5).item()):
            gray = corrupted.mean(dim=0, keepdim=True)
            corrupted = corrupted * (1.0 - 0.8 * severity) + gray * (0.8 * severity)
            metadata["desaturated"] = True
        else:
            metadata["desaturated"] = False
        metadata.update(
            {
                "channel_gains": gains.flatten().detach().cpu().tolist(),
                "channel_offsets": offsets.flatten().detach().cpu().tolist(),
            }
        )
        return self._blend(image, corrupted.clamp(0.0, 1.0), mask), mask, metadata

    def apply_named_operation(
        self,
        image: torch.Tensor,
        operation: str,
        severity: float,
        *,
        generator: torch.Generator | None = None,
        seed: int | None = None,
    ) -> dict[str, Any]:
        """Deterministic single-operation entry point for tests/validation suites."""

        return self(
            image,
            generator=generator,
            seed=seed,
            force_operations=(self._canonical_name(operation),),
            force_severity=severity,
        )

    def __call__(
        self,
        image: torch.Tensor,
        *,
        generator: torch.Generator | None = None,
        seed: int | None = None,
        force_operations: str | Sequence[str] | None = None,
        force_severity: float | Mapping[str, float] | None = None,
    ) -> dict[str, Any]:
        if not isinstance(image, torch.Tensor):
            raise TypeError("image must be a torch.Tensor.")
        squeeze_batch = image.ndim == 3
        if squeeze_batch:
            image = image.unsqueeze(0)
        if image.ndim != 4 or image.shape[1] < 1:
            raise ValueError(
                f"image must be [C,H,W] or [B,C,H,W], got {tuple(image.shape)}."
            )
        if image.shape[-2] < 2 or image.shape[-1] < 2:
            raise ValueError("Synthetic degradation needs spatial dimensions >= 2.")
        if not image.is_floating_point():
            if image.dtype != torch.uint8:
                raise TypeError("Integer images must use uint8 [0,255].")
            image_float = image.float() / 255.0
        else:
            image_float = image.float()
        if not bool(torch.isfinite(image_float).all().item()):
            raise ValueError("image contains NaN or Inf.")
        if bool((image_float.min() < 0.0).item()) or bool(
            (image_float.max() > 1.0).item()
        ):
            raise ValueError("SyntheticQualityDegrader expects image values in [0,1].")

        if generator is not None and seed is not None:
            raise ValueError("Provide generator or seed, not both.")
        if seed is not None and (
            isinstance(seed, bool) or not isinstance(seed, Integral)
        ):
            raise TypeError("seed must be an integer (bool is not accepted).")
        if generator is not None and not isinstance(generator, torch.Generator):
            raise TypeError("generator must be a torch.Generator or None.")
        if generator is None:
            resolved_seed = self.seed + self._call_index if seed is None else int(seed)
            generator = self._generator(image_float.device, resolved_seed)
            if seed is None:
                self._call_index += 1
        elif generator.device != image_float.device:
            raise ValueError(
                f"generator is on {generator.device}, image is on {image_float.device}."
            )

        forced: tuple[str, ...] | None
        if force_operations is None:
            forced = None
        elif isinstance(force_operations, str):
            forced = (self._canonical_name(force_operations),)
        else:
            forced = tuple(self._canonical_name(name) for name in force_operations)
        if isinstance(force_severity, Mapping):
            forced_severity = {
                self._canonical_name(name): self._probability(value, f"severity[{name}]")
                for name, value in force_severity.items()
            }
        elif force_severity is None:
            forced_severity = None
        else:
            scalar_severity = self._probability(force_severity, "force_severity")
            forced_severity = {
                name: scalar_severity for name in self.SUPPORTED_OPERATIONS
            }

        batch_size, _, height, width = image_float.shape
        # Forced operations are a test/validation control plane and therefore
        # get enough diagnostic slots even when random training max_operations
        # is configured as zero or smaller than the forced suite.
        operation_slots = max(
            self.max_operations, 0 if forced is None else len(forced)
        )
        degraded_images = []
        qualities = []
        masks = []
        operation_masks = image_float.new_zeros(
            (batch_size, operation_slots, 1, height, width)
        )
        operation_ids = torch.full(
            (batch_size, operation_slots),
            -1,
            device=image_float.device,
            dtype=torch.long,
        )
        operation_severities = image_float.new_zeros(
            (batch_size, operation_slots)
        )
        all_types: list[list[str]] = []
        all_params: list[list[dict[str, Any]]] = []

        for batch_index in range(batch_size):
            sample = image_float[batch_index].clone()
            quality = sample.new_ones((1, height, width))
            degradation_mask = sample.new_zeros((1, height, width))
            if forced is not None:
                selected = list(forced)
            elif self.max_operations == 0 or bool(
                (self._rand(generator, sample.device) >= self.apply_probability).item()
            ):
                selected = []
            else:
                candidates = [
                    name
                    for name in self.SUPPORTED_OPERATIONS
                    if self.enabled_operations[name]
                    and bool(
                        (
                            self._rand(generator, sample.device)
                            < self.operation_probabilities[name]
                        ).item()
                    )
                ]
                if candidates:
                    order = torch.randperm(
                        len(candidates), generator=generator, device=sample.device
                    ).tolist()
                    maximum = min(self.max_operations, len(candidates))
                    count = self._randint(generator, sample.device, 1, maximum + 1)
                    selected = [candidates[index] for index in order[:count]]
                else:
                    selected = []

            type_names = []
            parameter_rows = []
            for operation_index, name in enumerate(selected):
                if forced_severity is not None and name in forced_severity:
                    severity = forced_severity[name]
                else:
                    severity = self._uniform(
                        generator,
                        sample.device,
                        self.min_severity,
                        self.max_severity,
                    )
                if severity == 0.0:
                    # Zero severity is a true no-op: never pair an actually
                    # altered observation with an all-one quality target.
                    alpha = sample.new_zeros((1, height, width))
                    metadata = {"scope": "none", "zero_severity_noop": True}
                else:
                    sample, alpha, metadata = self._registry[name](
                        sample, severity, generator
                    )
                alpha = alpha.float().clamp(0.0, 1.0)
                if alpha.shape != (1, height, width):
                    raise RuntimeError(
                        f"{name} returned invalid mask shape {tuple(alpha.shape)}."
                    )
                quality = torch.minimum(quality, 1.0 - severity * alpha)
                degradation_mask = torch.maximum(degradation_mask, alpha)
                operation_masks[batch_index, operation_index] = alpha
                operation_ids[batch_index, operation_index] = (
                    self.SUPPORTED_OPERATIONS.index(name)
                )
                operation_severities[batch_index, operation_index] = severity
                type_names.append(name)
                parameter_rows.append(
                    {"type": name, "severity": severity, **metadata}
                )
            degraded_images.append(sample.clamp(0.0, 1.0))
            qualities.append(quality.clamp(0.0, 1.0))
            masks.append(degradation_mask.clamp(0.0, 1.0))
            all_types.append(type_names)
            all_params.append(parameter_rows)

        result: dict[str, Any] = {
            "image": torch.stack(degraded_images),
            "original_image": image_float.clone(),
            "quality_gt": torch.stack(qualities),
            "degradation_mask": torch.stack(masks),
            "severity": operation_severities.max(dim=1).values
            if operation_slots > 0
            else image_float.new_zeros((batch_size,)),
            "operation_masks": operation_masks,
            "operation_type_ids": operation_ids,
            "operation_severities": operation_severities,
            "degradation_type": all_types,
            "params": all_params,
        }
        if squeeze_batch:
            for key in (
                "image",
                "original_image",
                "quality_gt",
                "degradation_mask",
                "severity",
                "operation_masks",
                "operation_type_ids",
                "operation_severities",
            ):
                result[key] = result[key][0]
            result["degradation_type"] = result["degradation_type"][0]
            result["params"] = result["params"][0]
        return result



def _bool_config(value: object, name: str) -> bool:
    if not isinstance(value, bool):
        raise TypeError(f"{name} must be a bool, got {type(value).__name__}.")
    return value



def _config_value(container: object, name: str, default: Any) -> Any:
    """Read one value from either a mapping or argparse-style namespace."""

    if container is None:
        return default
    if isinstance(container, Mapping):
        return container.get(name, default)
    return getattr(container, name, default)



def _build_quality_degrader_from_args(
    args: object,
    *,
    seed: int,
) -> SyntheticQualityDegrader:
    """Resolve nested or flat synthetic degradation configuration."""

    nested = getattr(args, "quality_degradation", None)

    def value(name: str, default: Any) -> Any:
        nested_value = _config_value(nested, name, default)
        resolved = getattr(args, f"quality_{name}", nested_value)
        return getattr(args, f"quality_degradation_{name}", resolved)

    enabled: dict[str, bool] = {}
    probabilities: dict[str, Any] = {}
    for name in SyntheticQualityDegrader.SUPPORTED_OPERATIONS:
        raw = _config_value(nested, name, True)
        for flat_name in (f"quality_{name}", f"quality_degradation_{name}"):
            if hasattr(args, flat_name):
                raw = getattr(args, flat_name)
        explicit_probability = _config_value(
            nested, f"{name}_probability", None
        )
        for probability_name in (
            f"quality_{name}_probability",
            f"quality_degradation_{name}_probability",
        ):
            if hasattr(args, probability_name):
                explicit_probability = getattr(args, probability_name)
        if isinstance(raw, Mapping) or (
            raw is not None and not isinstance(raw, (bool, int, float, str))
        ):
            enabled[name] = _bool_config(
                _config_value(raw, "enabled", True), f"{name}.enabled"
            )
            probabilities[name] = _config_value(raw, "probability", 1.0)
        elif isinstance(raw, bool):
            enabled[name] = raw
            probabilities[name] = 1.0
        elif isinstance(raw, (int, float)) and not isinstance(raw, bool):
            enabled[name] = float(raw) > 0
            probabilities[name] = float(raw)
        else:
            raise TypeError(
                f"quality_degradation.{name} must be bool, probability, or mapping."
            )
        if explicit_probability is not None:
            probabilities[name] = explicit_probability

    return SyntheticQualityDegrader(
        apply_probability=value("apply_probability", 0.8),
        max_operations=value("max_operations", 2),
        min_severity=value("min_severity", 0.2),
        max_severity=value("max_severity", 1.0),
        enabled_operations=enabled,
        operation_probabilities=probabilities,
        seed=seed,
        local_probability=value("local_probability", 0.8),
    )



def _image_to_unit_interval(image: torch.Tensor) -> torch.Tensor:
    """Convert the repository's uint8-like tensors to pre-normalization [0,1]."""

    if not isinstance(image, torch.Tensor) or image.ndim != 4:
        raise ValueError("Training images must be tensors shaped [B,C,H,W].")
    image = image.float()
    if not bool(torch.isfinite(image).all().item()):
        raise ValueError("Training image contains NaN or Inf.")
    minimum = float(image.detach().amin().item())
    maximum = float(image.detach().amax().item())
    if minimum < 0.0:
        raise ValueError(f"Training image minimum must be >= 0, got {minimum}.")
    if maximum <= 1.0:
        return image
    if maximum <= 255.0:
        return image / 255.0
    raise ValueError(f"Training image maximum must be <= 255, got {maximum}.")


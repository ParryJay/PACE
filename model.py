"""PACE segmentation model for unregistered NBI/WLI pairs.

Encoder, decoder, heads and window reshaping are defined locally.
The same model and parameter names are used for both supported datasets.
No correspondence labels, p_C calibration or dataset agreement quantiles are used.
"""
from __future__ import annotations

import math
from typing import Any, Mapping, Sequence
import torch
from torch import nn
import torch.nn.functional as F
import torch.distributed as dist

try:
    import timm
except ImportError as exc:
    raise ImportError(
        "PACE requires `timm`. Install it with `pip install timm`."
    ) from exc


def _group_norm(num_channels: int) -> nn.GroupNorm:
    """Create GroupNorm with the largest valid group count up to 32."""
    for num_groups in range(min(32, num_channels), 0, -1):
        if num_channels % num_groups == 0:
            return nn.GroupNorm(num_groups, num_channels)
    raise RuntimeError("Unable to construct GroupNorm.")


class ConvNeXtBackbone(nn.Module):
    """Shared ConvNeXt backbone that extracts both modalities in one pass."""

    def __init__(
        self,
        model_name: str,
        pretrained: bool,
        out_indices: Sequence[int],
        d_model: int,
        freeze: bool,
    ) -> None:
        super().__init__()
        if not out_indices:
            raise ValueError("out_indices must contain at least one stage index.")

        self.backbone = timm.create_model(
            model_name,
            pretrained=pretrained,
            features_only=True,
            out_indices=tuple(out_indices),
        )
        if freeze:
            self.backbone.requires_grad_(False)

        self.projections = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(channels, d_model, kernel_size=1, bias=False),
                    _group_norm(d_model),
                    nn.GELU(),
                )
                for channels in self.backbone.feature_info.channels()
            ]
        )

        self.register_buffer(
            "image_mean",
            torch.tensor((0.485, 0.456, 0.406)).view(1, 3, 1, 1),
            persistent=False,
        )
        self.register_buffer(
            "image_std",
            torch.tensor((0.229, 0.224, 0.225)).view(1, 3, 1, 1),
            persistent=False,
        )

    def _normalize(self, images: torch.Tensor) -> torch.Tensor:
        if not images.is_floating_point():
            images = images.float()
        if images.detach().amax() > 1:
            images = images / 255.0
        mean = self.image_mean.to(dtype=images.dtype)
        std = self.image_std.to(dtype=images.dtype)
        return (images - mean) / std

    def forward(
        self,
        img_nbi: torch.Tensor,
        img_wli: torch.Tensor,
    ) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
        if img_nbi.shape != img_wli.shape:
            raise ValueError(
                "NBI and WLI inputs must have the same shape, got "
                f"{tuple(img_nbi.shape)} and {tuple(img_wli.shape)}."
            )

        batch_size = img_nbi.shape[0]
        images = torch.cat(
            [self._normalize(img_nbi), self._normalize(img_wli)], dim=0
        )
        features = self.backbone(images)

        feats_nbi: list[torch.Tensor] = []
        feats_wli: list[torch.Tensor] = []
        for projection, feature in zip(self.projections, features):
            feature = projection(feature)
            feats_nbi.append(feature[:batch_size])
            feats_wli.append(feature[batch_size:])
        return feats_nbi, feats_wli


class FPNSegFormerDecoder(nn.Module):
    """FPN-style multi-scale decoder used independently by each modality."""

    def __init__(
        self,
        in_channels: int,
        embed_dim: int,
        num_classes: int,
        num_scales: int,
        deep_context_channels: int | None = None,
    ) -> None:
        super().__init__()

        self.num_scales = num_scales
        self.lateral_convs = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(in_channels, embed_dim, kernel_size=1, bias=False),
                    _group_norm(embed_dim),
                    nn.ReLU(inplace=True),
                )
                for _ in range(num_scales)
            ]
        )
        self.fpn_convs = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(
                        embed_dim,
                        embed_dim,
                        kernel_size=3,
                        padding=1,
                        bias=False,
                    ),
                    _group_norm(embed_dim),
                    nn.ReLU(inplace=True),
                )
                for _ in range(num_scales)
            ]
        )
        self.fuse_conv = nn.Sequential(
            nn.Conv2d(
                embed_dim * num_scales,
                embed_dim,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            _group_norm(embed_dim),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=0.1),
            nn.Conv2d(
                embed_dim,
                embed_dim,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            _group_norm(embed_dim),
            nn.ReLU(inplace=True),
        )
        self.classifier = nn.Sequential(
            nn.Conv2d(
                embed_dim,
                embed_dim,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            _group_norm(embed_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(embed_dim, num_classes, kernel_size=1),
        )
        # Inject F4 semantics into the F3 lateral without changing the shapes
        # of the three-scale decoder weights.
        self.deep_context_projection: nn.Module | None = None
        self.deep_context_logit_scale: nn.Parameter | None = None
        if deep_context_channels is not None:
            self.deep_context_projection = nn.Sequential(
                nn.Conv2d(
                    deep_context_channels,
                    embed_dim,
                    kernel_size=1,
                    bias=False,
                ),
                _group_norm(embed_dim),
                nn.GELU(),
                nn.Conv2d(
                    embed_dim,
                    embed_dim,
                    kernel_size=3,
                    padding=1,
                    bias=False,
                ),
                _group_norm(embed_dim),
            )
            self.deep_context_logit_scale = nn.Parameter(torch.tensor(-2.0))
        self._init_weights()

    def _init_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(
                    module.weight, mode="fan_out", nonlinearity="relu"
                )
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
        # A low-channel logit projection needs its own scale: fan_out Kaiming
        # otherwise makes the initial two-class logits strongly overconfident.
        nn.init.normal_(self.classifier[-1].weight, std=0.01)

    def forward(
        self,
        features: Sequence[torch.Tensor],
        output_size: tuple[int, int],
        deep_context: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if len(features) != self.num_scales:
            raise ValueError(
                f"Expected {self.num_scales} feature maps, got {len(features)}."
            )

        laterals = [
            lateral(feature)
            for lateral, feature in zip(self.lateral_convs, features)
        ]
        if deep_context is not None:
            if (
                self.deep_context_projection is None
                or self.deep_context_logit_scale is None
            ):
                raise RuntimeError(
                    "deep_context was supplied to a decoder without a deep adapter."
                )
            context = self.deep_context_projection(deep_context)
            context = F.interpolate(
                context,
                size=laterals[-1].shape[-2:],
                mode="bilinear",
                align_corners=False,
            )
            context_scale = torch.sigmoid(self.deep_context_logit_scale).to(
                laterals[-1].dtype
            )
            laterals[-1] = laterals[-1] + context_scale * context
        fpn_features = list(laterals)
        for level in range(self.num_scales - 2, -1, -1):
            upsampled = F.interpolate(
                fpn_features[level + 1],
                size=laterals[level].shape[-2:],
                mode="bilinear",
                align_corners=False,
            )
            fpn_features[level] = laterals[level] + upsampled

        fpn_features = [
            convolution(feature)
            for convolution, feature in zip(self.fpn_convs, fpn_features)
        ]
        highest_resolution = fpn_features[0].shape[-2:]
        aligned_features = [
            feature
            if feature.shape[-2:] == highest_resolution
            else F.interpolate(
                feature,
                size=highest_resolution,
                mode="bilinear",
                align_corners=False,
            )
            for feature in fpn_features
        ]

        decoder_feature = self.fuse_conv(torch.cat(aligned_features, dim=1))
        classifier_feature = self.classifier[:-1](decoder_feature)
        # The unnormalized two-class projection is the decoder operation most
        # likely to overflow FP16 after many epochs (its preceding feature is
        # normalized, but the learned 1x1 weights are not).  Keep only this
        # inexpensive final projection in float32; returned logits and all
        # segmentation losses then remain finite without promoting the full FPN.
        with torch.autocast(device_type=decoder_feature.device.type, enabled=False):
            logits = self.classifier[-1](classifier_feature.float())
        return F.interpolate(
            logits,
            size=output_size,
            mode="bilinear",
            align_corners=False,
        )


class PreliminaryHead(nn.Module):
    """Predict one coarse logit map without changing the decoder pathway."""

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        out_channels: int = 1,
    ) -> None:
        super().__init__()
        if min(in_channels, hidden_channels, out_channels) < 1:
            raise ValueError("All preliminary-head channel counts must be positive.")
        self.layers = nn.Sequential(
            nn.Conv2d(
                in_channels,
                hidden_channels,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            _group_norm(hidden_channels),
            nn.GELU(),
            nn.Conv2d(hidden_channels, out_channels, kernel_size=1),
        )
        self._init_weights()

    def _init_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(
                    module.weight, mode="fan_out", nonlinearity="relu"
                )
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
        # Q/B/V start near probability 0.5 rather than saturating from the
        # hidden-to-one-channel fan_out initialization.
        nn.init.normal_(self.layers[-1].weight, std=0.01)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return logits at the input feature map's spatial resolution."""

        if x.ndim != 4:
            raise ValueError(
                f"PreliminaryHead expects [B,C,H,W], got {tuple(x.shape)}."
            )
        return self.layers(x)


class _WindowPartition:
    """Padded window reshaping shared by PACE local attention blocks."""

    @staticmethod
    def _partition(
        value: torch.Tensor,
        window_size: int,
        offset: int,
    ) -> tuple[torch.Tensor, tuple[int, int, int, int, int, int], torch.Tensor]:
        batch_size, channels, height, width = value.shape
        padded_height = math.ceil((height + offset) / window_size) * window_size
        padded_width = math.ceil((width + offset) / window_size) * window_size
        pad_bottom = padded_height - height - offset
        pad_right = padded_width - width - offset
        padded = F.pad(value, (offset, pad_right, offset, pad_bottom))
        windows = padded.reshape(
            batch_size,
            channels,
            padded_height // window_size,
            window_size,
            padded_width // window_size,
            window_size,
        ).permute(0, 2, 4, 3, 5, 1).reshape(
            -1, window_size * window_size, channels
        )
        valid = value.new_ones((batch_size, 1, height, width))
        valid = F.pad(valid, (offset, pad_right, offset, pad_bottom))
        valid_windows = valid.reshape(
            batch_size,
            1,
            padded_height // window_size,
            window_size,
            padded_width // window_size,
            window_size,
        ).permute(0, 2, 4, 3, 5, 1).reshape(-1, window_size * window_size)
        metadata = (
            batch_size,
            channels,
            height,
            width,
            padded_height,
            padded_width,
        )
        return windows, metadata, valid_windows < 0.5

    @staticmethod
    def _unpartition(
        windows: torch.Tensor,
        metadata: tuple[int, int, int, int, int, int],
        window_size: int,
        offset: int,
    ) -> torch.Tensor:
        batch_size, channels, height, width, padded_height, padded_width = metadata
        value = windows.reshape(
            batch_size,
            padded_height // window_size,
            padded_width // window_size,
            window_size,
            window_size,
            channels,
        ).permute(0, 5, 1, 3, 2, 4).reshape(
            batch_size, channels, padded_height, padded_width
        )
        return value[:, :, offset : offset + height, offset : offset + width]


def opinion_from_evidence(evidence: torch.Tensor) -> torch.Tensor:
    """Positive/negative evidence -> (positive belief, negative belief, unknown)."""
    strength = evidence.sum(-1, keepdim=True) + 2.0
    return torch.cat((evidence, torch.full_like(strength, 2.0)), -1) / strength


def discount(reliability: torch.Tensor, opinion: torch.Tensor) -> torch.Tensor:
    reliability = torch.as_tensor(reliability, device=opinion.device, dtype=opinion.dtype)
    while reliability.ndim < opinion.ndim:
        reliability = reliability.unsqueeze(-1)
    return torch.cat((reliability * opinion[..., :2],
                      1.0 - reliability * (1.0 - opinion[..., 2:])), -1)


def graph_edges(position, q, boundary, distance, theta=(1., 1., 1., 1.)):
    # Direct differences avoid cancellation in cdist's matrix-product path
    # when 128 soft nodes have nearly coincident centroids.
    edges = theta[0] * torch.cdist(position, position, compute_mode='donot_use_mm_for_euclid_dist')
    for weight, value in zip(theta[1:], (q, boundary, distance)):
        edges = edges + weight * (value.unsqueeze(2) - value.unsqueeze(1)).abs()
    return edges / edges.amax((1, 2), keepdim=True).detach().clamp_min(1e-8)


class LesionGraphBuilder(nn.Module):
    """Shared Q/B/V heads, soft assignment and lesion-centric node attributes."""
    def __init__(self, channels=256, token_dim=256, num_nodes=128,
                 hidden_channels=128, theta=(1., 1., 1., 1.),
                 node_spatial_strength=8.):
        super().__init__()
        if num_nodes < 1:
            raise ValueError('num_nodes must be positive')
        if not math.isfinite(node_spatial_strength) or node_spatial_strength <= 0:
            raise ValueError('node_spatial_strength must be finite and positive')
        self.num_nodes, self.theta = num_nodes, theta
        self.q_head = PreliminaryHead(channels, hidden_channels)
        self.b_head = PreliminaryHead(channels, hidden_channels)
        self.v_head = PreliminaryHead(channels, hidden_channels)
        self.visual = nn.Conv2d(channels, num_nodes, 1)
        self.structured = nn.Sequential(nn.Conv2d(5, hidden_channels, 3, padding=1),
                                        nn.GELU(), nn.Conv2d(hidden_channels, num_nodes, 1))
        rows = math.isqrt(num_nodes)
        cols = math.ceil(num_nodes / rows)
        yy, xx = torch.meshgrid(torch.linspace(0, 1, rows),
                                torch.linspace(0, 1, cols), indexing='ij')
        # Row-major order on the closest rectangular covering grid.
        grid = torch.stack((xx, yy), -1).reshape(-1, 2)[:num_nodes]
        self.position_prototypes = nn.Parameter(grid)
        # A stronger spatial prior prevents all soft nodes from initially
        # pooling the whole image. Keep the existing learnable parameter/key.
        inverse_softplus = node_spatial_strength + math.log(-math.expm1(-node_spatial_strength))
        self.raw_tau = nn.Parameter(torch.tensor(inverse_softplus))
        self.phi_f = nn.Linear(channels, token_dim)
        self.phi_p = nn.Linear(2, token_dim)
        self.phi_s = nn.Sequential(nn.Linear(3, token_dim), nn.GELU(),
                                   nn.Linear(token_dim, token_dim))
        self.modality_embedding = nn.Parameter(torch.zeros(2, 1, token_dim))
        nn.init.normal_(self.modality_embedding, std=.02)
        self.norm = nn.LayerNorm(token_dim)
        self.mass_head = nn.Sequential(nn.Linear(3, 32), nn.GELU(), nn.Linear(32, 1))
        self.evidence_head = nn.Sequential(nn.Linear(token_dim, hidden_channels),
                                          nn.GELU(), nn.Linear(hidden_channels, 2))

    def embed(self, feature, position, q, boundary, distance, modality):
        base = (self.phi_f(feature) + self.phi_s(torch.stack((q, boundary, distance), -1))
                + self.modality_embedding[modality])
        return self.norm(base + self.phi_p(position)), self.norm(base)

    def forward(self, features, modality):
        batch, _, height, width = features.shape
        ql, bl, vl = self.q_head(features), self.b_head(features), self.v_head(features)
        # The graph arithmetic and evidence are kept in float32 under AMP.
        with torch.autocast(device_type=features.device.type, enabled=False):
            q, boundary, quality = ql.float().sigmoid(), bl.float().sigmoid(), vl.float().sigmoid()
            yy, xx = torch.meshgrid(
                torch.linspace(0, 1, height, device=features.device),
                torch.linspace(0, 1, width, device=features.device), indexing='ij')
            position = torch.stack((xx, yy), 0).unsqueeze(0).expand(batch, -1, -1, -1)
            spatial_cost = ((position[:, None] - self.position_prototypes.clamp(0, 1)
                            [None, :, :, None, None]) ** 2).sum(2)
            assignment = (self.visual(features.float()) + self.structured(
                torch.cat((q, boundary, quality, position), 1))
                - F.softplus(self.raw_tau) * spatial_cost).softmax(1)
            af = assignment.flatten(2)
            mass = af.sum(-1)
            values = torch.cat((features.float(), q, boundary, position, quality), 1).flatten(2)
            pooled = af.bmm(values.transpose(1, 2)) / mass.unsqueeze(-1).clamp_min(1e-8)
            channels = features.shape[1]
            feature, nq, nb = pooled[..., :channels], pooled[..., channels], pooled[..., channels+1]
            np, nv = pooled[..., channels+2:channels+4], pooled[..., channels+4]
            lesion_mass = mass * nq
            total = lesion_mass.sum(1, keepdim=True)
            center = (lesion_mass.unsqueeze(-1) * np).sum(1) / total.clamp_min(1e-8)
            center = torch.where(total >= 1e-8, center, torch.full_like(center, .5))
            distance = torch.linalg.vector_norm(np - center[:, None], dim=-1) / math.sqrt(2)
            tokens, match_tokens = self.embed(feature, np, nq, nb, distance, modality)
            a = F.softplus(self.mass_head(torch.stack((mass/(height*width), nq, nb), -1))).squeeze(-1)
            a = a / a.sum(1, keepdim=True).clamp_min(1e-8)
            evidence = F.softplus(self.evidence_head(tokens))
            opinion = opinion_from_evidence(evidence)
            result = dict(assignment=assignment, mass=mass, a=a, feature=feature,
                          q=nq, b=nb, v=nv, position=np, distance=distance, tokens=tokens,
                          match_tokens=match_tokens, evidence=evidence, opinion=opinion,
                          discounted_opinion=discount(nv, opinion),
                          G=graph_edges(np, nq, nb, distance, self.theta),
                          Q=q, B=boundary, V=quality, q_logits=ql.float(),
                          b_logits=bl.float(), v_logits=vl.float())
        return result


class PairEstimator(nn.Module):
    def __init__(self, channels=256, rho_min=.2, rho_max=.85,
                 alpha_min=.05, alpha_max=.35):
        super().__init__()
        for low, high in ((rho_min, rho_max), (alpha_min, alpha_max)):
            if not 0 <= low < high <= 1:
                raise ValueError('Pair estimator intervals must lie in [0,1]')
        self.rho_min, self.rho_max = rho_min, rho_max
        self.alpha_min, self.alpha_max = alpha_min, alpha_max
        self.mlp = nn.Sequential(nn.Linear(2*channels+14, 128), nn.GELU(), nn.Linear(128, 2))
        nn.init.zeros_(self.mlp[-1].bias)

    @staticmethod
    def statistics(graph):
        weights = graph['mass'] / graph['mass'].sum(1, keepdim=True).clamp_min(1e-8)
        values = torch.stack((graph['q'], graph['b'], graph['v']), -1)
        mean = (weights[..., None] * values).sum(1)
        variance = (weights[..., None] * (values-mean[:, None]).square()).sum(1)
        return torch.cat((mean, variance.clamp_min(1e-12).sqrt(),
                          (graph['q'] > .5).float().mean(1, keepdim=True)), 1)

    def forward(self, f4_nbi, f4_wli, graph_nbi, graph_wli):
        with torch.autocast(device_type=f4_nbi.device.type, enabled=False):
            pair = torch.cat((f4_nbi.float().mean((2, 3)), f4_wli.float().mean((2, 3)),
                              self.statistics(graph_nbi), self.statistics(graph_wli)), 1)
            raw = self.mlp(pair).sigmoid()
            return (self.rho_min+(self.rho_max-self.rho_min)*raw[:, 0],
                    self.alpha_min+(self.alpha_max-self.alpha_min)*raw[:, 1])


def _feasible_partial(plan, a, b, rho):
    """Round finite-iteration numerical residuals into the partial OT polytope."""
    # Finite Sinkhorn iterations do not guarantee exact marginals.
    # Differentiable marginal rounding enforces the stated hard constraints;
    # it converges to the identity as the Sinkhorn residual goes to zero.
    plan = plan * (a / plan.sum(2).clamp_min(1e-30)).clamp(max=1).unsqueeze(2)
    plan = plan * (b / plan.sum(1).clamp_min(1e-30)).clamp(max=1).unsqueeze(1)
    total = plan.sum((1, 2))
    plan = plan * (rho / total.clamp_min(1e-30)).clamp(max=1)[:, None, None]
    left = (a-plan.sum(2)).clamp_min(0)
    right = (b-plan.sum(1)).clamp_min(0)
    missing = (rho-plan.sum((1, 2))).clamp_min(0)
    fill = left[:, :, None] * right[:, None, :]
    return plan + fill * (missing/fill.sum((1, 2)).clamp_min(1e-30))[:, None, None]


def pfgw_solve(cost, graph_nbi, graph_wli, a_nbi, a_wli, rho, alpha,
               epsilon=.05, outer_iterations=5, inner_iterations=50):
    """Batched, unrolled partial FGW, with a forbidden dummy-to-dummy edge."""
    if epsilon <= 0 or min(outer_iterations, inner_iterations) < 1:
        raise ValueError('Positive epsilon and iteration counts required')
    with torch.autocast(device_type=cost.device.type, enabled=False):
        cost, gn, gw = cost.float(), graph_nbi.float(), graph_wli.float()
        a, b = a_nbi.float(), a_wli.float()
        rho, alpha = rho.float().reshape(-1), alpha.float().reshape(-1, 1, 1)
        plan = rho[:, None, None] * a[:, :, None] * b[:, None, :]
        semantic = cost / cost.amax((1, 2), keepdim=True).detach().clamp_min(1e-8)
        dummy = (1-rho).unsqueeze(1)
        la = torch.cat((a, dummy), 1).clamp_min(1e-30).log()
        lb = torch.cat((b, dummy), 1).clamp_min(1e-30).log()
        for _ in range(outer_iterations):
            row, col = plan.sum(2), plan.sum(1)
            structure = (gn.square().bmm(row.unsqueeze(2))
                         + gw.square().bmm(col.unsqueeze(2)).transpose(1, 2)
                         - 2*gn.bmm(plan).bmm(gw.transpose(1, 2)))
            structure = structure / structure.amax((1, 2), keepdim=True).detach().clamp_min(1e-8)
            effective = (1-alpha)*semantic + alpha*structure
            kernel = F.pad(-effective/epsilon, (0, 1, 0, 1))
            forbidden = torch.zeros_like(kernel, dtype=torch.bool)
            forbidden[:, -1, -1] = True
            kernel = kernel.masked_fill(forbidden, -1e6)
            u, v = torch.zeros_like(la), torch.zeros_like(lb)
            for _ in range(inner_iterations):
                u = la - torch.logsumexp(kernel + v[:, None, :], 2)
                v = lb - torch.logsumexp(kernel + u[:, :, None], 1)
            plan = (kernel+u[:, :, None]+v[:, None, :]).exp()[:, :-1, :-1]
            plan = _feasible_partial(plan, a, b, rho)
        return plan


def transport_statistics(plan, marginal):
    mass = plan.sum(2)
    fraction = (mass/marginal.clamp_min(1e-8)).clamp(0, 1)
    weights = torch.where((mass >= 1e-8).unsqueeze(-1),
                          plan/mass.unsqueeze(-1).clamp_min(1e-8),
                          torch.full_like(plan, 1/plan.shape[2]))
    return fraction, weights


def evidence_message(fraction, weights, source_opinion):
    return discount(fraction, weights.bmm(source_opinion))


class CorrespondenceAudit(nn.Module):
    """Parameter-free, per-pair/direction kappa audit with checkpointed EB prior."""
    def __init__(self, tau=.3, momentum=.99):
        super().__init__()
        self.tau, self.momentum = tau, momentum
        self.register_buffer('mu', torch.tensor(.5))
        self.register_buffer('s0_sq', torch.tensor(.05))

    @torch.no_grad()
    def statistics(self, own, message, fraction):
        eligible = (fraction >= .5) & (own[..., 2] < self.tau) & (message[..., 2] < self.tau)
        labels_o, labels_c = own[..., 0] > own[..., 1], message[..., 0] > message[..., 1]
        k = eligible.sum(1)
        denom = k.clamp_min(1).float()
        p_obs = ((labels_o == labels_c) & eligible).sum(1)/denom
        ro, rc = (labels_o & eligible).sum(1)/denom, (labels_c & eligible).sum(1)/denom
        pe = ro*rc+(1-ro)*(1-rc)
        gap = 1-pe
        eta = ((1-p_obs)/gap.clamp_min(1e-8)).clamp(0, 1)
        half = .5/denom
        corrected = torch.minimum(torch.maximum(p_obs, half), 1-half)
        se2 = corrected*(1-corrected)/(denom*gap.square().clamp_min(1e-16))
        return dict(k=k, p_obs=p_obs, p_e=pe, eta=eta, se2=se2.clamp_min(1e-8),
                    fallback=(k < 4) | (gap < .05))

    @torch.no_grad()
    def forward(self, own_nbi, msg_nbi, pi_nbi, own_wli, msg_wli, pi_wli):
        audits = [self.statistics(own_nbi, msg_nbi, pi_nbi),
                  self.statistics(own_wli, msg_wli, pi_wli)]
        if self.training:
            eta = torch.cat([x['eta'] for x in audits])
            se2 = torch.cat([x['se2'] for x in audits])
            valid = ~torch.cat([x['fallback'] for x in audits])
            # Count, sum and sum of squares give identical priors on every DDP rank.
            stats = torch.stack((valid.sum().float(), (eta*valid).sum(),
                                 (eta.square()*valid).sum(), (se2*valid).sum()))
            if dist.is_available() and dist.is_initialized():
                dist.all_reduce(stats)
            if stats[0] > 0:
                mean = stats[1]/stats[0]
                variance = (stats[2]/stats[0]-mean.square()-stats[3]/stats[0]).clamp_min(1e-4)
                self.mu.lerp_(mean, 1-self.momentum)
                self.s0_sq.lerp_(variance, 1-self.momentum)
        for audit in audits:
            se2 = audit['se2']
            variance = (1/se2+1/self.s0_sq).reciprocal()
            mean = variance*(audit['eta']/se2+self.mu/self.s0_sq)
            eta_plus = (mean+variance.sqrt()).clamp(0, 1)
            audit['eta_plus'] = torch.where(audit['fallback'], torch.full_like(eta_plus, .5), eta_plus)
            audit['t'] = (1-audit['eta_plus']).detach()
        flagged = (audits[0]['eta_plus'] > .6) | (audits[1]['eta_plus'] > .6)
        for audit in audits:
            audit['flagged'] = flagged
        return audits


def route_weights(own, message, reliability):
    calibrated = discount(reliability, message)
    conflict = own[..., 0]*calibrated[..., 1] + own[..., 1]*calibrated[..., 0]
    gamma = torch.stack((1-calibrated[..., 2]-conflict, calibrated[..., 2], conflict), -1)
    assert torch.isfinite(gamma).all(), 'Non-finite TSPR weights'
    assert (gamma >= -2e-6).all() and (gamma <= 1+2e-6).all(), 'TSPR outside simplex'
    assert torch.allclose(gamma.sum(-1), torch.ones_like(gamma[..., 0]), atol=2e-6)
    assert (gamma[..., 0] <= reliability[:, None]+2e-6).all(), 'Shared routing exceeds reliability'
    return gamma


def resized_assignment(assignment, size):
    value = F.interpolate(assignment, size=size, mode='bilinear', align_corners=False)
    return value/value.sum(1, keepdim=True).clamp_min(1e-8)


def spatial_routes(assignment, gamma, size):
    a = resized_assignment(assignment, size)
    return torch.einsum('bnhw,bnk->bkhw', a, gamma)


def _expert(dim):
    return nn.Sequential(nn.Linear(dim, 2*dim), nn.GELU(), nn.Linear(2*dim, dim), nn.LayerNorm(dim))


class _BiasedAttention(nn.Module):
    def __init__(self, dim, num_heads=4):
        super().__init__()
        if dim % num_heads:
            raise ValueError('Attention dimension must divide num_heads')
        self.num_heads, self.head_dim = num_heads, dim//num_heads
        self.q, self.k, self.v, self.out = [nn.Linear(dim, dim) for _ in range(4)]

    def forward(self, query, context, bias=None, padding=None):
        b, n, dim = query.shape
        shape = lambda x: x.reshape(b, -1, self.num_heads, self.head_dim).transpose(1, 2)
        q, k, v = shape(self.q(query)), shape(self.k(context)), shape(self.v(context))
        with torch.autocast(device_type=query.device.type, enabled=False):
            logits = q.float() @ k.float().transpose(-1, -2) / math.sqrt(self.head_dim)
            if bias is not None:
                logits = logits+bias.float().unsqueeze(1)
            if padding is not None:
                logits = logits.masked_fill(padding[:, None, None, :], -1e4)
            z = logits.softmax(-1) @ v.float()
        return self.out(z.transpose(1, 2).reshape(b, n, dim).to(query.dtype))


class TSPRGlobal(nn.Module):
    def __init__(self, dim=256, num_heads=4):
        super().__init__()
        self.attention = _BiasedAttention(dim, num_heads)
        self.shared, self.private = _expert(dim), _expert(dim)

    def forward(self, x, other, assignment, other_assignment, weights, reliability, gamma):
        size = x.shape[-2:]
        a = resized_assignment(assignment, size).flatten(2)
        oa = resized_assignment(other_assignment, other.shape[-2:]).flatten(2)
        r = oa/oa.sum(2, keepdim=True).clamp_min(1e-8)
        prior = a.transpose(1, 2).bmm(weights).bmm(r)
        prior = prior/prior.sum(2, keepdim=True).clamp_min(1e-8)
        bias = ((1-reliability[:, None, None])/other.shape[-2]/other.shape[-1]
                + reliability[:, None, None]*prior+1e-8).log()
        query, context = x.flatten(2).transpose(1, 2), other.flatten(2).transpose(1, 2)
        z = self.attention(query, context, bias)
        routes = spatial_routes(assignment, gamma, size).flatten(2).transpose(1, 2)
        updated = query+routes[..., :1]*self.shared(z)+routes[..., 1:2]*self.private(query)
        return updated.transpose(1, 2).reshape_as(x), z.transpose(1, 2).reshape_as(x)


class TSPRNode(nn.Module):
    def __init__(self, feature_dim=256, token_dim=256, num_heads=4):
        super().__init__()
        self.topdown = nn.Linear(feature_dim, token_dim)
        self.attention = _BiasedAttention(token_dim, num_heads)
        self.shared, self.private = _expert(token_dim), _expert(token_dim)

    def context(self, z4, assignment):
        a = resized_assignment(assignment, z4.shape[-2:]).flatten(2)
        r = a/a.sum(2, keepdim=True).clamp_min(1e-8)
        return self.topdown(r.bmm(z4.flatten(2).transpose(1, 2)))

    def forward(self, tokens, other_tokens, u, other_u, weights, gamma):
        z = self.attention(tokens+u, other_tokens+other_u, (weights+1e-8).log())
        updated = tokens+gamma[..., :1]*self.shared(z)+gamma[..., 1:2]*self.private(tokens)
        return updated, z


class TSPRWindow(nn.Module):
    """Two window blocks, offsets 0/4; padding avoids cyclic wraparound leakage."""
    def __init__(self, dim=256, num_heads=4, window_size=8):
        super().__init__()
        self.window_size = window_size
        self.attentions = nn.ModuleList([_BiasedAttention(dim, num_heads) for _ in range(2)])
        self.shared, self.private = _expert(dim), _expert(dim)

    def forward(self, x, u, other, other_u, assignment, gamma, kv_mode='self'):
        query = x+u
        context = query if kv_mode == 'self' else other+other_u
        z = query
        for attention, offset in zip(self.attentions, (0, self.window_size//2)):
            qw, meta, _ = _WindowPartition._partition(z, self.window_size, offset)
            kw, _, padding = _WindowPartition._partition(context, self.window_size, offset)
            zw = attention(qw, kw, padding=padding)
            z = _WindowPartition._unpartition(zw, meta, self.window_size, offset)
        base = x.permute(0, 2, 3, 1)
        routes = spatial_routes(assignment, gamma, x.shape[-2:]).permute(0, 2, 3, 1)
        updated = base+routes[..., :1]*self.shared(z.permute(0, 2, 3, 1))+routes[..., 1:2]*self.private(base)
        return updated.permute(0, 3, 1, 2)


def decision_llr(probability, eta, p_les):
    p = probability.clamp(1e-4, 1-1e-4)
    base = torch.as_tensor(p_les, dtype=p.dtype, device=p.device).clamp(1e-4, 1-1e-4)
    eta = torch.as_tensor(eta, dtype=p.dtype, device=p.device)
    while eta.ndim < p.ndim:
        eta = eta.unsqueeze(-1)
    return ((1-eta)*p/base+eta).log() - ((1-eta)*(1-p)/(1-base)+eta).log()


class _ModalityAdapter(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.adapter = nn.Sequential(nn.Conv2d(dim, max(dim//4, 4), 1), nn.GELU(),
                                     nn.Conv2d(max(dim//4, 4), dim, 1))
        nn.init.zeros_(self.adapter[-1].weight)
        nn.init.zeros_(self.adapter[-1].bias)

    def forward(self, x):
        return x+self.adapter(x)


class PACE(nn.Module):
    def __init__(self, encoder_name='convnextv2_base', encoder_pretrained=True,
                 d_model=256, token_dim=None, num_nodes=128, hidden_channels=128,
                 freeze_backbone=False, rho_min=.2, rho_max=.85,
                 alpha_min=.05, alpha_max=.35, epsilon=.05, outer_iterations=5,
                 inner_iterations=50, tau=.5, theta=(1., 1., 1., 1.),
                 lambda_f=1., lambda_s=.5, lambda_b=.5, match_use_abs_pos=False,
                 opinion_grad='detach', l2_kv='self', decision_fusion='add',
                 force_t=None, num_heads=4, p_les=.5, node_spatial_strength=8.):
        super().__init__()
        if opinion_grad not in ('detach', 'through') or l2_kv not in ('self', 'cross'):
            raise ValueError('Invalid opinion_grad/l2_kv')
        if decision_fusion not in ('add', 'off'):
            raise ValueError('Invalid decision_fusion')
        if force_t is not None and not 0 <= force_t <= 1:
            raise ValueError('force_t must be in [0,1]')
        if not 0 < tau <= 1 or min(theta) < 0 or min(lambda_f, lambda_s, lambda_b) < 0:
            raise ValueError('Invalid graph/audit/cost settings')
        if not math.isfinite(node_spatial_strength) or node_spatial_strength <= 0:
            raise ValueError('node_spatial_strength must be finite and positive')
        token_dim = token_dim or d_model
        self.encoder = ConvNeXtBackbone(encoder_name, encoder_pretrained, (0, 1, 2, 3), d_model, freeze_backbone)
        # Identity-initialized modality adapters preserve initial encoder features.
        self.adapters_nbi = nn.ModuleList([_ModalityAdapter(d_model) for _ in range(4)])
        self.adapters_wli = nn.ModuleList([_ModalityAdapter(d_model) for _ in range(4)])
        self.decoder_nbi = FPNSegFormerDecoder(d_model, d_model, 2, 3, d_model)
        self.decoder_wli = FPNSegFormerDecoder(d_model, d_model, 2, 3, d_model)
        self.graph_builder = LesionGraphBuilder(
            d_model, token_dim, num_nodes, hidden_channels, theta,
            node_spatial_strength=node_spatial_strength,
        )
        self.pair_estimator = PairEstimator(d_model, rho_min, rho_max, alpha_min, alpha_max)
        self.psi_nbi, self.psi_wli = nn.Linear(token_dim, token_dim), nn.Linear(token_dim, token_dim)
        self.audit = CorrespondenceAudit(tau)
        self.global_fusion = TSPRGlobal(d_model, num_heads)
        self.node_fusion = TSPRNode(d_model, token_dim, num_heads)
        self.window_fusion = TSPRWindow(d_model, num_heads)
        self.back_projection = nn.Linear(token_dim, d_model, bias=False)
        self.u2_projection = nn.Conv2d(token_dim, d_model, 1)
        self.register_buffer('p_les', torch.tensor(float(p_les)))
        self.match_use_abs_pos, self.opinion_grad = match_use_abs_pos, opinion_grad
        self.l2_kv, self.decision_fusion, self.force_t = l2_kv, decision_fusion, force_t
        self.solver_config = dict(epsilon=epsilon, outer_iterations=outer_iterations, inner_iterations=inner_iterations)
        self.cost_weights = lambda_f, lambda_s, lambda_b

    @property
    def scope_state_prefixes(self):
        # Preserve reference optimizer grouping and fusion-only stage selection.
        return ('adapters_', 'graph_builder.visual.', 'graph_builder.structured.',
                'graph_builder.position_prototypes', 'graph_builder.raw_tau',
                'graph_builder.phi_', 'graph_builder.modality_embedding',
                'graph_builder.norm.', 'graph_builder.mass_head.',
                'graph_builder.evidence_head.', 'pair_estimator.', 'psi_',
                'global_fusion.', 'node_fusion.', 'window_fusion.',
                'back_projection.', 'u2_projection.',
                'decoder_nbi.deep_context_', 'decoder_wli.deep_context_')

    def semantic_cost(self, nbi, wli):
        key = 'tokens' if self.match_use_abs_pos else 'match_tokens'
        fn = F.normalize(self.psi_nbi(nbi[key]), dim=-1)
        fw = F.normalize(self.psi_wli(wli[key]), dim=-1)
        cosine = 1-fn.bmm(fw.transpose(1, 2))
        def roles(graph):
            q, b = graph['q'], graph['b']
            return torch.stack((q*(1-b), b, (1-q)*(1-b)), -1).clamp_min(1e-8)
        pn, pw = roles(nbi)[:, :, None], roles(wli)[:, None, :]
        middle = .5*(pn+pw)
        js = .5*((pn*(pn.log()-middle.log())).sum(-1)+(pw*(pw.log()-middle.log())).sum(-1))
        boundary = (nbi['b'][:, :, None]-wli['b'][:, None, :]).abs()
        lf, ls, lb = self.cost_weights
        return lf*cosine+ls*js+lb*boundary

    def forward(self, img_nbi, img_wli, return_dict=True):
        fn, fw = self.encoder(img_nbi, img_wli)
        fn = [adapter(x) for adapter, x in zip(self.adapters_nbi, fn)]
        fw = [adapter(x) for adapter, x in zip(self.adapters_wli, fw)]
        gn, gw = self.graph_builder(fn[2], 0), self.graph_builder(fw[2], 1)
        with torch.autocast(device_type=img_nbi.device.type, enabled=False):
            rho, alpha = self.pair_estimator(fn[3], fw[3], gn, gw)
            cost = self.semantic_cost(gn, gw)
            plan = pfgw_solve(cost, gn['G'], gw['G'], gn['a'], gw['a'], rho, alpha, **self.solver_config)
            pn, wn = transport_statistics(plan, gn['a'])
            pw, ww = transport_statistics(plan.transpose(1, 2), gw['a'])
            own_n, own_w = gn['discounted_opinion'], gw['discounted_opinion']
            if self.opinion_grad == 'detach':
                own_n, own_w = own_n.detach(), own_w.detach()
            mn, mw = evidence_message(pn, wn, own_w), evidence_message(pw, ww, own_n)
            an, aw = self.audit(own_n, mn, pn, own_w, mw, pw)
            for audit in (an, aw):
                audit['t_audit'] = audit['t']
                if self.force_t is not None:
                    audit['t'] = torch.full_like(audit['t'], self.force_t)
            gamma_n, gamma_w = route_weights(own_n, mn, an['t']), route_weights(own_w, mw, aw['t'])
        f4n, z4n = self.global_fusion(fn[3], fw[3], gn['assignment'], gw['assignment'], wn, an['t'], gamma_n)
        f4w, z4w = self.global_fusion(fw[3], fn[3], gw['assignment'], gn['assignment'], ww, aw['t'], gamma_w)
        u3n, u3w = self.node_fusion.context(z4n, gn['assignment']), self.node_fusion.context(z4w, gw['assignment'])
        tn, z3n = self.node_fusion(gn['tokens'], gw['tokens'], u3n, u3w, wn, gamma_n)
        tw, z3w = self.node_fusion(gw['tokens'], gn['tokens'], u3w, u3n, ww, gamma_w)
        f3n = fn[2]+torch.einsum('bnhw,bnc->bchw', gn['assignment'], self.back_projection(tn-gn['tokens']))
        f3w = fw[2]+torch.einsum('bnhw,bnc->bchw', gw['assignment'], self.back_projection(tw-gw['tokens']))
        u2n = F.interpolate(self.u2_projection(torch.einsum('bnhw,bnc->bchw', gn['assignment'], z3n)),
                            size=fn[1].shape[-2:], mode='bilinear', align_corners=False)
        u2w = F.interpolate(self.u2_projection(torch.einsum('bnhw,bnc->bchw', gw['assignment'], z3w)),
                            size=fw[1].shape[-2:], mode='bilinear', align_corners=False)
        f2n = self.window_fusion(fn[1], u2n, fw[1], u2w, gn['assignment'], gamma_n, self.l2_kv)
        f2w = self.window_fusion(fw[1], u2w, fn[1], u2n, gw['assignment'], gamma_w, self.l2_kv)
        ln = self.decoder_nbi([fn[0], f2n, f3n], img_nbi.shape[-2:], deep_context=f4n)
        lw = self.decoder_wli([fw[0], f2w, f3w], img_wli.shape[-2:], deep_context=f4w)
        outputs = dict(Pi=plan, rho=rho, alpha=alpha, cost_sem=cost,
                       graph_nbi=gn, graph_wli=gw, audit_nbi=an, audit_wli=aw,
                       gamma_nbi=gamma_n, gamma_wli=gamma_w,
                       gamma_mean_nbi=gamma_n.mean(1), gamma_mean_wli=gamma_w.mean(1))
        for name, graph, own, message, audit, logits, fraction, weights in (
                ('nbi', gn, own_n, mn, an, ln, pn, wn), ('wli', gw, own_w, mw, aw, lw, pw, ww)):
            po = (own[..., 0]+self.p_les*own[..., 2]).clamp(1e-4, 1-1e-4)
            pc = (message[..., 0]+self.p_les*message[..., 2]).clamp(1e-4, 1-1e-4)
            if self.opinion_grad == 'detach':
                po, pc = po.detach(), pc.detach()
            llr = decision_llr(pc, audit['eta_plus'], self.p_les)
            graph.update(p_o=po, p_c=pc, message=message, fraction=fraction, weights=weights,
                         node_q_fused=(torch.logit(po)+llr).sigmoid())
            correction = F.interpolate(torch.einsum('bnhw,bn->bhw', graph['assignment'], llr)[:, None],
                                       size=logits.shape[-2:], mode='bilinear', align_corners=False)
            # A two-class decoder represents the binary logit as l_fg-l_bg.
            final = logits+torch.cat((torch.zeros_like(correction), correction), 1) if self.decision_fusion == 'add' else logits
            outputs.update({f'logits_final_{name}': final, f'logits_dec_{name}': logits,
                            f'logits_{name}': final, f'q_{name}': graph['Q'],
                            f'b_{name}': graph['B'], f'v_{name}': graph['V'],
                            f'q_logits_{name}': graph['q_logits'], f'b_logits_{name}': graph['b_logits'],
                            f'v_logits_{name}': graph['v_logits']})
        return outputs


def build_model(args: Any, *, pretrained: bool | None = None) -> PACE:
    """Construct the shared PACE model from a CLI namespace or checkpoint config.

    ``pretrained=False`` prevents downloading backbone weights when a complete
    checkpoint will immediately replace all model parameters for inference.
    """
    def value(name: str, default: Any) -> Any:
        return args.get(name, default) if isinstance(args, Mapping) else getattr(args, name, default)

    return PACE(
        encoder_name=value("encoder_name", "convnextv2_base"),
        encoder_pretrained=bool(value("encoder_pretrained", True)) if pretrained is None else bool(pretrained),
        d_model=int(value("d_model", 256)),
        token_dim=value("token_dim", None),
        num_nodes=int(value("num_tokens", 128)),
        hidden_channels=int(value("preliminary_hidden_channels", 128)),
        freeze_backbone=bool(value("freeze_backbone", False)),
        rho_min=float(value("rho_min", 0.2)),
        rho_max=float(value("rho_max", 0.85)),
        alpha_min=float(value("alpha_min", 0.05)),
        alpha_max=float(value("alpha_max", 0.35)),
        epsilon=float(value("transport_epsilon", 0.05)),
        outer_iterations=int(value("transport_fgw_iterations", 5)),
        inner_iterations=int(value("transport_sinkhorn_iterations", 50)),
        tau=float(value("audit_tau", 0.5)),
        node_spatial_strength=float(value("node_spatial_strength", 8.0)),
        theta=tuple(float(value("theta_" + name, 1.0)) for name in ("p", "q", "b", "d")),
        lambda_f=float(value("lambda_f", 1.0)),
        lambda_s=float(value("lambda_s", 0.5)),
        lambda_b=float(value("lambda_b", 0.5)),
        match_use_abs_pos=bool(value("match_use_abs_pos", False)),
        opinion_grad=value("opinion_grad", "detach"),
        l2_kv=value("l2_kv", "self"),
        decision_fusion=value("decision_fusion", "add"),
        force_t=value("force_t", None),
        num_heads=int(value("fusion_num_heads", 4)),
    )

"""Configuration objects for the J-Space Explorer.

Everything is plain dataclasses so the config can come from a JSON/env file,
the CLI, or the web layer without pulling in heavy deps.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field, asdict
from typing import Any, Optional

# Default model requested by the user: a local Qwen3.5-based safetensors model.
DEFAULT_MODEL_PATH = os.environ.get(
    "JSPACE_MODEL_PATH", "/mnt/Disk 3/models/GRaPE-2.1-Flash"
)


@dataclass
class ModelConfig:
    """How to load the model."""

    path: str = DEFAULT_MODEL_PATH
    # "auto" spreads the model across all visible GPUs (needed for a ~19GB model
    # on 24GB cards once activations/gradients are added).
    device_map: str = "auto"
    # bfloat16 matches the checkpoint's stored dtype and is required for stable
    # Jacobian computation on Ampere (3090) hardware.
    dtype: str = "bfloat16"
    # Some bleeding-edge architectures (qwen3_5) ship modeling code with the
    # checkpoint; allow it so we can load before the arch lands in a release.
    trust_remote_code: bool = True
    # Use Unsloth's FastLanguageModel for the *generation* path when available.
    # The lens path always uses the raw HF model (needs autograd + hidden states).
    prefer_unsloth: bool = False
    max_memory: Optional[dict[str, str]] = None
    # Attention kernel. "eager" is required for the Jacobian lens's
    # double-backward (flash/SDPA fused kernels lack a second derivative).
    attn_implementation: str = "eager"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class LensConfig:
    """Parameters controlling the lens computations."""

    # How many top tokens to report per (layer, position).
    top_k: int = 10

    # --- Jacobian lens fitting (aligned with anthropics/jacobian-lens) --------
    # The lens operator J_l = E[ d h_final / d h_l ] is *fit once* over a corpus
    # and cached (optionally saved to disk); applying it is then just a matmul
    # ``unembed(J_l @ h)`` -- instant. These control the fit.
    #
    # Number of corpus prompts to average the Jacobian over. Quality saturates
    # quickly; the paper uses ~1000, ~100 is usable, and a handful already gives
    # a sensible interactive lens.
    # Kept small by default so the one-time fit stays interactive (~2 min for a
    # few layers on a 9B model); raise it for higher-fidelity operators.
    jacobian_corpus_size: int = 3
    # Output dimensions (rows of J_l) computed per backward pass. The prompt is
    # replicated this many times along the batch axis; higher = more GPU memory,
    # fewer passes. Total FLOPs are unchanged.
    jacobian_dim_batch: int = 16
    # Leading positions excluded from the average (attention sinks have atypical
    # residual statistics). Matches the reference SKIP_FIRST_N_POSITIONS.
    jacobian_skip_first: int = 16
    # Truncate each corpus prompt to this many tokens when fitting.
    jacobian_max_seq_len: int = 128
    # Path to persist / load fitted lens operators (dict[layer]->tensor). When
    # set, a fitted lens is saved here and reused across runs. None = memory
    # only.
    jacobian_cache_path: Optional[str] = os.environ.get(
        "JSPACE_LENS_CACHE",
        os.path.join(os.path.expanduser("~"), ".cache", "jspace", "lens.pt"),
    )

    # --- J-space sparse decomposition ----------------------------------------
    # Sparsity k for the J-space sparse decomposition (union of k-dim faces).
    jspace_sparsity_k: int = 8
    # Candidate tokens considered when decomposing an activation (seeded from
    # the logit-lens top tokens). Bounds the dictionary size.
    decompose_candidates: int = 256
    # Max iterations for gradient-pursuit sparse decomposition.
    pursuit_iters: int = 32


@dataclass
class AppConfig:
    model: ModelConfig = field(default_factory=ModelConfig)
    lens: LensConfig = field(default_factory=LensConfig)
    host: str = "127.0.0.1"
    port: int = 8000

    @classmethod
    def from_env(cls) -> "AppConfig":
        cfg = cls()
        if v := os.environ.get("JSPACE_MODEL_PATH"):
            cfg.model.path = v
        if v := os.environ.get("JSPACE_HOST"):
            cfg.host = v
        if v := os.environ.get("JSPACE_PORT"):
            cfg.port = int(v)
        return cfg

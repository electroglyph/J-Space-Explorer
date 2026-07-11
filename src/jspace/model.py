"""Model loading and low-level access to the residual stream.

The lens methods need three things from a model that a plain generation API does
not expose:
  1. The per-layer residual-stream hidden states (``output_hidden_states``).
  2. The final norm + unembedding (``lm_head``) so we can map any residual
     vector to vocabulary logits.
  3. Autograd through the network, so we can take Jacobians of the final
     residual stream with respect to an intermediate layer.

Because of (3) we cannot use llama.cpp / GGUF (no autograd) or a pure inference
engine for the *lens* path; we load the raw safetensors weights with
HuggingFace ``transformers``. Unsloth is used only for the optional fast
generation path (see ``generate``), never for the lens math.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

import torch

from .config import ModelConfig

logger = logging.getLogger(__name__)

_DTYPES = {
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
    "float32": torch.float32,
}


def build_sampling_kwargs(do_sample: bool, temperature: float,
                          top_p=None, top_k=None, min_p=None) -> dict:
    """Assemble HuggingFace sampling kwargs, omitting any that are unset.

    When ``do_sample`` is False (greedy), no sampling params are passed (they
    would otherwise trigger warnings/errors). ``top_k=0`` disables top-k.
    """
    if not do_sample:
        return {}
    kw: dict = {"temperature": float(temperature)}
    if top_p is not None:
        kw["top_p"] = float(top_p)
    if top_k is not None:
        kw["top_k"] = int(top_k)
    if min_p is not None:
        kw["min_p"] = float(min_p)
    return kw


@dataclass
class ModelInfo:
    num_layers: int
    hidden_size: int
    vocab_size: int
    dtype: torch.dtype
    architecture: str


class LensModel:
    """Wraps a HuggingFace causal LM and exposes residual-stream internals.

    ``hidden_states`` from ``transformers`` has ``num_layers + 1`` entries:
    index 0 is the embedding output, index i (1..L) is the output of block i,
    i.e. the residual stream *after* layer i. We keep that convention and call
    index i "layer i" (with layer 0 = embeddings).
    """

    def __init__(self, model, tokenizer, info: ModelInfo, config: ModelConfig):
        self.model = model
        self.tokenizer = tokenizer
        self.info = info
        self.config = config
        self._unsloth_model = None  # lazily created for fast generation

    # ------------------------------------------------------------------ load
    @classmethod
    def load(cls, config: ModelConfig) -> "LensModel":
        from transformers import AutoModelForCausalLM, AutoTokenizer

        dtype = _DTYPES.get(config.dtype, torch.bfloat16)
        logger.info("Loading tokenizer from %s", config.path)
        tokenizer = AutoTokenizer.from_pretrained(
            config.path, trust_remote_code=config.trust_remote_code
        )

        logger.info("Loading model from %s (dtype=%s, device_map=%s)",
                    config.path, config.dtype, config.device_map)
        model_kwargs = dict(
            torch_dtype=dtype,
            device_map=config.device_map,
            trust_remote_code=config.trust_remote_code,
            output_hidden_states=True,
            low_cpu_mem_usage=True,
        )
        if getattr(config, "attn_implementation", None):
            model_kwargs["attn_implementation"] = config.attn_implementation
        # Derive per-GPU max_memory so accelerate keeps the whole model on the
        # GPUs. The lens path backprops through the network and reads the
        # unembedding, so any module offloaded to CPU/meta (accelerate's default
        # when it thinks memory is tight) breaks with "Cannot copy out of meta
        # tensor". We reserve headroom per card for activations + autograd.
        if config.max_memory:
            model_kwargs["max_memory"] = config.max_memory
        elif config.device_map == "auto" and torch.cuda.is_available():
            model_kwargs["max_memory"] = cls._auto_max_memory(config)

        try:
            model = AutoModelForCausalLM.from_pretrained(config.path, **model_kwargs)
        except (KeyError, ValueError) as exc:
            # A multimodal *ForConditionalGeneration* checkpoint may not resolve
            # under AutoModelForCausalLM on older transformers; try the generic
            # AutoModel and locate the text/causal submodule.
            logger.warning("AutoModelForCausalLM failed (%s); trying AutoModel", exc)
            from transformers import AutoModel
            model = AutoModel.from_pretrained(config.path, **model_kwargs)

        model.eval()
        # Gradients must flow for the Jacobian lens, but we never update weights.
        for p in model.parameters():
            p.requires_grad_(False)

        info = cls._introspect(model, tokenizer)
        logger.info("Loaded %s: %d layers, hidden=%d, vocab=%d",
                    info.architecture, info.num_layers, info.hidden_size,
                    info.vocab_size)
        obj = cls(model, tokenizer, info, config)
        obj._ensure_head_on_device()
        return obj

    @staticmethod
    def _auto_max_memory(config: ModelConfig) -> dict:
        """Offer nearly all of each visible GPU so nothing is offloaded to CPU.

        accelerate's "auto" heuristic can offload the head/final-norm to CPU
        (leaving them as meta tensors) when it thinks VRAM is tight, which
        breaks the autograd lens path. We instead advertise ~92%% of each GPU's
        total memory and deliberately omit a "cpu" key so device_map keeps the
        whole model on the GPUs.
        """
        max_memory: dict = {}
        for i in range(torch.cuda.device_count()):
            total = torch.cuda.get_device_properties(i).total_memory
            # Reserve headroom for activations + autograd graph.
            usable = int(total * 0.92)
            max_memory[i] = f"{usable // (1024 ** 2)}MiB"
        logger.info("Auto max_memory: %s", max_memory)
        return max_memory

    @staticmethod
    def _introspect(model, tokenizer) -> ModelInfo:
        cfg = model.config
        text_cfg = getattr(cfg, "text_config", cfg)
        num_layers = getattr(text_cfg, "num_hidden_layers", None) or getattr(
            cfg, "num_hidden_layers", 0
        )
        hidden = getattr(text_cfg, "hidden_size", None) or getattr(
            cfg, "hidden_size", 0
        )
        vocab = getattr(text_cfg, "vocab_size", None) or getattr(
            cfg, "vocab_size", len(tokenizer)
        )
        arch = (cfg.architectures or ["unknown"])[0] if getattr(
            cfg, "architectures", None
        ) else type(model).__name__
        dtype = next(model.parameters()).dtype
        return ModelInfo(
            num_layers=int(num_layers),
            hidden_size=int(hidden),
            vocab_size=int(vocab),
            dtype=dtype,
            architecture=arch,
        )

    # ------------------------------------------------------------- internals
    @property
    def device(self) -> torch.device:
        return next(self.model.parameters()).device

    def _ensure_head_on_device(self) -> None:
        """Make sure the unembedding + final norm are on a real GPU.

        If accelerate offloaded them (params on ``cpu``/``meta`` behind an
        AlignDevicesHook), forward passes fail with "Cannot copy out of meta
        tensor". We detach any accelerate hook and move these small modules onto
        the first CUDA device so the lens can always run.
        """
        if not torch.cuda.is_available():
            return
        target = torch.device("cuda:0")
        try:
            from accelerate.hooks import remove_hook_from_module
        except Exception:
            remove_hook_from_module = None
        for mod in (self._find_lm_head(), self._find_final_norm()):
            if mod is None:
                continue
            params = list(mod.parameters())
            if not params:
                continue
            needs_fix = any(p.is_meta or p.device.type != "cuda" for p in params)
            if not needs_fix:
                continue
            if remove_hook_from_module is not None:
                try:
                    remove_hook_from_module(mod, recurse=True)
                except Exception:
                    pass
            try:
                mod.to(target)
                logger.info("Moved %s onto %s (was offloaded)",
                            type(mod).__name__, target)
            except NotImplementedError:
                logger.warning("Could not move %s off meta device",
                               type(mod).__name__)

    def _find_lm_head(self) -> torch.nn.Module:
        """Locate the unembedding (vocab projection) module."""
        if hasattr(self.model, "lm_head") and self.model.lm_head is not None:
            return self.model.lm_head
        # Some wrappers nest the causal LM.
        for attr in ("language_model", "model", "transformer"):
            sub = getattr(self.model, attr, None)
            if sub is not None and hasattr(sub, "lm_head"):
                return sub.lm_head
        # Tied embeddings fallback: reuse input embedding matrix.
        emb = self.model.get_output_embeddings()
        if emb is not None:
            return emb
        raise RuntimeError("Could not locate an lm_head / output embedding.")

    def _find_final_norm(self) -> Optional[torch.nn.Module]:
        """Locate the final RMSNorm/LayerNorm applied before the unembedding.

        The logit lens is much more faithful when the same final norm the model
        uses is applied to intermediate activations before unembedding.
        """
        candidates = []
        base = self.model
        for attr in ("model", "language_model", "transformer"):
            sub = getattr(base, attr, None)
            if sub is not None:
                candidates.append(sub)
                inner = getattr(sub, "model", None)
                if inner is not None:
                    candidates.append(inner)
        candidates.append(base)
        for c in candidates:
            for name in ("norm", "final_layernorm", "ln_f", "final_norm"):
                norm = getattr(c, name, None)
                if norm is not None:
                    return norm
        return None

    def unembed(self, hidden: torch.Tensor, apply_final_norm: bool = True) -> torch.Tensor:
        """Map residual-stream vectors to vocabulary logits.

        ``hidden`` has shape ``(..., hidden_size)``. Returns ``(..., vocab)``.
        """
        lm_head = self._find_lm_head()
        head_device = next(lm_head.parameters()).device
        head_dtype = next(lm_head.parameters()).dtype
        h = hidden.to(head_device, head_dtype)
        if apply_final_norm:
            norm = self._find_final_norm()
            if norm is not None:
                norm_param = next(norm.parameters())
                h = norm(h.to(norm_param.device, norm_param.dtype))
                h = h.to(head_device, head_dtype)
        return lm_head(h)

    # --------------------------------------------------------------- tokenize
    def build_input_ids(self, prompt: str, use_chat_template: bool = True) -> torch.Tensor:
        """Turn a prompt string into input_ids on the model device."""
        tok = self.tokenizer
        if use_chat_template and getattr(tok, "chat_template", None):
            messages = [{"role": "user", "content": prompt}]
            ids = tok.apply_chat_template(
                messages, add_generation_prompt=True, return_tensors="pt"
            )
        else:
            ids = tok(prompt, return_tensors="pt").input_ids
        # transformers 5.x may return a BatchEncoding/dict from either path;
        # normalize to the input_ids tensor.
        if hasattr(ids, "input_ids"):
            ids = ids.input_ids
        elif isinstance(ids, dict):
            ids = ids["input_ids"]
        return ids.to(self.device)

    def token_strings(self, input_ids: torch.Tensor) -> list[str]:
        ids = input_ids.flatten().tolist()
        return [self.tokenizer.decode([i]) for i in ids]

    def decode_token(self, token_id: int) -> str:
        return self.tokenizer.decode([token_id])

    # ----------------------------------------------------------------- chat
    def build_chat_ids(self, messages: list[dict],
                       add_generation_prompt: bool = True) -> torch.Tensor:
        """Tokenize a conversation (list of {role, content}) via the chat
        template. ``add_generation_prompt`` appends the assistant-turn header so
        the model (or the lens) sees the point where the reply begins.
        """
        tok = self.tokenizer
        if not getattr(tok, "chat_template", None):
            text = "".join(f"{m['role']}: {m['content']}\n" for m in messages)
            if add_generation_prompt:
                text += "assistant:"
            ids = tok(text, return_tensors="pt").input_ids
        else:
            ids = tok.apply_chat_template(
                messages, add_generation_prompt=add_generation_prompt,
                return_tensors="pt",
            )
        if hasattr(ids, "input_ids"):
            ids = ids.input_ids
        elif isinstance(ids, dict):
            ids = ids["input_ids"]
        return ids.to(self.device)

    def chat(self, messages: list[dict], max_new_tokens: int = 256,
             **kwargs) -> str:
        """Generate the assistant's reply to a conversation."""
        input_ids = self.build_chat_ids(messages, add_generation_prompt=True)
        temperature = kwargs.pop("temperature", 0.7)
        do_sample = kwargs.pop("do_sample", temperature > 0)
        sample_kwargs = build_sampling_kwargs(
            do_sample, temperature,
            top_p=kwargs.pop("top_p", None),
            top_k=kwargs.pop("top_k", None),
            min_p=kwargs.pop("min_p", None),
        )
        eos = self.eos_ids()
        with torch.no_grad():
            out = self.model.generate(
                input_ids=input_ids,
                max_new_tokens=max_new_tokens,
                do_sample=do_sample,
                **sample_kwargs,
                pad_token_id=(self.tokenizer.pad_token_id
                              if self.tokenizer.pad_token_id is not None
                              else self.tokenizer.eos_token_id),
                eos_token_id=(sorted(eos) if eos else None),
                **kwargs,
            )
        new_tokens = out[0][input_ids.shape[1]:]
        return self.tokenizer.decode(new_tokens, skip_special_tokens=True)

    def eos_ids(self) -> set:
        """Every token id that should end generation for this model.

        Combines the tokenizer/config EOS ids with common chat end markers
        (some checkpoints set only one of these), so generation actually stops.
        """
        ids = set()
        tok = self.tokenizer
        cfg = self.model.config
        for src in (getattr(tok, "eos_token_id", None),
                    getattr(cfg, "eos_token_id", None),
                    getattr(getattr(cfg, "text_config", cfg), "eos_token_id", None)):
            if src is None:
                continue
            if isinstance(src, (list, tuple, set)):
                ids |= {int(x) for x in src}
            else:
                ids.add(int(src))
        for marker in ("<|im_end|>", "<|endoftext|>", "<|eot_id|>", "<|end|>"):
            try:
                tid = tok.convert_tokens_to_ids(marker)
                if isinstance(tid, int) and tid >= 0 and tid != getattr(
                    tok, "unk_token_id", -1):
                    ids.add(int(tid))
            except Exception:
                pass
        return ids

    # --------------------------------------------------------------- forward
    def forward_hidden_states(self, input_ids: torch.Tensor):
        """Run a forward pass returning all residual-stream hidden states.

        Returns a tuple of length ``num_layers + 1`` of tensors shaped
        ``(batch, seq, hidden)``. No grad; use for the logit lens.
        """
        with torch.no_grad():
            out = self.model(
                input_ids=input_ids,
                output_hidden_states=True,
                use_cache=False,
            )
        return out.hidden_states

    # -------------------------------------------------------------- generate
    def generate(self, prompt: str, max_new_tokens: int = 128,
                 use_chat_template: bool = True, **kwargs) -> str:
        """Fast generation. Uses the already-loaded HF model.

        (Unsloth acceleration can be enabled via config.prefer_unsloth, but the
        default keeps a single model in memory to avoid doubling VRAM use.)
        """
        input_ids = self.build_input_ids(prompt, use_chat_template)
        with torch.no_grad():
            out = self.model.generate(
                input_ids=input_ids,
                max_new_tokens=max_new_tokens,
                **kwargs,
            )
        new_tokens = out[0][input_ids.shape[1]:]
        return self.tokenizer.decode(new_tokens, skip_special_tokens=True)

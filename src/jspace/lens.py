"""Lens engines: logit lens and the Jacobian lens (J-lens).

Background (Transformer Circuits, "Verbalizable Representations Form a Global
Workspace"; reference implementation: anthropics/jacobian-lens):

* **Logit lens** takes an intermediate residual-stream activation ``h_l`` and
  applies the model's final norm + unembedding directly, reading off the top
  vocabulary tokens. It assumes representations share coordinates across layers,
  which breaks down in early layers.

* **Jacobian lens (J-lens)** transports ``h_l`` into the final-layer basis with
  the average input-output Jacobian, then decodes with the unembedding::

      lens_l(h) = unembed( J_l @ h ),   J_l = E[ d h_final / d h_l ]

  The expectation is over prompts and source positions, with cotangents summed
  over all current-and-future target positions (the reference estimator). Early
  positions (attention sinks) and the final position are excluded.

The operator ``J_l`` is **fit once** over a corpus and cached (optionally saved
to disk). Applying it to any activation is then just a matrix multiply, so the
interactive lens grid and the J-space decomposition are fast: the only heavy
work -- the model's own backward pass -- happens once per (corpus, layer set).
"""
from __future__ import annotations

import logging
import math
import os
import threading
from dataclasses import dataclass, field
from typing import Callable, Optional, Sequence

import torch

from .config import LensConfig
from .model import LensModel

logger = logging.getLogger(__name__)


@dataclass
class TokenScore:
    token_id: int
    token: str
    score: float


@dataclass
class LensCell:
    """Lens readout for one (layer, position)."""
    layer: int
    position: int
    top: list[TokenScore]


@dataclass
class LensResult:
    method: str  # "logit" or "jacobian"
    prompt: str
    tokens: list[str]           # the input tokens (columns)
    layers: list[int]           # layer indices (rows)
    cells: list[LensCell]       # flattened grid of readouts
    meta: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "method": self.method,
            "prompt": self.prompt,
            "tokens": self.tokens,
            "layers": self.layers,
            "cells": [
                {
                    "layer": c.layer,
                    "position": c.position,
                    "top": [
                        {"token_id": t.token_id, "token": t.token, "score": t.score}
                        for t in c.top
                    ],
                }
                for c in self.cells
            ],
            "meta": self.meta,
        }


def _softmax_top_k(logits: torch.Tensor, k: int) -> tuple[list[int], list[float]]:
    probs = torch.softmax(logits.float(), dim=-1)
    vals, idx = torch.topk(probs, k)
    return idx.tolist(), vals.tolist()


class LensEngine:
    def __init__(self, model: LensModel, config: Optional[LensConfig] = None):
        self.model = model
        self.config = config or LensConfig()
        # Fitted Jacobian operators J_l, keyed by source layer. fp32 CPU
        # tensors (hidden x hidden); moved to the activation's device on apply.
        self._jacobian: dict[int, torch.Tensor] = {}
        # Signature of the corpus the current operators were fit on.
        self._fit_signature: Optional[tuple] = None
        # Guards all mutation/reads of _jacobian / _fit_signature across the
        # FastAPI threadpool and the /api/fit worker thread.
        self._fit_lock = threading.RLock()
        if self.config.jacobian_cache_path:
            self._load_operators(self.config.jacobian_cache_path)

    # ------------------------------------------------------------ logit lens
    @torch.no_grad()
    def logit_lens(
        self,
        prompt: str,
        layers: Optional[Sequence[int]] = None,
        positions: Optional[Sequence[int]] = None,
        use_chat_template: bool = True,
        top_k: Optional[int] = None,
        input_ids: Optional[torch.Tensor] = None,
    ) -> LensResult:
        if input_ids is None:
            input_ids = self.model.build_input_ids(prompt, use_chat_template)
        hidden_states = self.model.forward_hidden_states(input_ids)
        return self._read_lens_from_hidden(
            prompt, input_ids, hidden_states, layers, positions,
            method="logit", transform=None, top_k=top_k,
        )

    # --------------------------------------------------------- jacobian lens
    def jacobian_lens(
        self,
        prompt: str,
        layers: Optional[Sequence[int]] = None,
        positions: Optional[Sequence[int]] = None,
        corpus: Optional[Sequence[str]] = None,
        use_chat_template: bool = True,
        top_k: Optional[int] = None,
        input_ids: Optional[torch.Tensor] = None,
    ) -> LensResult:
        """Read the corpus-averaged Jacobian lens for a prompt.

        Ensures the operators ``J_l`` for the requested layers are fit (once)
        over ``corpus``, then applies ``unembed(J_l @ h)`` -- a matmul per cell.
        """
        if input_ids is None:
            input_ids = self.model.build_input_ids(prompt, use_chat_template)
        hidden_states = self.model.forward_hidden_states(input_ids)
        num_layers = len(hidden_states) - 1
        layer_list = list(layers) if layers is not None else list(range(num_layers + 1))
        layer_list = [l for l in layer_list if 0 <= l <= num_layers]

        corpus = list(corpus) if corpus else default_corpus()
        self.ensure_fitted(layer_list, corpus, use_chat_template)

        def transform(h: torch.Tensor, layer: int) -> torch.Tensor:
            J = self._jacobian.get(layer)
            if J is None:
                return h
            return J.to(h.device, torch.float32) @ h.float()

        return self._read_lens_from_hidden(
            prompt, input_ids, hidden_states, layer_list, positions,
            method="jacobian", transform=transform,
            meta={
                "corpus_size": min(len(corpus), self.config.jacobian_corpus_size),
                "fitted_layers": sorted(self._jacobian.keys()),
            },
            top_k=top_k,
        )

    # ------------------------------------------------- streaming generation
    @torch.no_grad()
    def stream_generate_with_lens(
        self,
        input_ids: torch.Tensor,
        layers: Optional[Sequence[int]] = None,
        method: str = "logit",
        max_new_tokens: int = 200,
        top_k: int = 6,
        do_sample: bool = False,
        temperature: float = 0.7,
        top_p: Optional[float] = None,
        sampling_top_k: Optional[int] = None,
        min_p: Optional[float] = None,
        stop_token_ids: Optional[Sequence[int]] = None,
    ):
        """Generate token-by-token, yielding each new token AND its J-space.

        This is a manual decode loop (KV-cache) so we can, at every step, read
        the residual stream that produced the just-emitted token and decode it
        through the requested lens. Yields dicts:

          {"type":"token", "index":i, "token":str, "token_id":int,
           "layers":[...], "cells":[{"layer":l, "top":[{token,score}...]}]}

        Generation stops at ``max_new_tokens`` or when an EOS/stop token is
        produced (see ``stop_token_ids``).
        """
        model = self.model.model
        device = self.model.device
        num_layers = self.model.info.num_layers
        layer_list = list(layers) if layers is not None else list(range(num_layers + 1))
        layer_list = [l for l in layer_list if 0 <= l <= num_layers]
        k = max(1, min(int(top_k), self.model.info.vocab_size))

        # Resolve stop tokens: caller-provided plus the model/tokenizer EOS set.
        stops = set(int(t) for t in (stop_token_ids or []))
        stops |= self._eos_token_ids()

        # Jacobian transform (fitted operator) if requested.
        transform = None
        if method == "jacobian":
            self.ensure_fitted(layer_list, default_corpus())

            def transform(h, layer):
                J = self._jacobian.get(layer)
                if J is None:
                    return h
                return J.to(h.device, torch.float32) @ h.float()

        cur = input_ids
        past = None
        emitted = 0
        while emitted < max_new_tokens:
            out = model(
                input_ids=cur if past is None else cur[:, -1:],
                past_key_values=past,
                use_cache=True,
                output_hidden_states=True,
            )
            past = out.past_key_values
            hidden_states = out.hidden_states  # tuple[(1, seq_or_1, hidden)]
            # Next-token logits at the last position.
            last_logits = out.logits[0, -1, :].float()
            next_id = self._pick_next(
                last_logits, do_sample, temperature, top_p, sampling_top_k, min_p
            )

            # Read the lens at the LAST position of each layer (the residual
            # that produced this token).
            cells = []
            for ell in layer_list:
                h = hidden_states[ell][0, -1]  # (hidden,)
                apply_norm = True
                if transform is not None:
                    h = transform(h, ell)
                elif ell == num_layers:
                    apply_norm = False  # already post-norm
                logits = self.model.unembed(
                    h.unsqueeze(0), apply_final_norm=apply_norm
                ).squeeze(0)
                ids, scores = _softmax_top_k(logits, k)
                cells.append({
                    "layer": ell,
                    "top": [
                        {"token_id": int(i), "token": self.model.decode_token(int(i)),
                         "score": float(s)}
                        for i, s in zip(ids, scores)
                    ],
                })

            tok_str = self.model.decode_token(int(next_id))
            yield {
                "type": "token", "index": emitted,
                "token": tok_str, "token_id": int(next_id),
                "layers": layer_list, "cells": cells,
            }
            emitted += 1
            if int(next_id) in stops:
                break
            cur = torch.tensor([[int(next_id)]], device=device)

    def _eos_token_ids(self) -> set:
        """All token ids that should end generation (delegates to the model)."""
        return self.model.eos_ids()

    def _pick_next(self, logits, do_sample, temperature, top_p, top_k, min_p):
        if not do_sample or temperature <= 0:
            return int(torch.argmax(logits).item())
        logits = logits / max(temperature, 1e-5)
        probs = torch.softmax(logits, dim=-1)
        if top_k:
            kth = torch.topk(probs, min(int(top_k), probs.numel())).values.min()
            probs = torch.where(probs < kth, torch.zeros_like(probs), probs)
        if min_p:
            probs = torch.where(probs < float(min_p) * probs.max(),
                                torch.zeros_like(probs), probs)
        if top_p and 0 < top_p < 1:
            sorted_p, sorted_i = torch.sort(probs, descending=True)
            cdf = torch.cumsum(sorted_p, dim=-1)
            cutoff = cdf > top_p
            cutoff[0] = False  # always keep the top token
            sorted_p[cutoff] = 0.0
            probs = torch.zeros_like(probs).scatter(0, sorted_i, sorted_p)
        s = probs.sum()
        if s <= 0:
            return int(torch.argmax(logits).item())
        probs = probs / s
        return int(torch.multinomial(probs, 1).item())

    # ---------------------------------------------------- shared lens reader
    @torch.no_grad()
    def _read_lens_from_hidden(
        self,
        prompt: str,
        input_ids: torch.Tensor,
        hidden_states,
        layers: Optional[Sequence[int]],
        positions: Optional[Sequence[int]],
        method: str,
        transform: Optional[Callable[[torch.Tensor, int], torch.Tensor]],
        meta: Optional[dict] = None,
        top_k: Optional[int] = None,
    ) -> LensResult:
        num_layers = len(hidden_states) - 1
        seq_len = input_ids.shape[1]
        layer_list = list(layers) if layers is not None else list(range(num_layers + 1))
        layer_list = [l for l in layer_list if 0 <= l <= num_layers]
        pos_list = list(positions) if positions is not None else list(range(seq_len))
        pos_list = [p for p in pos_list if 0 <= p < seq_len]

        tokens = self.model.token_strings(input_ids)
        # Per-call top_k avoids mutating the shared config across threads.
        k = top_k if top_k is not None else self.config.top_k
        k = max(1, min(int(k), self.model.info.vocab_size))
        cells: list[LensCell] = []

        for ell in layer_list:
            layer_hidden = hidden_states[ell][0]  # (seq, hidden)
            # transformers 5.x returns hidden_states[-1] already post-final-norm.
            # For the plain logit lens that means the final layer must NOT be
            # normed again. (The Jacobian transform targets the pre-norm
            # residual, so it always needs the final norm applied once.)
            apply_norm = not (transform is None and ell == num_layers)
            for pos in pos_list:
                h = layer_hidden[pos]
                if transform is not None:
                    h = transform(h, ell)
                logits = self.model.unembed(
                    h.unsqueeze(0), apply_final_norm=apply_norm
                ).squeeze(0)
                ids, scores = _softmax_top_k(logits, k)
                top = [
                    TokenScore(i, self.model.decode_token(i), float(s))
                    for i, s in zip(ids, scores)
                ]
                cells.append(LensCell(layer=ell, position=pos, top=top))

        result_meta = {
            "num_layers": num_layers,
            "seq_len": seq_len,
            "top_k": k,
        }
        if meta:
            result_meta.update(meta)
        return LensResult(
            method=method, prompt=prompt, tokens=[tokens[p] for p in pos_list],
            layers=layer_list, cells=cells, meta=result_meta,
        )

    # ============================================================ fit / cache
    def ensure_fitted(
        self, layers: Sequence[int], corpus: Sequence[str],
        use_chat_template: bool = True,
        progress: Optional[Callable[[int, int], None]] = None,
    ) -> None:
        """Fit any missing operators for ``layers`` over ``corpus`` (idempotent).

        If the corpus changed since the last fit, all cached operators are
        discarded first. Operators already fitted for the same corpus are
        reused, so re-requesting a layer is free. Thread-safe: the whole
        check/clear/fit/save runs under the engine's fit lock.

        Note: the averaging corpus is generic *raw text*, so the fitted operator
        does NOT depend on the analysis prompt's chat-template setting. The
        ``use_chat_template`` argument is therefore ignored for fitting and
        excluded from the cache signature -- a lens fit once is reused whether
        prompts are later run with or without the chat template.
        """
        with self._fit_lock:
            # A None/empty corpus means "use the standard corpus" -- and, so the
            # signature matches an operator already fit/loaded over that corpus,
            # we resolve it the same way fit() does.
            corpus = list(corpus) if corpus else default_corpus()
            signature = (
                tuple(corpus[: self.config.jacobian_corpus_size]),
                self.model.info.num_layers,
            )
            if self._fit_signature != signature:
                self._jacobian.clear()
                self._fit_signature = signature
            missing = [l for l in layers if l not in self._jacobian]
            if not missing:
                return
            self.fit(missing, corpus, progress=progress)
            if self.config.jacobian_cache_path:
                self._save_operators(self.config.jacobian_cache_path)

    def fit(
        self, layers: Sequence[int], corpus: Sequence[str],
        progress: Optional[Callable[[int, int], None]] = None,
    ) -> None:
        """Fit the Jacobian operators ``J_l = E[d h_final / d h_l]``.

        Averages ``jacobian_for_prompt`` over the corpus (weighted by the number
        of valid source positions per prompt), matching the reference estimator.
        The corpus is fit as raw text (no chat template).
        """
        layers = sorted({int(l) for l in layers})
        hidden = self.model.info.hidden_size
        accum = {l: torch.zeros(hidden, hidden, dtype=torch.float32) for l in layers}
        weight = 0
        n_ctx = min(self.config.jacobian_corpus_size, len(corpus))
        for i, text in enumerate(corpus[:n_ctx]):
            try:
                jac, n_valid = self._jacobian_for_prompt(
                    text, layers, use_chat_template=False
                )
            except ValueError as exc:
                logger.debug("skipping corpus item %d: %s", i, exc)
                continue
            if n_valid <= 0:
                continue
            for l in layers:
                accum[l] += jac[l] * n_valid
            weight += n_valid
            if progress:
                progress(i + 1, n_ctx)
        if weight == 0:
            logger.warning("Jacobian fit got no valid positions; using identity")
            with self._fit_lock:
                for l in layers:
                    self._jacobian[l] = torch.eye(hidden, dtype=torch.float32)
            return
        with self._fit_lock:
            for l in layers:
                self._jacobian[l] = accum[l] / weight
        logger.info("Fitted Jacobian operators for layers %s over %d contexts",
                    layers, n_ctx)

    def _jacobian_for_prompt(
        self, prompt: str, source_layers: Sequence[int], use_chat_template: bool,
    ) -> tuple[dict[int, torch.Tensor], int]:
        """Per-prompt Jacobian estimator for several source layers at once.

        One forward pass on the prompt replicated ``dim_batch`` times, then
        ``ceil(hidden / dim_batch)`` backward passes reusing the retained graph.
        Batch element ``b`` carries a one-hot cotangent at output dimension
        ``dim_start + b`` set at **every valid target position**, so the
        gradient at source position ``p`` sums current-and-future target
        effects; we then mean over valid source positions.

        Returns ``({layer: (hidden, hidden) fp32 CPU}, n_valid_source_positions)``.
        """
        hidden = self.model.info.hidden_size
        dim_batch = max(1, int(self.config.jacobian_dim_batch))
        input_ids = self.model.build_input_ids(prompt, use_chat_template)
        # Truncate for fitting.
        if input_ids.shape[1] > self.config.jacobian_max_seq_len:
            input_ids = input_ids[:, : self.config.jacobian_max_seq_len]
        seq_len = input_ids.shape[1]
        mask = _valid_position_mask(seq_len, self.config.jacobian_skip_first)
        n_valid = int(mask.sum())

        captured: dict = {}
        handles = []
        for l in source_layers:
            handles.append(self._register_source_hook(l, captured))
        target_handle = self._register_target_hook(captured)

        jac = {l: torch.zeros(hidden, hidden, dtype=torch.float32) for l in source_layers}
        try:
            with torch.enable_grad():
                replicated = input_ids.expand(dim_batch, -1)
                out = self.model.model(
                    input_ids=replicated, output_hidden_states=True, use_cache=False
                )
                target = self._resolve_target_batched(captured, out)  # (B, seq, H)
                sources = {l: captured[f"src_{l}"] for l in source_layers
                           if f"src_{l}" in captured}
                if not sources:
                    return jac, 0
                device = target.device
                valid_positions = mask.nonzero(as_tuple=True)[0].to(device)
                b_idx = torch.arange(dim_batch, device=device)
                cotangent = torch.zeros_like(target)
                n_passes = math.ceil(hidden / dim_batch)
                src_list = list(sources.values())
                src_layers = list(sources.keys())

                for pass_idx, dim_start in enumerate(range(0, hidden, dim_batch)):
                    n_dims = min(dim_batch, hidden - dim_start)
                    cotangent.zero_()
                    # batch element b -> output dim (dim_start + b) at all valid
                    # target positions.
                    cotangent[
                        b_idx[:n_dims, None],
                        valid_positions[None, :],
                        dim_start + b_idx[:n_dims, None],
                    ] = 1.0
                    grads = torch.autograd.grad(
                        outputs=target,
                        inputs=src_list,
                        grad_outputs=cotangent,
                        retain_graph=(pass_idx < n_passes - 1),
                        allow_unused=True,
                    )
                    for l, g in zip(src_layers, grads):
                        if g is None:
                            continue
                        # g: (B, seq, H). Row (dim_start+b) of J = sum over valid
                        # source positions of g[b] (mean applied later via /n).
                        rows = g[:n_dims][:, mask, :].sum(dim=1).float().cpu()
                        jac[l][dim_start:dim_start + n_dims] = rows
            for l in source_layers:
                jac[l] /= max(1, n_valid)
            return jac, n_valid
        finally:
            for h in handles:
                h.remove()
            if target_handle is not None:
                target_handle.remove()

    # ------------------------------------------------------------- J-space
    def jacobian_transpose_batched(
        self, prompt_or_ids, layer: int, cotangents: torch.Tensor,
        use_chat_template: bool = True,
    ) -> Optional[torch.Tensor]:
        """Compute ``J_l^T @ C`` for a batch of cotangents in one backward pass.

        Given ``cotangents`` of shape ``(n, hidden)`` returns ``(n, hidden)``
        where row i is ``J_l^T @ cotangents[i]``. Seeding with unembedding rows
        ``w_v`` yields the J-lens vectors for those tokens. Uses the *fitted*
        operator when available (a matmul); otherwise falls back to a single
        double-backward on ``prompt_or_ids``.
        """
        J = self._jacobian.get(layer)
        if J is not None:
            # J: (H_out, H_in);  row i of result = J^T @ C[i]  == (C @ J)[i].
            dev = cotangents.device
            C = cotangents.float().to(dev)
            return C @ J.to(dev).float()
        # Fallback: no fitted operator -> compute via one VJP.
        if isinstance(prompt_or_ids, str):
            input_ids = self.model.build_input_ids(prompt_or_ids, use_chat_template)
        elif prompt_or_ids is None:
            # Cannot run a VJP without a context; caller should have fitted the
            # operator first (jlens_direction ensures this).
            logger.warning("jacobian_transpose_batched: layer %d not fitted and "
                           "no context given; returning None", layer)
            return None
        else:
            input_ids = prompt_or_ids
        return self._jacobian_transpose_vjp(input_ids, layer, cotangents)

    # -------------------------------------------------- J-space intervention
    def jlens_direction(self, layer: int, token_id: int) -> Optional[torch.Tensor]:
        """Unit J-lens vector for ``token_id`` at ``layer``: normalize J_l^T w_v.

        This is the residual-stream direction that most raises token ``v``'s
        (present + future) logit through the fitted Jacobian -- the axis we
        steer along or swap in the J-space. The operator for ``layer`` is fit
        on demand (over the default corpus) if it is not already cached.
        """
        if layer not in self._jacobian:
            self.ensure_fitted([layer], default_corpus())
        lm_head = self.model._find_lm_head()
        w_v = lm_head.weight.detach()[token_id].unsqueeze(0)  # (1, hidden)
        d = self.jacobian_transpose_batched(None, layer, w_v)  # (1, hidden)
        if d is None:
            return None
        d = d.squeeze(0).float()
        n = d.norm().clamp_min(1e-8)
        return d / n

    def generate_with_interventions(
        self, input_ids: torch.Tensor, interventions: Sequence[dict],
        max_new_tokens: int = 256, **gen_kwargs,
    ) -> str:
        """Generate while editing the J-space at one or more layers.

        Each intervention is a dict:
          {"layer": int, "token_id": int, "mode": "steer"|"ablate"|"swap",
           "strength": float, "swap_to": Optional[int]}

        * steer : add ``strength * ||h|| * d_hat`` (inject the concept).
        * ablate: remove the component of ``h`` along ``d_hat`` (suppress it).
        * swap  : remove the source direction's projection and add the target
                  direction scaled to that same projection (exchange concepts).

        Interventions are applied to EVERY position at their layer during
        generation, via forward hooks on the decoder blocks.
        """
        # Group interventions by layer and precompute directions.
        by_layer: dict[int, list[dict]] = {}
        for iv in interventions:
            layer = int(iv["layer"])
            d = self.jlens_direction(layer, int(iv["token_id"]))
            if d is None:
                continue
            entry = dict(iv)
            entry["dir"] = d
            if iv.get("mode") == "swap" and iv.get("swap_to") is not None:
                d2 = self.jlens_direction(layer, int(iv["swap_to"]))
                entry["dir_to"] = d2
            by_layer.setdefault(layer, []).append(entry)

        handles = []

        def make_hook(edits):
            def hook(module, inp, out):
                hs = out[0] if isinstance(out, tuple) else out
                h = hs.float()
                for e in edits:
                    d = e["dir"].to(h.device)
                    strength = float(e.get("strength", 0.3))
                    mode = e.get("mode", "steer")
                    # Per-position residual norm for scale-aware steering.
                    hnorm = h.norm(dim=-1, keepdim=True)
                    proj = (h * d).sum(dim=-1, keepdim=True)  # (.,.,1)
                    if mode == "ablate":
                        h = h - proj * d
                    elif mode == "swap" and e.get("dir_to") is not None:
                        d2 = e["dir_to"].to(h.device)
                        h = h - proj * d + proj * d2
                    else:  # steer
                        h = h + strength * hnorm * d
                hs2 = h.to(hs.dtype)
                if isinstance(out, tuple):
                    return (hs2,) + tuple(out[1:])
                return hs2
            return hook

        try:
            for layer, edits in by_layer.items():
                mod = self._decoder_layer_module(layer)
                if mod is not None:
                    handles.append(mod.register_forward_hook(make_hook(edits)))
            with torch.no_grad():
                out = self.model.model.generate(
                    input_ids=input_ids,
                    max_new_tokens=max_new_tokens,
                    pad_token_id=(self.model.tokenizer.pad_token_id
                                  if self.model.tokenizer.pad_token_id is not None
                                  else self.model.tokenizer.eos_token_id),
                    eos_token_id=(sorted(self.model.eos_ids()) or None),
                    **gen_kwargs,
                )
            new_tokens = out[0][input_ids.shape[1]:]
            return self.model.tokenizer.decode(new_tokens, skip_special_tokens=True)
        finally:
            for h in handles:
                h.remove()

    def _jacobian_transpose_vjp(
        self, input_ids: torch.Tensor, layer: int, cotangents: torch.Tensor,
    ) -> Optional[torch.Tensor]:
        captured: dict = {}
        handle = self._register_source_hook(layer, captured)
        if handle is None and layer != 0:
            return None
        target_handle = self._register_target_hook(captured)
        try:
            with torch.enable_grad():
                out = self.model.model(
                    input_ids=input_ids, output_hidden_states=True, use_cache=False
                )
                h_src = captured.get(f"src_{layer}")
                if h_src is None:
                    return None
                target = self._resolve_target(captured, out)[-1]  # (hidden,)
                C = cotangents.to(target.device, target.dtype)  # (n, hidden)
                grads = torch.autograd.grad(
                    outputs=target, inputs=h_src, grad_outputs=C,
                    is_grads_batched=True, retain_graph=False, allow_unused=True,
                )[0]
                if grads is None:
                    return None
                return grads[:, 0, -1, :].detach().float()
        finally:
            if handle is not None:
                handle.remove()
            if target_handle is not None:
                target_handle.remove()

    # --------------------------------------------------------------- hooks
    def _register_source_hook(self, layer: int, captured: dict):
        """Capture the layer-``layer`` residual stream as a grad-requiring leaf."""
        key = f"src_{layer}"
        if layer == 0:
            emb = self.model.model.get_input_embeddings()

            def emb_hook(module, inp, out):
                out = out.clone().requires_grad_(True)
                captured[key] = out
                return out
            return emb.register_forward_hook(emb_hook)
        mod = self._decoder_layer_module(layer)
        if mod is None:
            return None

        def layer_hook(module, inp, out):
            hs = out[0] if isinstance(out, tuple) else out
            hs = hs.clone().requires_grad_(True)
            captured[key] = hs
            if isinstance(out, tuple):
                return (hs,) + tuple(out[1:])
            return hs
        return mod.register_forward_hook(layer_hook)

    def _register_target_hook(self, captured: dict):
        """Hook the last decoder block to capture its PRE-final-norm output.

        transformers 5.x overwrites ``hidden_states[-1]`` with the post-norm
        ``last_hidden_state``; to apply the final norm exactly once in the lens
        we need the residual stream *before* the final norm as the target.
        """
        last_block = self._decoder_layer_module(self.model.info.num_layers)
        if last_block is None:
            return None

        def thook(module, inp, out):
            hs = out[0] if isinstance(out, tuple) else out
            captured["target_full"] = hs  # (B, seq, hidden)
        return last_block.register_forward_hook(thook)

    def _resolve_target(self, captured: dict, out) -> torch.Tensor:
        """Return the (seq, hidden) pre-final-norm target for batch element 0."""
        tgt = captured.get("target_full")
        if tgt is not None:
            return tgt[0]
        return out.hidden_states[-1][0]

    def _resolve_target_batched(self, captured: dict, out) -> torch.Tensor:
        """Return the (B, seq, hidden) pre-final-norm target."""
        tgt = captured.get("target_full")
        if tgt is not None:
            return tgt
        return out.hidden_states[-1]

    def _decoder_layer_module(self, layer: int):
        """Return the decoder block whose output is hidden_states[layer]."""
        base = self.model.model
        holder = base
        for _ in range(4):
            if hasattr(holder, "layers"):
                break
            for attr in ("model", "language_model", "transformer"):
                sub = getattr(holder, attr, None)
                if sub is not None:
                    holder = sub
                    break
            else:
                break
        layers = getattr(holder, "layers", None)
        if layers is None:
            return None
        idx = layer - 1  # hidden_states[layer] == output of block idx
        if 0 <= idx < len(layers):
            return layers[idx]
        return None

    # --------------------------------------------------------- persistence
    def _save_operators(self, path: str) -> None:
        try:
            os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
            torch.save(
                {
                    "jacobian": {l: t for l, t in self._jacobian.items()},
                    "signature": self._fit_signature,
                    "num_layers": self.model.info.num_layers,
                    "hidden_size": self.model.info.hidden_size,
                },
                path,
            )
            logger.info("Saved fitted lens operators to %s", path)
        except Exception as exc:
            logger.warning("Could not save lens operators: %s", exc)

    def _load_operators(self, path: str) -> None:
        if not os.path.exists(path):
            return
        try:
            blob = torch.load(path, map_location="cpu")
            if blob.get("hidden_size") != self.model.info.hidden_size:
                logger.warning("Cached lens hidden size mismatch; ignoring %s", path)
                return
            self._jacobian = {int(l): t.float() for l, t in blob["jacobian"].items()}
            self._fit_signature = blob.get("signature")
            logger.info("Loaded %d fitted lens operators from %s",
                        len(self._jacobian), path)
        except Exception as exc:
            logger.warning("Could not load lens operators: %s", exc)


def _valid_position_mask(seq_len: int, skip_first: int) -> torch.Tensor:
    """Boolean mask over positions to include in the Jacobian average.

    Early positions act as attention sinks and the final position has no
    next-token target, so both are excluded. Falls back gracefully for short
    prompts (keeps all but the last position).
    """
    mask = torch.zeros(seq_len, dtype=torch.bool)
    lo = min(skip_first, max(0, seq_len - 2))
    mask[lo: seq_len - 1] = True
    if mask.sum() == 0 and seq_len > 0:
        mask[max(0, seq_len - 2)] = True
    return mask


def default_corpus() -> list[str]:
    """A small default corpus for averaging the Jacobian.

    Deliberately generic English so the averaged operator reflects typical
    contexts rather than one topic. Users can supply their own corpus.
    """
    return [
        "The quick brown fox jumps over the lazy dog near the river bank.",
        "In the beginning the universe was created, which made many people angry.",
        "She sells seashells by the seashore on a bright summer morning today.",
        "The economy grew steadily as interest rates fell over the past year.",
        "Photosynthesis converts sunlight, water, and carbon dioxide into sugar.",
        "He opened the ancient book and began to read the long forgotten story.",
        "The recipe calls for two cups of flour, a fresh egg, and a pinch of salt.",
        "Scientists announced a surprising new discovery about distant galaxies.",
    ]

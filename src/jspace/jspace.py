"""Sparse J-space decomposition.

The paper defines the **J-space** at a layer as the set of points expressible as
a *sparse nonnegative* combination of J-lens vectors. Each J-lens vector is the
layer-``ell`` direction that most increases the (present/future) logit of a
particular vocabulary token, i.e. the columns of ``J_ell^T @ W_U^T`` -- one
direction per token.

Given an activation ``h_ell`` we approximate its J-space component by solving

    min_a  || h_ell - D a ||^2   s.t.  a >= 0,  ||a||_0 <= k

where the dictionary ``D`` has the (unit-normalized) J-lens vectors as columns.
We solve it with **gradient pursuit** (a matching-pursuit variant): greedily add
the atom whose correlation with the residual is largest, then take a
nonnegative least-squares style gradient step on the active set. The active
tokens and their coefficients are the activation's local J-space coordinates.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

import torch

from .config import LensConfig
from .lens import LensEngine
from .model import LensModel

logger = logging.getLogger(__name__)


@dataclass
class JSpaceAtom:
    token_id: int
    token: str
    coefficient: float


@dataclass
class JSpaceDecomposition:
    layer: int
    position: int
    atoms: list[JSpaceAtom]
    residual_fraction: float  # ||residual|| / ||h||  (0 = perfect, 1 = none)
    jspace_variance_fraction: float  # ||projection||^2 / ||h||^2

    def to_dict(self) -> dict:
        return {
            "layer": self.layer,
            "position": self.position,
            "atoms": [
                {"token_id": a.token_id, "token": a.token, "coefficient": a.coefficient}
                for a in self.atoms
            ],
            "residual_fraction": self.residual_fraction,
            "jspace_variance_fraction": self.jspace_variance_fraction,
        }


class JSpaceDecomposer:
    def __init__(self, model: LensModel, engine: LensEngine,
                 config: Optional[LensConfig] = None):
        self.model = model
        self.engine = engine
        self.config = config or LensConfig()

    def _candidate_dictionary(
        self, layer: int, h: torch.Tensor, n_candidates: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Build a SMALL J-lens dictionary for candidate tokens, fast.

        Uses the already-fitted operator ``J_l``:

          1. Pick ``n_candidates`` candidate tokens from the logit-lens readout
             of ``h`` (the atoms almost always lie among tokens the activation
             already projects onto).
          2. Each candidate's J-lens vector is ``d_v = J_l^T @ w_v``; computing
             all of them is a single matmul ``W_cand @ J_l`` (no backward pass).

        Returns ``(D, token_ids)`` with ``D`` shaped ``(hidden, n_candidates)``,
        unit-normalized columns.
        """
        lm_head = self.model._find_lm_head()
        W_U = lm_head.weight.detach()  # (vocab, hidden)

        # 1) Candidate tokens from the logit-lens readout of this activation.
        logits = self.model.unembed(h.unsqueeze(0)).squeeze(0).float()
        n_candidates = min(n_candidates, W_U.shape[0])
        cand_ids = torch.topk(logits, n_candidates).indices  # (n_cand,)
        W_cand = W_U[cand_ids].float()  # (n_cand, hidden)

        # 2) J_l^T @ W_cand^T via the fitted operator -> (n_cand, hidden).
        cols = self.engine.jacobian_transpose_batched(None, layer, W_cand)
        if cols is None:
            # Fallback: plain logit-lens directions (identity Jacobian).
            D = W_cand.t().to(h.device)
        else:
            D = cols.t().to(h.device)  # (hidden, n_cand)
        norms = D.norm(dim=0, keepdim=True).clamp_min(1e-8)
        D = D / norms
        return D, cand_ids.to(h.device)

    def decompose(
        self,
        prompt: str,
        layer: int,
        position: int,
        corpus=None,
        use_chat_template: bool = True,
        n_candidates: Optional[int] = None,
        input_ids=None,
    ) -> JSpaceDecomposition:
        """Sparse nonnegative decomposition of one activation onto J-lens atoms."""
        from .lens import default_corpus
        corpus = list(corpus) if corpus else default_corpus()
        n_candidates = n_candidates or self.config.decompose_candidates

        if input_ids is None:
            input_ids = self.model.build_input_ids(prompt, use_chat_template)
        hidden_states = self.model.forward_hidden_states(input_ids)
        num_layers = len(hidden_states) - 1
        seq_len = input_ids.shape[1]
        if not (0 <= layer <= num_layers):
            raise ValueError(f"layer {layer} out of range (0..{num_layers})")
        if not (0 <= position < seq_len):
            raise ValueError(
                f"position {position} out of range (0..{seq_len - 1})"
            )
        h = hidden_states[layer][0, position].float()  # (H,)

        # Ensure the operator for this layer is fitted (once), then decompose
        # with a single matmul.
        self.engine.ensure_fitted([layer], corpus, use_chat_template)
        D, token_ids = self._candidate_dictionary(layer, h, n_candidates)

        coeffs, active = _gradient_pursuit(
            h, D, k=self.config.jspace_sparsity_k,
            iters=self.config.pursuit_iters,
        )

        projection = D[:, active] @ coeffs
        residual = h - projection
        h_norm = h.norm().clamp_min(1e-8)
        residual_fraction = float((residual.norm() / h_norm).item())
        jspace_var = float(((projection.norm() ** 2) / (h_norm ** 2)).item())

        atoms = [
            JSpaceAtom(
                token_id=int(token_ids[idx].item()),
                token=self.model.decode_token(int(token_ids[idx].item())),
                coefficient=float(c.item()),
            )
            for idx, c in zip(active.tolist(), coeffs)
        ]
        atoms.sort(key=lambda a: a.coefficient, reverse=True)
        return JSpaceDecomposition(
            layer=layer, position=position, atoms=atoms,
            residual_fraction=residual_fraction,
            jspace_variance_fraction=jspace_var,
        )


def _gradient_pursuit(
    h: torch.Tensor, D: torch.Tensor, k: int, iters: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Greedy sparse nonnegative approximation of ``h`` by columns of ``D``.

    Returns ``(coeffs, active_indices)`` with ``coeffs >= 0`` and at most ``k``
    active atoms. Combines matching-pursuit atom selection with a projected
    gradient refinement of the active coefficients (a lightweight stand-in for
    the paper's gradient pursuit).
    """
    device = h.device
    residual = h.clone()
    active: list[int] = []
    coeffs = torch.zeros(0, device=device)

    for _ in range(k):
        # Correlation of each atom with the residual (nonnegative selection).
        corr = D.t() @ residual  # (vocab,)
        if active:
            corr[torch.tensor(active, device=device)] = float("-inf")
        best = int(torch.argmax(corr).item())
        if corr[best] <= 0:
            break
        active.append(best)

        # Refine all active coefficients with projected gradient descent
        # (nonnegative least squares on the active subdictionary).
        A = D[:, active]  # (H, |active|)
        c = torch.zeros(len(active), device=device)
        # Step size from the largest eigenvalue bound of A^T A (Lipschitz).
        AtA = A.t() @ A
        lip = torch.linalg.matrix_norm(AtA, ord=2).clamp_min(1e-6)
        step = 1.0 / lip
        Ath = A.t() @ h
        for _ in range(iters):
            grad = AtA @ c - Ath
            c = torch.clamp(c - step * grad, min=0.0)
        coeffs = c
        residual = h - A @ coeffs

    return coeffs, torch.tensor(active, device=device, dtype=torch.long)

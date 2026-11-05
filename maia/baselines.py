"""Non-interactive neuron-labeling baselines (paper Sec. 4.1, Table A3, Table 2).

* ``vlm_exemplar_label``  one-shot labeling of precomputed dataset exemplars by a VLM
                          (the paper's "MILAN (GPT-4V)" row; here with your configured VLM).
* ``milan_label``         original MILAN captioner (Hernandez et al. 2022) via its official repo.
* ``CLIPDissect``         CLIP-Dissect (Oikarinen & Weng 2022) with the SoftWPMI similarity.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import List

import numpy as np
import torch

from maia.backbone import Backbone

log = logging.getLogger(__name__)

EXEMPLAR_LABEL_PROMPT = (
    "These are the {n} real images that most strongly activate one neuron of a vision model. In each "
    "image the region that drives the activation is bright and outlined in red; the rest is darkened. "
    "Write a short, specific label (a few words) for the concept the neuron detects. Answer only with:\n"
    "[DESCRIPTION]: <one sentence>\n[LABEL]: <label>"
)


def vlm_exemplar_label(vlm: Backbone, exemplar_masks: List[str]) -> dict:
    from maia.agent import parse_fields

    out = vlm.complete(EXEMPLAR_LABEL_PROMPT.format(n=len(exemplar_masks)), images=exemplar_masks,
                       max_tokens=300, temperature=0.0)
    f = parse_fields(out, ("LABEL", "DESCRIPTION"))
    return {"label": f.get("LABEL", out.strip()), "description": f.get("DESCRIPTION", "")}


def milan_label(exemplar_images, exemplar_masks_np, model_key: str = "base"):
    """Label with the original MILAN model. Requires the official repo on PYTHONPATH:
        git clone https://github.com/evandez/neuron-descriptions && pip install -e neuron-descriptions
    """
    try:
        from src import milan  # type: ignore
    except ImportError as e:  # pragma: no cover
        raise ImportError("MILAN not installed: clone github.com/evandez/neuron-descriptions and add it to "
                          "PYTHONPATH (see README, 'Baselines').") from e
    model = milan.pretrained(model_key)
    imgs = torch.stack([torch.as_tensor(np.asarray(i), dtype=torch.float32).permute(2, 0, 1) / 255 for i in exemplar_images])
    masks = torch.as_tensor(np.stack(exemplar_masks_np)[:, None], dtype=torch.float32)
    caption = model.predict([(imgs, masks)])[0]  # API of the official repo
    return {"label": caption, "description": caption}


class CLIPDissect:
    """CLIP-Dissect with SoftWPMI (Oikarinen & Weng, 2022).

    acts      [N_images, N_units] max activation of each unit on the probe images
    concepts  list of concept words (the paper uses the 20k most common English words)
    """

    def __init__(self, device="cuda", clip_arch="ViT-B-16", pretrained="openai", lam=1.0, top_k=5, p=0.05):
        import open_clip

        self.model, _, self.preprocess = open_clip.create_model_and_transforms(clip_arch, pretrained=pretrained)
        self.tok = open_clip.get_tokenizer(clip_arch)
        self.model = self.model.to(device).eval()
        self.device, self.lam, self.top_k, self.p = device, lam, top_k, p

    @torch.no_grad()
    def image_embeddings(self, paths: List[str], batch_size: int = 128) -> torch.Tensor:
        from PIL import Image

        out = []
        for s in range(0, len(paths), batch_size):
            x = torch.stack([self.preprocess(Image.open(p).convert("RGB")) for p in paths[s : s + batch_size]]).to(self.device)
            e = self.model.encode_image(x)
            out.append(e / e.norm(dim=-1, keepdim=True))
        return torch.cat(out)

    @torch.no_grad()
    def text_embeddings(self, concepts: List[str], batch_size: int = 512) -> torch.Tensor:
        out = []
        for s in range(0, len(concepts), batch_size):
            t = self.tok(concepts[s : s + batch_size]).to(self.device)
            e = self.model.encode_text(t)
            out.append(e / e.norm(dim=-1, keepdim=True))
        return torch.cat(out)

    @torch.no_grad()
    def label(self, acts: np.ndarray, img_emb: torch.Tensor, txt_emb: torch.Tensor, concepts: List[str]) -> List[str]:
        P = (img_emb @ txt_emb.T).float()  # N x M
        P = torch.softmax(P / 0.01, dim=0)  # normalize over images, as in the paper's code
        A = torch.as_tensor(acts, dtype=torch.float32, device=P.device)  # N x U
        n = A.shape[0]
        k = max(1, int(self.p * n))
        labels = []
        logp_c = torch.log(P.mean(0) + 1e-10)  # M
        for u in range(A.shape[1]):
            idx = A[:, u].topk(k).indices
            # soft "top-k" membership: product over exemplar images of P(concept | image)
            logp_c_given_k = torch.log(1 - torch.prod(1 - P[idx], dim=0) + 1e-10)
            score = logp_c_given_k - self.lam * logp_c
            labels.append(concepts[int(score.argmax())])
        return labels


def load_concept_list(path: str | Path) -> List[str]:
    return [l.strip() for l in Path(path).read_text(encoding="utf-8").splitlines() if l.strip()]

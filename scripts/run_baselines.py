#!/usr/bin/env python
"""Baseline neuron labels for the sampled neurons (paper Sec. 4.1 / Table A3).

  --method vlm_exemplars   one-shot labeling of dataset exemplars by the describer VLM ("MILAN (GPT-4V)" row)
  --method milan           original MILAN (needs github.com/evandez/neuron-descriptions installed)
  --method clip_dissect    CLIP-Dissect, SoftWPMI, 20k concept set, probe set = the CC3M exemplar pool
"""
import logging

import numpy as np
import torch

from _common import ROOT, parser, setup

from maia.backbone import backbone_from_config
from maia.baselines import CLIPDissect, load_concept_list, milan_label, vlm_exemplar_label
from maia.utils import get_device, read_json, write_json
from maia.vision.exemplars import cache_path, list_images, load_exemplars
from maia.vision.models import forward_model, load_vision_model
from maia.vision.system import VisionNeuronSystem, pil_batch_to_tensor, preprocess_pil

log = logging.getLogger("baselines")


def _out(cfg, method, u):
    return ROOT / cfg["paths"]["outputs"] / "descriptions" / method / u["model"] / f"{u['layer']}_{u['unit']}" / "result.json"


@torch.no_grad()
def probe_activations(lm, layer, paths, device, bs=64):
    from PIL import Image

    mod = lm.resolve_layer(layer)
    store = {}
    h = mod.register_forward_hook(lambda m, i, o: store.__setitem__("o", o[0] if isinstance(o, tuple) else o))
    acts = []
    for s in range(0, len(paths), bs):
        imgs = [preprocess_pil(Image.open(p).convert("RGB"), lm.resolution) for p in paths[s:s + bs]]
        forward_model(lm, pil_batch_to_tensor(imgs, *lm.normalize, device))
        o = store["o"].float()
        acts.append((o.flatten(2).amax(2) if o.ndim == 4 else o[:, lm.n_prefix_tokens:].amax(1)).cpu())
    h.remove()
    return torch.cat(acts).numpy()


def main():
    p = parser(__doc__)
    p.add_argument("--model", required=True)
    p.add_argument("--method", choices=["vlm_exemplars", "milan", "clip_dissect"], required=True)
    p.add_argument("--probe-size", type=int, default=20000, help="CLIP-Dissect probe images")
    args = p.parse_args()
    cfg = setup(args)
    device = get_device(cfg["device"])
    units = read_json(ROOT / cfg["paths"]["outputs"] / "neurons.json")[args.model]
    lm = load_vision_model(args.model, device)
    from PIL import Image

    if args.method == "clip_dissect":
        cd = CLIPDissect(device)
        paths = list_images(ROOT / cfg["paths"]["cc3m_images"])[: args.probe_size]
        concepts = load_concept_list(ROOT / cfg["paths"]["clip_dissect_concepts"])
        img_emb, txt_emb = cd.image_embeddings(paths), cd.text_embeddings(concepts)
        for layer in sorted({u["layer"] for u in units}):
            acts = probe_activations(lm, layer, paths, device)
            lay_units = [u for u in units if u["layer"] == layer]
            labels = cd.label(acts[:, [u["unit"] for u in lay_units]], img_emb, txt_emb, concepts)
            for u, l in zip(lay_units, labels):
                write_json({**u, "label": l, "description": l}, _out(cfg, "clip_dissect", u))
        return

    vlm = backbone_from_config(cfg, "describer") if args.method == "vlm_exemplars" else None
    for u in units:
        out = _out(cfg, args.method, u)
        if out.exists():
            continue
        acts, idx, paths = load_exemplars(cache_path(ROOT / cfg["paths"]["exemplar_cache"], lm.name, u["layer"]))
        imgs = [Image.open(paths[i]).convert("RGB") for i in idx[u["unit"]][:15]]
        sys_ = VisionNeuronSystem(u["unit"], u["layer"], lm.name, device, loaded=lm)
        if args.method == "vlm_exemplars":
            _, masks = sys_.neuron(imgs)
            r = vlm_exemplar_label(vlm, masks)
        else:
            pim, _, maps = sys_.activation_maps(imgs)
            masks = [(m >= np.quantile(m, 0.95)).astype(np.float32) for m in maps]
            r = milan_label(pim, masks)
        write_json({**u, **r}, out)
        log.info("%s -> %s", u, r["label"])


if __name__ == "__main__":
    main()

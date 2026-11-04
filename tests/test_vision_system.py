import numpy as np
import torch
from PIL import Image

from maia.utils import base64_to_pil
from maia.vision.models import IMAGENET_MEAN, IMAGENET_STD, LoadedModel, _resnet_alias
from maia.vision.system import VisionNeuronSystem, render_mask
from maia.vision.tools import VisionTools


def tiny_resnet():
    from torchvision.models import resnet18

    m = resnet18(weights=None).eval()
    return LoadedModel("resnet18", m, (IMAGENET_MEAN, IMAGENET_STD), 224, "conv", _resnet_alias)


def test_neuron_returns_activation_and_mask():
    sys_ = VisionNeuronSystem(3, "layer4", "resnet18", "cpu", loaded=tiny_resnet())
    imgs = [Image.fromarray(np.random.RandomState(i).randint(0, 255, (300, 260, 3), dtype=np.uint8)) for i in range(3)]
    acts, masks = sys_.neuron(imgs)
    assert len(acts) == 3 and all(isinstance(a, float) for a in acts)
    assert base64_to_pil(masks[0]).size == (224, 224)


def test_activation_matches_manual_hook():
    lm = tiny_resnet()
    sys_ = VisionNeuronSystem(5, "layer3", "resnet18", "cpu", loaded=lm)
    img = Image.fromarray(np.full((224, 224, 3), 128, dtype=np.uint8))
    _, acts, _ = sys_.activation_maps([img])
    store = {}
    h = lm.model.layer3.register_forward_hook(lambda m, i, o: store.__setitem__("o", o))
    from maia.vision.system import pil_batch_to_tensor

    with torch.no_grad():
        lm.model(pil_batch_to_tensor([img], IMAGENET_MEAN, IMAGENET_STD, "cpu"))
    h.remove()
    assert abs(float(store["o"][0, 5].max()) - float(acts[0])) < 1e-4


def test_render_mask_inactive_is_dark():
    img = Image.new("RGB", (32, 32), (200, 200, 200))
    dark = np.asarray(render_mask(img, None, None))
    assert dark.max() < 100
    amap = np.zeros((32, 32))
    amap[8:16, 8:16] = 1
    out = np.asarray(render_mask(img, amap, 0.5))
    assert out[12, 12, 0] == 200 and out[0, 0, 0] < 100  # inside kept, outside darkened


def test_tool_ablation_blocks_tools():
    t = VisionTools(describer=None, allowed={"dataset_exemplars"})
    try:
        t.text2image(["a dog"])
        raise AssertionError("text2image should be disabled")
    except RuntimeError as e:
        assert "not available" in str(e)

"""Student models + deploy wrapper (uint8 BGR NHWC in -> P(pigeon) out)."""
from __future__ import annotations

import torch
import torch.nn as nn

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


class TinyPigeonNet(nn.Module):
    """~80k params, ~0.06 ms @64px on 1 CPU thread. Train from scratch or distil from the teacher."""

    def __init__(self, widths=(16, 32, 64, 96)):
        super().__init__()
        layers, c = [], 3
        for w in widths:
            layers += [nn.Conv2d(c, w, 3, 2, 1, bias=False), nn.BatchNorm2d(w), nn.ReLU(inplace=True),
                       nn.Conv2d(w, w, 3, 1, 1, groups=w, bias=False), nn.BatchNorm2d(w), nn.ReLU(inplace=True)]
            c = w
        self.features = nn.Sequential(*layers)
        self.head = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Dropout(0.1), nn.Linear(c, 1))

    def forward(self, x):
        return self.head(self.features(x))


def build_model(arch: str, pretrained: bool = True) -> tuple[nn.Module, tuple, tuple]:
    """Returns (model producing 1 logit, mean, std)."""
    if arch == "tiny":
        return TinyPigeonNet(), IMAGENET_MEAN, IMAGENET_STD
    import timm
    m = timm.create_model(arch, pretrained=pretrained, num_classes=1)
    cfg = m.pretrained_cfg or {}
    return m, tuple(cfg.get("mean", IMAGENET_MEAN)), tuple(cfg.get("std", IMAGENET_STD))


class Deploy(nn.Module):
    """Bakes preprocessing into the graph: uint8 BGR NHWC -> sigmoid prob (N,)."""

    def __init__(self, net: nn.Module, mean, std):
        super().__init__()
        self.net = net
        self.register_buffer("m", torch.tensor(mean).view(1, 3, 1, 1) * 255)
        self.register_buffer("s", torch.tensor(std).view(1, 3, 1, 1) * 255)

    def forward(self, x):
        x = x.permute(0, 3, 1, 2).flip(1).float()  # NHWC BGR -> NCHW RGB
        return torch.sigmoid(self.net((x - self.m) / self.s)).reshape(-1)


def export_onnx(net: nn.Module, mean, std, size: int, path: str, meta: dict | None = None):
    import json
    import onnx
    d = Deploy(net.eval(), mean, std).eval()
    x = torch.zeros(2, size, size, 3, dtype=torch.uint8)
    torch.onnx.export(d, (x,), path, opset_version=17, input_names=["crops"], output_names=["p_pigeon"],
                      dynamic_axes={"crops": {0: "n"}, "p_pigeon": {0: "n"}}, dynamo=False)
    m = onnx.load(path)
    if meta:
        e = m.metadata_props.add()
        e.key, e.value = "pigeonkit", json.dumps(meta)
    onnx.save(m, path)
    return path

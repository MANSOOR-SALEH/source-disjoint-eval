"""
Frozen-vision feature extractor for non-speech corpora (e.g. Camelyon17).

    python src/extract_vision.py --manifest manifests/camelyon.csv \
        --image-root /data/camelyon17_v1.0 --encoder resnet50 \
        --out features/camelyon.resnet50.npz

Mirrors src/extract.py for images: a frozen ImageNet ResNet-50 is used purely as
a measurement probe (never fine-tuned). The final convolutional feature map is
mean- and standard-deviation-pooled over spatial positions, giving one
$2\\times2048$ vector per patch---the image analogue of the masked mean+std
pooling used for the speech encoders.

Output matches src/probe.py's reader exactly:
    emb     float16 [N, 1, 2D]    (one "layer"; axis 1 kept for compatibility)
    utt_id  str     [N]           (same row order as the manifest)
    layers  int     [1] = [0]
    encoder str,  hf_id str

Because there is a single feature "layer", probe.py's fixed/convention/nested
layer selection all collapse to layer 0; the protocol contrast, source matching,
test restriction, draws, and feasibility checks are unchanged.
"""
import argparse
import os

import numpy as np
import pandas as pd


def _build(encoder, device, dtype):
    import torch
    import torch.nn as nn
    import torchvision.models as M
    if encoder != "resnet50":
        raise SystemExit("only resnet50 is wired here; add an arch if needed")
    net = M.resnet50(weights=M.ResNet50_Weights.IMAGENET1K_V2)
    body = nn.Sequential(*list(net.children())[:-2])  # -> [B,2048,h,w]
    body = body.to(device, dtype).eval()
    return body


def _transform():
    from torchvision import transforms as T
    return T.Compose([
        T.Resize(224), T.CenterCrop(224), T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--image-root", default="")
    ap.add_argument("--encoder", default="resnet50")
    ap.add_argument("--out", required=True)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--fp32", action="store_true")
    ap.add_argument("--limit", type=int, default=0, help="smoke-test on N patches")
    a = ap.parse_args()

    import torch
    from PIL import Image
    device = a.device if torch.cuda.is_available() else "cpu"
    dtype = torch.float32 if (a.fp32 or device == "cpu") else torch.float16

    df = pd.read_csv(a.manifest)
    if a.limit:
        df = df.head(a.limit)
    for c in ("utt_id", "path"):
        if c not in df.columns:
            raise SystemExit(f"manifest missing column '{c}'")

    body, tf = _build(a.encoder, device, dtype), _transform()
    emb = None
    ids = df.utt_id.astype(str).to_numpy()
    n = len(df)
    print(f"[{a.encoder}] {n} patches")

    buf, pos = [], []
    def flush(batch, idxs):
        nonlocal emb
        x = torch.stack(batch).to(device, dtype)
        with torch.no_grad():
            fmap = body(x).float()                 # [b,2048,h,w]
            mean = fmap.mean(dim=(2, 3))
            std = fmap.std(dim=(2, 3), unbiased=False)
            feat = torch.cat([mean, std], dim=1).cpu().numpy().astype(np.float16)
        if emb is None:
            emb = np.zeros((n, 1, feat.shape[1]), dtype=np.float16)
        emb[idxs, 0, :] = feat

    for i, p in enumerate(df.path.astype(str)):
        try:
            img = Image.open(os.path.join(a.image_root, p)).convert("RGB")
            buf.append(tf(img)); pos.append(i)
        except Exception as e:
            raise SystemExit(f"cannot read {p}: {e}")
        if len(buf) == a.batch_size:
            flush(buf, pos); buf, pos = [], []
            if (i + 1) % 2560 == 0:
                print(f"  {i + 1}/{n}")
    if buf:
        flush(buf, pos)

    if not np.isfinite(emb.astype(np.float32)).all():
        raise SystemExit("non-finite features; retry with --fp32")
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    np.savez_compressed(a.out, emb=emb, utt_id=ids, layers=np.array([0]),
                        encoder=np.array(a.encoder),
                        hf_id=np.array(f"torchvision/{a.encoder}"))
    print(f"wrote {a.out}  shape={emb.shape}  {os.path.getsize(a.out)/1e9:.2f} GB")


if __name__ == "__main__":
    main()

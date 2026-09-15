"""
Extract per-layer mean+std pooled embeddings for one (corpus, encoder) pair.

    python src/extract.py --manifest manifests/aydid.csv --encoder wavlm \
        --audio-root /data/AYDID --out features/aydid.wavlm.npz

Manifest columns (CSV, header required):
    utt_id, path, label, speaker_id, source_id
`path` is relative to --audio-root. `source_id` is the recording-provenance
variable: programme / show_name for AYDID, recording batch or country for SADIC.

Output: a single .npz with
    emb      float16 [N, n_layers+1, 2D]   layer 0 = pre-transformer output
    utt_id   str     [N]
    layers   int     [n_layers+1]

Size check: AYDID (17.5k utts) x 25 layers x 2048 dims x 2 bytes = 1.8 GB.
Budget ~13 GB for seven encoders on AYDID plus ~4 GB on SADIC.
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd
import soundfile as sf
import torch
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(__file__))
from encoders import build, REGISTRY  # noqa: E402


def load_wav(path: str, target_sr: int = 16000) -> np.ndarray:
    wav, sr = sf.read(path, dtype="float32", always_2d=False)
    if wav.ndim > 1:
        wav = wav.mean(axis=1)
    if sr != target_sr:
        raise ValueError(
            f"{path} is {sr} Hz. Resample the corpus once, offline, rather than "
            f"per-encoder -- a resampler in the extraction loop is a silent "
            f"source of encoder-to-encoder differences."
        )
    return wav


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--encoder", required=True, choices=sorted(REGISTRY))
    ap.add_argument("--audio-root", default="")
    ap.add_argument("--out", required=True)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--fp32", action="store_true", help="disable fp16 (needed on some CPUs)")
    ap.add_argument("--limit", type=int, default=0, help="smoke-test on N utterances")
    args = ap.parse_args()

    df = pd.read_csv(args.manifest)
    required = {"utt_id", "path", "label", "speaker_id", "source_id"}
    missing = required - set(df.columns)
    if missing:
        raise SystemExit(f"manifest missing columns: {sorted(missing)}")
    if args.limit:
        df = df.head(args.limit)

    # Sort by duration so padding waste stays small; restore original order after.
    paths = [os.path.join(args.audio_root, p) for p in df["path"]]
    sizes = np.array([os.path.getsize(p) for p in paths])
    order = np.argsort(sizes)

    enc = build(args.encoder, device=args.device,
                dtype=torch.float32 if args.fp32 else torch.float16)
    print(f"[{args.encoder}] {enc.spec.hf_id} | {enc.spec.notes}")

    embs, done = None, np.zeros(len(df), dtype=bool)
    for i in tqdm(range(0, len(order), args.batch_size), desc=args.encoder):
        idx = order[i: i + args.batch_size]
        wavs = [load_wav(paths[j]) for j in idx]
        e = enc.embed(wavs)                      # [b, L, 2D]
        if embs is None:
            embs = np.zeros((len(df), e.shape[1], e.shape[2]), dtype=np.float16)
        embs[idx] = e.astype(np.float16)
        done[idx] = True

    assert done.all(), f"{(~done).sum()} utterances not embedded"
    if not np.isfinite(embs.astype(np.float32)).all():
        raise SystemExit("non-finite values in embeddings -- check fp16 overflow, retry with --fp32")

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    np.savez_compressed(
        args.out,
        emb=embs,
        utt_id=df["utt_id"].to_numpy().astype(str),
        layers=np.arange(embs.shape[1]),
        encoder=np.array(args.encoder),
        hf_id=np.array(enc.spec.hf_id),
    )
    print(f"wrote {args.out}  shape={embs.shape}  "
          f"{os.path.getsize(args.out) / 1e9:.2f} GB")


if __name__ == "__main__":
    main()

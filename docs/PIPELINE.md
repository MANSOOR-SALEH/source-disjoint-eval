# Pipeline

The analysis tools operate on **pooled per-layer embeddings**, not on audio.
Extraction is done once; every protocol, fold and probe afterwards is cheap.

## Feature format

A `.npz` per (corpus, encoder):

| key | shape | meaning |
|---|---|---|
| `emb` | `[N, L, 2D]` float16 | mean- and std-pooled hidden states, one row per manifest row, in manifest order |
| `utt_id` | `[N]` str | must match the manifest exactly, in order |
| `layers` | `[L]` int | layer indices, `0` = pre-transformer output |
| `encoder` | scalar str | encoder key |

`probe.py` asserts that `utt_id` matches the manifest row-for-row and refuses to
run otherwise; a subset of a larger feature file can be selected with
`--subset-of`.

## Pooling

Pool with a length mask. Padding must not enter the statistics: Whisper pads
every clip to 30 s and batched wav2vec2-family inputs are padded to the batch
maximum, so unmasked pooling makes utterance duration an encoder-dependent
feature — the confound the protocol is meant to control.

Pool in float32 even when the forward pass runs in float16. The variance term
squares the hidden states and overflows fp16 for large-activation encoders.

## Audio

Normalise the corpus once, offline, to a single sample rate, channel count and
encoding. Resampling inside the extraction loop is a silent source of
encoder-to-encoder differences. Check for format variation that correlates with
the label before normalising: if one class is disproportionately 44.1 kHz or
float-encoded, that is itself a provenance signal worth reporting.

Restrict clip duration to a band that every encoder handles identically.
Whisper truncates at 30 s, so clips beyond that are not comparable across
encoders.

## Order of operations

```
manifest -> class-completeness check   (abort here if it fails)
         -> folds (size + source matching + test restriction)
         -> features (once per encoder)
         -> probe
         -> coverage check
         -> stats
         -> across-draw repeat
```

The class-completeness check comes first because it can rule the protocol out
entirely, before any GPU time is spent.

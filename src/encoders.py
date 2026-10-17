"""
Encoder registry for the source-disjoint encoder benchmark.

Every encoder exposes the same interface:
    enc = build("wavlm")
    emb = enc.embed(waveforms, lengths)   # -> np.ndarray [B, n_layers, 2*D]  (mean||std)

Design notes
------------
* We pool at extraction time. Storing per-layer mean+std vectors turns every
  downstream probe into a CPU-cheap operation on a [N, n_layers, 2D] array.
* Masked pooling is mandatory: Whisper pads every clip to 30 s, and batched
  wav2vec2-family inputs are padded to the batch max. Pooling over pad frames
  silently makes short utterances look different from long ones -- which is the
  exact duration confound this project is supposed to control for.
* Valid output-frame counts are derived from the input/output length ratio
  rather than per-model formulas, so adding an encoder needs no new maths.
"""

from dataclasses import dataclass
from typing import Callable, List, Optional

import numpy as np
import torch


# --------------------------------------------------------------------------- #
# Registry
# --------------------------------------------------------------------------- #

@dataclass
class EncoderSpec:
    key: str
    hf_id: str
    family: str            # "w2v2" | "whisper" | "w2vbert" | "artst"
    n_layers: int          # transformer layers (hidden_states has n_layers + 1)
    dim: int
    notes: str = ""


REGISTRY = {
    "wavlm":     EncoderSpec("wavlm",     "microsoft/wavlm-large",            "w2v2",    24, 1024,
                             "P3 baseline encoder; already extracted in confusion_prior_audit"),
    "xlsr":      EncoderSpec("xlsr",      "facebook/wav2vec2-xls-r-300m",     "w2v2",    24, 1024,
                             "P2 primary encoder"),
    "mhubert":   EncoderSpec("mhubert",   "utter-project/mHuBERT-147",        "w2v2",    12,  768,
                             "compact multilingual; only 12 layers -- do not compare layer index across encoders"),
    "whisper":   EncoderSpec("whisper",   "openai/whisper-large-v3",          "whisper", 32, 1280,
                             "encoder only; weak supervision; P3 winner under speaker-disjoint"),
    "mms":       EncoderSpec("mms",       "facebook/mms-300m",                "w2v2",    24, 1024,
                             "same architecture family as XLS-R -- differs in pretraining data, not design"),
    "w2vbert":   EncoderSpec("w2vbert",   "facebook/w2v-bert-2.0",            "w2vbert", 24, 1024,
                             "the SeamlessM4T-v2 speech encoder, released standalone; cleaner than "
                             "digging it out of the full seamless checkpoint"),
    "artst":     EncoderSpec("artst",     "MBZUAI/artst_asr_v2",              "artst",   12,  768,
                             "Arabic-specific, SpeechT5 backbone, ASR-fine-tuned on MGB2. NOTE: "
                             "'MBZUAI/ArTST' is a raw fairseq checkpoint with no transformers "
                             "preprocessor -- only the artst_asr_* repos load through transformers. "
                             "Alternative: MBZUAI/artst_asr_v3_qasr, fine-tuned on QASR and "
                             "described by the authors as better for dialectal Arabic."),
    # Fallback if ArTST integration stalls: an Arabic-fine-tuned wav2vec2 keeps
    # the 'Arabic-specific pretraining' arm alive at near-zero integration cost.
    "ar_w2v2":   EncoderSpec("ar_w2v2",   "jonatasgrosman/wav2vec2-large-xlsr-53-arabic", "w2v2", 24, 1024,
                             "FALLBACK for artst only -- do not report both without saying why"),
}


# --------------------------------------------------------------------------- #
# Base
# --------------------------------------------------------------------------- #

class BaseEncoder:
    def __init__(self, spec: EncoderSpec, device: str = "cuda", dtype=torch.float16):
        self.spec = spec
        self.device = device
        self.dtype = dtype
        self._load()

    def _load(self):
        raise NotImplementedError

    def _forward(self, waveforms: List[np.ndarray]):
        """Return (hidden_states, valid_frames) with hidden_states a list of
        [B, T, D] tensors (embedding output + every transformer layer)."""
        raise NotImplementedError

    @torch.no_grad()
    def embed(self, waveforms: List[np.ndarray]) -> np.ndarray:
        hs, valid = self._forward(waveforms)
        out = []
        for layer in hs:                                    # [B, T, D]
            # Pool in fp32 even when the forward pass ran in fp16: the variance
            # term squares the hidden states, and fp16 tops out at 65504, so
            # WavLM-scale activations overflow to inf here and nowhere else.
            layer = layer.float()
            B, T, D = layer.shape
            idx = torch.arange(T, device=layer.device)[None, :]      # [1, T]
            mask = (idx < valid[:, None]).to(layer.dtype)            # [B, T]
            denom = mask.sum(1).clamp(min=1)[:, None]                # [B, 1]
            m = (layer * mask[:, :, None]).sum(1) / denom
            var = ((layer - m[:, None, :]) ** 2 * mask[:, :, None]).sum(1) / denom
            s = var.clamp(min=1e-8).sqrt()
            out.append(torch.cat([m, s], dim=-1))                    # [B, 2D]
        return torch.stack(out, dim=1).float().cpu().numpy()         # [B, L, 2D]

    @staticmethod
    def _valid_frames(n_samples: torch.Tensor, padded_samples: int, out_T: int) -> torch.Tensor:
        """Generic length mapping: output frames scale with input samples."""
        v = torch.round(out_T * n_samples.float() / float(padded_samples))
        return v.clamp(min=1, max=out_T).long()


# --------------------------------------------------------------------------- #
# wav2vec2 family: WavLM, XLS-R, mHuBERT, MMS, Arabic w2v2
# --------------------------------------------------------------------------- #

class W2V2Encoder(BaseEncoder):
    def _load(self):
        from transformers import AutoFeatureExtractor, AutoModel
        self.fe = AutoFeatureExtractor.from_pretrained(self.spec.hf_id)
        self.model = AutoModel.from_pretrained(
            self.spec.hf_id, torch_dtype=self.dtype
        ).to(self.device).eval()

    def _forward(self, waveforms):
        n = torch.tensor([len(w) for w in waveforms], device=self.device)
        batch = self.fe(waveforms, sampling_rate=16000, return_tensors="pt",
                        padding=True, return_attention_mask=True)
        inputs = batch["input_values"].to(self.device, self.dtype)
        am = batch.get("attention_mask")
        kw = {"attention_mask": am.to(self.device)} if am is not None else {}
        out = self.model(inputs, output_hidden_states=True, **kw)
        hs = out.hidden_states
        out_T = hs[0].shape[1]
        # exact length map when the model exposes it, generic ratio otherwise
        try:
            valid = self.model._get_feat_extract_output_lengths(n).clamp(max=out_T)
        except Exception:
            valid = self._valid_frames(n, inputs.shape[-1], out_T)
        return hs, valid


# --------------------------------------------------------------------------- #
# Whisper (encoder only)
# --------------------------------------------------------------------------- #

class WhisperEncoder(BaseEncoder):
    PADDED = 480_000   # Whisper always pads to 30 s @ 16 kHz

    def _load(self):
        from transformers import WhisperFeatureExtractor, WhisperModel
        self.fe = WhisperFeatureExtractor.from_pretrained(self.spec.hf_id)
        full = WhisperModel.from_pretrained(self.spec.hf_id, torch_dtype=self.dtype)
        self.model = full.encoder.to(self.device).eval()
        del full

    def _forward(self, waveforms):
        clipped = [w[: self.PADDED] for w in waveforms]
        n = torch.tensor([len(w) for w in clipped], device=self.device)
        feats = self.fe(clipped, sampling_rate=16000, return_tensors="pt").input_features
        feats = feats.to(self.device, self.dtype)
        out = self.model(feats, output_hidden_states=True)
        hs = out.hidden_states
        out_T = hs[0].shape[1]                       # 1500 for large-v3
        valid = self._valid_frames(n, self.PADDED, out_T)
        return hs, valid


# --------------------------------------------------------------------------- #
# w2v-BERT 2.0  (= SeamlessM4T-v2 speech encoder)
# --------------------------------------------------------------------------- #

class W2VBertEncoder(BaseEncoder):
    def _load(self):
        from transformers import AutoFeatureExtractor, AutoModel
        self.fe = AutoFeatureExtractor.from_pretrained(self.spec.hf_id)
        self.model = AutoModel.from_pretrained(
            self.spec.hf_id, torch_dtype=self.dtype
        ).to(self.device).eval()

    def _forward(self, waveforms):
        n = torch.tensor([len(w) for w in waveforms], device=self.device)
        batch = self.fe(waveforms, sampling_rate=16000, return_tensors="pt", padding=True)
        inputs = batch["input_features"].to(self.device, self.dtype)
        am = batch.get("attention_mask")
        kw = {"attention_mask": am.to(self.device)} if am is not None else {}
        out = self.model(inputs, output_hidden_states=True, **kw)
        hs = out.hidden_states
        out_T = hs[0].shape[1]
        pad_samples = int(n.max().item())
        valid = self._valid_frames(n, pad_samples, out_T)
        return hs, valid


# --------------------------------------------------------------------------- #
# ArTST  -- integration stub, TIME-BOXED
# --------------------------------------------------------------------------- #

class ArTSTEncoder(BaseEncoder):
    """
    ArTST, SpeechT5 backbone. Two things differ from the other adapters:

    * The published `MBZUAI/ArTST` repo is a fairseq checkpoint with no
      transformers preprocessor, so it cannot load here. The `artst_asr_*`
      repos are the transformers exports and do carry preprocessor_config.json.
    * We take the encoder out of SpeechT5ForSpeechToText and use only the
      SpeechT5FeatureExtractor, skipping SpeechT5Processor entirely -- the
      processor pulls in a sentencepiece tokenizer we have no use for.

    CAVEAT FOR THE PAPER: this checkpoint is ASR-fine-tuned, not a raw
    Arabic-pretrained encoder. That puts it in the same category as Whisper
    (supervised ASR) rather than in the self-supervised group, and the paper
    must say so rather than filing it under 'Arabic-pretrained'.
    """

    def _load(self):
        from transformers import SpeechT5FeatureExtractor, SpeechT5ForSpeechToText
        self.fe = SpeechT5FeatureExtractor.from_pretrained(self.spec.hf_id)
        full = SpeechT5ForSpeechToText.from_pretrained(
            self.spec.hf_id, torch_dtype=self.dtype)
        self.model = full.speecht5.encoder.to(self.device).eval()
        del full

    def _forward(self, waveforms):
        n = torch.tensor([len(w) for w in waveforms], device=self.device)
        batch = self.fe(audio=waveforms, sampling_rate=16000, return_tensors="pt",
                        padding=True, return_attention_mask=True)
        inputs = batch["input_values"].to(self.device, self.dtype)
        am = batch.get("attention_mask")
        kw = {"attention_mask": am.to(self.device)} if am is not None else {}
        out = self.model(input_values=inputs, output_hidden_states=True, **kw)
        hs = out.hidden_states
        out_T = hs[0].shape[1]
        valid = self._valid_frames(n, inputs.shape[-1], out_T)
        return hs, valid


_FAMILIES: dict = {
    "w2v2": W2V2Encoder,
    "whisper": WhisperEncoder,
    "w2vbert": W2VBertEncoder,
    "artst": ArTSTEncoder,
}


def build(key: str, device: str = "cuda", dtype=torch.float16) -> BaseEncoder:
    spec = REGISTRY[key]
    return _FAMILIES[spec.family](spec, device=device, dtype=dtype)
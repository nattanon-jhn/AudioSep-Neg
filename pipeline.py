# pipeline.py
# =========================================================
# AudioSep-Neg Inference Pipeline (Training-like Architecture)
# =========================================================

import os
from typing import Optional, Dict, Any

import torch
import numpy as np
import librosa
from scipy.io.wavfile import write

from utils import ignore_warnings, parse_yaml, load_ss_model
from models.clap_encoder import CLAP_Encoder
from llm_router import route_prompt, rule_based_router


# =========================================================
# Build AudioSep model
# =========================================================
def build_audiosep(
    config_yaml: str,
    checkpoint_path: str,
    device: torch.device,
    clap_ckpt_path: str = "checkpoint/music_speech_audioset_epoch_15_esc_89.98.pt",
):
    """
    Build AudioSep model for inference.

    - Uses the SAME AudioSep LightningModule as training
    - Loads CLAP + AudioSep checkpoint
    """

    ignore_warnings()
    configs = parse_yaml(config_yaml)

    # -----------------------------------------------------
    # CLAP encoder (frozen, same as training)
    # -----------------------------------------------------
    query_encoder = CLAP_Encoder(
        pretrained_path=clap_ckpt_path,
        sampling_rate=32000,
        amodel="HTSAT-base",
    )

    try:
        query_encoder.load_audio_pretrained(clap_ckpt_path)
        print("✔ Loaded CLAP pretrained weights")
    except Exception as e:
        print(f"⚠️ Failed to load CLAP pretrained weights: {repr(e)}")
        print("⚠️ Using random-initialized CLAP")

    query_encoder.eval()

    # -----------------------------------------------------
    # Load AudioSep (Lightning checkpoint)
    # -----------------------------------------------------
    model = load_ss_model(
        configs=configs,
        checkpoint_path=checkpoint_path,
        query_encoder=query_encoder,
    ).eval().to(device)

    print(f"✅ Loaded AudioSep checkpoint: {checkpoint_path}")
    return model


# =========================================================
# Audio separation (training-like forward)
# =========================================================
def separate_audio(
    model,
    audio_file: str,
    text: str,
    output_file: str,
    device: str | torch.device = "cuda",
    enable_llm_router: bool = True,
    llm_router_kwargs: Optional[Dict[str, Any]] = None,
):
    """
    Inference that mirrors TRAINING architecture:

    mixture
      → CLAP(text)
      → DualQueryFusion
      → ss_model
      → FG / BG head
      → routing (same semantics as training)
    """

    llm_router_kwargs = llm_router_kwargs or {}

    print(f"\n🎧 Input audio : {audio_file}")
    print(f"🧭 User prompt: {text}")

    # -----------------------------------------------------
    # Load mixture (same preprocessing as training)
    # -----------------------------------------------------
    mixture, _ = librosa.load(audio_file, sr=32000, mono=True)
    mixture = mixture.astype(np.float32)
    mixture_t = torch.tensor(mixture)[None, None, :].to(device)  # (1,1,T)

    # -----------------------------------------------------
    # Prompt routing (LLM / rule-based)
    # -----------------------------------------------------
    if enable_llm_router:
        pr = route_prompt(
            text,
            enable_llm_router=True,
            llm_kwargs=llm_router_kwargs,
        )
    else:
        pr = rule_based_router(text)

    print(
        f"🧾 Routed -> class={pr.cls}, route={pr.route}, "
        f"positive='{pr.positive_sentence}', negative='{pr.negative_sentence}'"
    )

    # -----------------------------------------------------
    # Build text inputs (B=1, training-compatible)
    # -----------------------------------------------------
    text_pos = [pr.positive_sentence or ""]
    text_neg = [pr.negative_sentence or ""]

    # -----------------------------------------------------
    # TRAINING-LIKE FORWARD
    # -----------------------------------------------------
    with torch.no_grad():

        # ---- CLAP embeddings (text-only; hybrid not possible in inference) ----
        e_pos = model.query_encoder.get_query_embed(
            modality="text",
            text=text_pos,
            device=device,
        )

        if text_neg[0].strip():
            e_neg = model.query_encoder.get_query_embed(
                modality="text",
                text=text_neg,
                device=device,
            )
        else:
            e_neg = torch.zeros_like(e_pos)

        # ---- Dual-query fusion (same as training) ----
        e_mix = model.fusion(e_pos, e_neg)

        # ---- Backbone ----
        out = model.ss_model(
            {
                "mixture": mixture_t,
                "condition": e_mix,
            }
        )

        base = out["waveform"] if isinstance(out, dict) else out
        if base.dim() == 2:
            base = base.unsqueeze(1)  # (B,1,T)

        # ---- FG / BG heads (same as training) ----
        y_fg = model.fg_head(base)  # (1,1,T)
        y_bg = model.bg_head(base)  # (1,1,T)

        # ---- Routing (same rule as training) ----
        route = model._route_from_prompt_type(pr.cls)
        out_wave = y_bg if route == "BG" else y_fg

        out_wave = out_wave.squeeze().cpu().numpy().astype(np.float32)

    # -----------------------------------------------------
    # Safety + write wav
    # -----------------------------------------------------
    out_wave = np.clip(out_wave, -1.0, 1.0)

    write(
        output_file,
        32000,
        np.round(out_wave * 32767).astype(np.int16),
    )

    print(f"💾 Saved output: {output_file}")


# =========================================================
# CLI usage
# =========================================================
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = build_audiosep(
        config_yaml="config/audiosep_base_step2.yaml",
        checkpoint_path="checkpoint/step=40000.ckpt",
        device=device,
        clap_ckpt_path="checkpoint/music_speech_audioset_epoch_15_esc_89.98.pt",
    )

    separate_audio(
        model=model,
        audio_file="input.wav",
        text="keep water drop",
        output_file="output.wav",
        device=device,
        enable_llm_router=True,
    )

# losses.py
# ============================================================
# AudioSep-Neg Losses
#
# Supports:
#  - Step 1: routing + head training
#  - Step 2: semantic conditioning + refinement
#
# Losses:
#   - l1_wav
#   - l1_wav_consistency
#   - l1_wav_energy
#   - l1_wav_energy_multiscale   (RECOMMENDED for Step 2)
# ============================================================

import torch
import torch.nn.functional as F
import torchaudio


# ============================================================
# Basic helpers
# ============================================================

def l1(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return torch.mean(torch.abs(x - y))


def energy(x: torch.Tensor) -> torch.Tensor:
    """
    Mean square energy
    x: (B, T)
    """
    return torch.mean(x ** 2)


# ============================================================
# 1) L1 waveform loss
# ============================================================

def l1_wav(output_dict, target_dict):
    """
    Basic L1 waveform loss

    Required keys:
      output_dict["segment"]: (B, T)
      target_dict["segment"]: (B, T)
    """
    return l1(output_dict["segment"], target_dict["segment"])


# ============================================================
# 2) L1 + mixture consistency
# ============================================================

def l1_wav_consistency(
    output_dict,
    target_dict,
    lambda_consistency: float = 0.1,
):
    """
    L1 waveform + mixture consistency

    FG + BG ≈ mixture

    Required:
      output_dict["segment"]
      target_dict["segment"]

    Optional (for consistency):
      output_dict["fg"]
      output_dict["bg"]
      target_dict["mixture"]
    """

    loss_main = l1(output_dict["segment"], target_dict["segment"])

    loss_consistency = 0.0
    if (
        "fg" in output_dict
        and "bg" in output_dict
        and "mixture" in target_dict
    ):
        recon = output_dict["fg"] + output_dict["bg"]
        loss_consistency = l1(recon, target_dict["mixture"])

    return loss_main + lambda_consistency * loss_consistency


# ============================================================
# 3) L1 + energy preservation
# ============================================================

def l1_wav_energy(
    output_dict,
    target_dict,
    energy_weight: float = 0.03,
):
    """
    L1 waveform + energy preservation

    Useful for:
      - BG routing
      - preventing silence collapse
    """

    loss_main = l1(output_dict["segment"], target_dict["segment"])

    e_pred = energy(output_dict["segment"])
    e_tgt = energy(target_dict["segment"])

    loss_energy = torch.abs(e_pred - e_tgt)

    return loss_main + energy_weight * loss_energy


# ============================================================
# 4) L1 + energy + MULTI-SCALE spectral loss (Step 2 ⭐)
# ============================================================

def multiscale_spectral_loss(
    x: torch.Tensor,
    y: torch.Tensor,
    sample_rate: int = 32000,
    scales=(1024, 2048, 4096),
):
    """
    Multi-scale STFT magnitude loss
    x, y: (B, T)
    """
    loss = 0.0
    for n_fft in scales:
        hop = n_fft // 4

        X = torch.stft(
            x,
            n_fft=n_fft,
            hop_length=hop,
            win_length=n_fft,
            return_complex=True,
        )
        Y = torch.stft(
            y,
            n_fft=n_fft,
            hop_length=hop,
            win_length=n_fft,
            return_complex=True,
        )

        loss += torch.mean(torch.abs(torch.abs(X) - torch.abs(Y)))

    return loss / len(scales)


def l1_wav_energy_multiscale(
    output_dict,
    target_dict,
    lambda_energy: float = 0.03,
    lambda_spec: float = 0.1,
):
    """
    ⭐ RECOMMENDED Step 2 Loss ⭐

    Combines:
      - L1 waveform
      - Energy preservation
      - Multi-scale spectral consistency

    Required:
      output_dict["segment"]
      target_dict["segment"]
    """

    pred = output_dict["segment"]
    tgt = target_dict["segment"]

    # main L1
    loss_main = l1(pred, tgt)

    # energy
    e_pred = energy(pred)
    e_tgt = energy(tgt)
    loss_energy = torch.abs(e_pred - e_tgt)

    # spectral
    loss_spec = multiscale_spectral_loss(pred, tgt)

    return (
        loss_main
        + lambda_energy * loss_energy
        + lambda_spec * loss_spec
    )


# ============================================================
# Loss selector
# ============================================================

def get_loss_function(loss_type: str):

    if loss_type == "l1_wav":
        return l1_wav

    elif loss_type == "l1_wav_consistency":
        return l1_wav_consistency

    elif loss_type == "l1_wav_energy":
        return l1_wav_energy

    elif loss_type == "l1_wav_energy_multiscale":
        return l1_wav_energy_multiscale

    else:
        raise NotImplementedError(f"Unknown loss_type: {loss_type}")

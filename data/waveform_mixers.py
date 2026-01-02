# data/waveform_mixers.py
# ============================================================
# Prompt-aware SegmentMixer (Canonical, Semantic-clean)
#
# prompt_type semantics:
#   positive      -> keep FG
#       text_pos = FG
#       text_neg = ""
#
#   negative      -> remove FG (keep BG)
#       text_pos = ""
#       text_neg = FG
#
#   negation      -> all except FG (keep BG)
#       text_pos = ""
#       text_neg = FG
#
#   contrastive   -> keep FG, remove BG
#       text_pos = FG
#       text_neg = BG
# ============================================================

import random
import torch
import torch.nn as nn


class SegmentMixer(nn.Module):
    def __init__(
        self,
        max_mix_num: int,
        lower_db: int,
        higher_db: int,
        prompt_types=("positive", "negative", "negation", "contrastive"),
    ):
        super().__init__()
        self.max_mix_num = max_mix_num
        self.lower_db = lower_db
        self.higher_db = higher_db
        self.prompt_types = list(prompt_types)

    def __call__(self, waveforms: torch.Tensor, captions):
        """
        Args:
            waveforms: (B, T)
            captions : list[str] length B

        Returns:
            dict with keys:
              mixture, fg, bg           -> (B, T)
              caption_pos, caption_neg  -> list[str]
              prompt_type               -> list[str]
        """
        B, T = waveforms.shape

        mixtures = []
        fg_list = []
        bg_list = []
        caption_pos_list = []
        caption_neg_list = []
        prompt_type_list = []

        for i in range(B):
            # ---------------------------
            # Select FG / BG
            # ---------------------------
            fg = waveforms[i].clone()
            fg_caption = captions[i]

            bg_idx = random.choice([j for j in range(B) if j != i])
            bg = waveforms[bg_idx].clone()
            bg_caption = captions[bg_idx]

            # Loudness normalization
            bg = dynamic_loudnorm(bg, fg, self.lower_db, self.higher_db)

            mixture = fg + bg

            # Prevent clipping
            max_val = torch.max(torch.abs(mixture))
            if max_val > 1:
                scale = 0.9 / max_val
                fg *= scale
                bg *= scale
                mixture *= scale

            # ---------------------------
            # Prompt type
            # ---------------------------
            prompt_type = random.choice(self.prompt_types)

            # ---------------------------
            # CANONICAL SEMANTIC MAPPING
            # ---------------------------
            if prompt_type == "positive":
                caption_pos = fg_caption
                caption_neg = ""

            elif prompt_type in ["negative", "negation"]:
                caption_pos = ""
                caption_neg = fg_caption

            elif prompt_type == "contrastive":
                caption_pos = fg_caption
                caption_neg = bg_caption

            else:
                raise ValueError(f"Unknown prompt_type: {prompt_type}")

            # ---------------------------
            # Collect
            # ---------------------------
            mixtures.append(mixture)
            fg_list.append(fg)
            bg_list.append(bg)
            caption_pos_list.append(caption_pos)
            caption_neg_list.append(caption_neg)
            prompt_type_list.append(prompt_type)

        return {
            "mixture": torch.stack(mixtures),       # (B, T)
            "fg": torch.stack(fg_list),              # (B, T)
            "bg": torch.stack(bg_list),              # (B, T)
            "caption_pos": caption_pos_list,         # list[str]
            "caption_neg": caption_neg_list,         # list[str]
            "prompt_type": prompt_type_list,         # list[str]
        }


# ============================================================
# Loudness utilities
# ============================================================

def get_energy(x: torch.Tensor) -> torch.Tensor:
    return torch.mean(x ** 2)


def rescale_to_match_energy(audio: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
    ratio = (get_energy(audio) / max(get_energy(reference), 1e-8)) ** 0.5
    ratio = torch.clamp(ratio, 0.02, 50.0)
    return audio / ratio


def dynamic_loudnorm(
    audio: torch.Tensor,
    reference: torch.Tensor,
    lower_db: int = -10,
    higher_db: int = 10,
) -> torch.Tensor:
    audio = rescale_to_match_energy(audio, reference)
    delta_db = random.randint(lower_db, higher_db)
    gain = 10 ** (delta_db / 20.0)
    return gain * audio

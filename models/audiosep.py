# models/audiosep.py
# ============================================================
# AudioSep-Neg (STEP 2 – Cleaned & Hardened)
#
# STEP 2 goals:
# - REAL CLAP conditioning
# - Train decoder (+ FiLM if exists)
# - Freeze encoder + CLAP
# - Per-sample routing + per-sample loss
# - FG / BG heads
# - Correct & rich debug audio (mixture / target / non-target / pred)
#
# Canonical semantics:
#   positive    -> route FG
#   contrastive -> route FG
#   negative    -> route BG
#   negation    -> route BG
# ============================================================

from typing import Any, Dict, List, Optional, Union
import os
import random

import lightning.pytorch as pl
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR
import soundfile as sf

from models.clap_encoder import CLAP_Encoder
from models.dual_query_fusion import DualQueryFusion
from huggingface_hub import PyTorchModelHubMixin


# ============================================================
# FG / BG heads
# ============================================================
class WaveformHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv1d(1, 1, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)  # (B,1,T)


# ============================================================
# AudioSep Lightning Module (STEP 2)
# ============================================================
class AudioSep(pl.LightningModule, PyTorchModelHubMixin):

    def __init__(
        self,
        ss_model: nn.Module,
        waveform_mixer,
        query_encoder: Optional[nn.Module] = None,
        loss_function=None,
        learning_rate: float = 1e-4,
        lr_lambda_func=None,
        use_text_ratio: float = 1.0,
        sample_rate: int = 32000,
        debug_save_dir: str = "debug_audio",
        debug_every_n_steps: int = 1000,
        fusion_out_dim: int = 512,
        freeze_encoder: bool = True,
    ):
        super().__init__()

        # ---------------- Core ----------------
        self.ss_model = ss_model
        self.waveform_mixer = waveform_mixer
        self.query_encoder = query_encoder or CLAP_Encoder()
        self.loss_function = loss_function

        if learning_rate is None:
            # inference mode
            learning_rate = 0.0
        self.learning_rate = float(learning_rate)

        self.lr_lambda_func = lr_lambda_func
        self.use_text_ratio = float(use_text_ratio)

        self.sample_rate = int(sample_rate)
        self.debug_save_dir = debug_save_dir
        self.debug_every_n_steps = int(debug_every_n_steps)
        os.makedirs(self.debug_save_dir, exist_ok=True)

        # ---------------- Dual-query fusion ----------------
        self.fusion = DualQueryFusion(emb_dim=512, out_dim=fusion_out_dim)

        # ---------------- FG / BG heads ----------------
        self.fg_head = WaveformHead()
        self.bg_head = WaveformHead()

        # ---------------- Freeze policy (STEP 2) ----------------
        if freeze_encoder:
            for name, p in self.ss_model.named_parameters():
                if name.startswith(("encoder", "enc", "down")):
                    p.requires_grad = False

        # CLAP always frozen
        for p in self.query_encoder.parameters():
            p.requires_grad = False
        self.query_encoder.eval()

    # ========================================================
    # Helpers
    # ========================================================
    @staticmethod
    def _ensure_bt(x: torch.Tensor) -> torch.Tensor:
        return x.squeeze(1) if x.dim() == 3 else x

    @staticmethod
    def _ensure_b1t(x: torch.Tensor) -> torch.Tensor:
        return x.unsqueeze(1) if x.dim() == 2 else x

    @staticmethod
    def _sanitize_text_list(x: Union[List[Any], Any], B: int) -> List[str]:
        if isinstance(x, list):
            out = [(t if isinstance(t, str) else "") for t in x]
            return (out + [""] * B)[:B]
        if isinstance(x, str):
            return [x] * B
        return [""] * B

    @staticmethod
    def _route_from_prompt_type(pt: Optional[str]) -> str:
        if pt is None:
            return "FG"
        pt = str(pt).lower()
        if pt in ["positive", "contrastive"]:
            return "FG"
        if pt in ["negative", "negation"]:
            return "BG"
        return "FG"

    # ========================================================
    # Training step
    # ========================================================
    def training_step(self, batch: Dict[str, Any], batch_idx: int):

        random.seed(batch_idx)

        at = batch["audio_text"]
        audio_bt = self._ensure_bt(at["waveform"])  # (B,T)

        # ---------------- Mixer ----------------
        mix = self.waveform_mixer(
            waveforms=audio_bt,
            captions=at.get("text", None),
        )

        mixture = mix["mixture"]        # (B,T)
        fg_t = mix["fg"]                # (B,T)
        bg_t = mix["bg"]                # (B,T)
        prompt_type = mix["prompt_type"]

        B = mixture.size(0)

        # ---------------- Text ----------------
        text_pos = self._sanitize_text_list(mix.get("caption_pos"), B)
        text_neg = self._sanitize_text_list(mix.get("caption_neg"), B)

        # ---------------- CLAP ----------------
        e_pos = self.query_encoder.get_query_embed(
            modality="hybird",
            text=text_pos,
            audio=fg_t,
            use_text_ratio=self.use_text_ratio,
        )

        if any(t.strip() for t in text_neg):
            e_neg = self.query_encoder.get_query_embed(
                modality="hybird",
                text=text_neg,
                audio=fg_t,
                use_text_ratio=self.use_text_ratio,
            )
        else:
            e_neg = torch.zeros_like(e_pos)

        e_mix = self.fusion(e_pos, e_neg)

        # ---------------- Backbone ----------------
        out = self.ss_model({
            "mixture": mixture.unsqueeze(1),
            "condition": e_mix,
        })

        base = out["waveform"] if isinstance(out, dict) else out
        base = self._ensure_b1t(base)

        # ---------------- Heads ----------------
        y_fg = self.fg_head(base)
        y_bg = self.bg_head(base)

        # ---------------- Routing ----------------
        route_list: List[str] = []
        route_bg_mask = torch.zeros(B, dtype=torch.bool, device=mixture.device)

        for i in range(B):
            r = self._route_from_prompt_type(prompt_type[i])
            route_list.append(r)
            if r == "BG":
                route_bg_mask[i] = True

        # ---------------- Assemble pred / target ----------------
        pred = y_fg.clone()
        pred[route_bg_mask] = y_bg[route_bg_mask]

        target = self._ensure_b1t(fg_t).clone()
        target[route_bg_mask] = self._ensure_b1t(bg_t)[route_bg_mask]

        pred_bt = pred.squeeze(1)
        target_bt = target.squeeze(1)

        # ========================================================
        # DEBUG (EXACT REQUIREMENT)
        # ========================================================
        trainer_step = int(self.trainer.global_step)
        do_debug = (
            trainer_step == 0 or
            (
                self.debug_every_n_steps > 0 and
                trainer_step % self.debug_every_n_steps == 0
            )
        )

        if do_debug:
            os.makedirs(self.debug_save_dir, exist_ok=True)

            fg_indices = [i for i, r in enumerate(route_list) if r == "FG"]
            bg_indices = [i for i, r in enumerate(route_list) if r == "BG"]

            idx_fg = fg_indices[0] if fg_indices else None
            idx_bg = bg_indices[0] if bg_indices else None

            # ---------- FG ----------
            if idx_fg is not None:
                sf.write(
                    f"{self.debug_save_dir}/step{trainer_step}_FG_mixture.wav",
                    mixture[idx_fg].detach().cpu().numpy(),
                    self.sample_rate,
                )
                sf.write(
                    f"{self.debug_save_dir}/step{trainer_step}_FG_target.wav",
                    fg_t[idx_fg].detach().cpu().numpy(),
                    self.sample_rate,
                )
                sf.write(
                    f"{self.debug_save_dir}/step{trainer_step}_FG_non_target.wav",
                    bg_t[idx_fg].detach().cpu().numpy(),
                    self.sample_rate,
                )
                sf.write(
                    f"{self.debug_save_dir}/step{trainer_step}_pred_FG.wav",
                    y_fg[idx_fg].squeeze(0).detach().cpu().numpy(),
                    self.sample_rate,
                )

            # ---------- BG ----------
            if idx_bg is not None:
                sf.write(
                    f"{self.debug_save_dir}/step{trainer_step}_BG_mixture.wav",
                    mixture[idx_bg].detach().cpu().numpy(),
                    self.sample_rate,
                )
                sf.write(
                    f"{self.debug_save_dir}/step{trainer_step}_BG_target.wav",
                    bg_t[idx_bg].detach().cpu().numpy(),
                    self.sample_rate,
                )
                sf.write(
                    f"{self.debug_save_dir}/step{trainer_step}_BG_non_target.wav",
                    fg_t[idx_bg].detach().cpu().numpy(),
                    self.sample_rate,
                )
                sf.write(
                    f"{self.debug_save_dir}/step{trainer_step}_pred_BG.wav",
                    y_bg[idx_bg].squeeze(0).detach().cpu().numpy(),
                    self.sample_rate,
                )

            # ---------- ROUTE ----------
            with open(
                f"{self.debug_save_dir}/step{trainer_step}_route.txt",
                "w",
                encoding="utf-8",
            ) as f:
                f.write(f"global_step = {trainer_step}\n")
                f.write(f"batch_size  = {B}\n")
                f.write(f"prompt_type = {prompt_type}\n")
                f.write(f"route_list  = {route_list}\n")
                f.write(f"idx_fg      = {idx_fg}\n")
                f.write(f"idx_bg      = {idx_bg}\n")
                f.write(f"text_pos    = {text_pos}\n")
                f.write(f"text_neg    = {text_neg}\n")

        # ---------------- Loss ----------------
        loss = self.loss_function(
            {"segment": pred_bt, "fg": y_fg.squeeze(1), "bg": y_bg.squeeze(1)},
            {"segment": target_bt, "mixture": mixture},
        )

        self.log("train_loss", loss, prog_bar=True, on_step=True)
        self.log("route_bg_ratio", route_bg_mask.float().mean(), on_step=True)

        return loss

    # ========================================================
    # Optimizer
    # ========================================================
    def configure_optimizers(self):

        param_set = {}

        def add_params(ps):
            for p in ps:
                if p.requires_grad:
                    param_set[id(p)] = p

        add_params(self.fusion.parameters())
        add_params(self.fg_head.parameters())
        add_params(self.bg_head.parameters())
        add_params(self.ss_model.parameters())

        optimizer = optim.AdamW(list(param_set.values()), lr=self.learning_rate)

        if self.lr_lambda_func is None:
            return optimizer

        scheduler = LambdaLR(optimizer, self.lr_lambda_func)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
            },
        }

    # ========================================================
    # Non-strict checkpoint loading
    # ========================================================
    def load_state_dict(self, state_dict, strict: bool = True):
        return super().load_state_dict(state_dict, strict=False)

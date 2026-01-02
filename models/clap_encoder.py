import random
import torch
import torch.nn as nn
import torchaudio
from transformers import RobertaTokenizer

from models.CLAP.open_clip import create_model
from models.CLAP.training.data import get_audio_features


class CLAP_Encoder(nn.Module):
    """
    CLAP encoder สำหรับ AudioSep
    - ใน __init__ จะไม่โหลด pretrained checkpoint โดยตรง (เพื่อเลี่ยง PyTorch 2.6 pickle error)
    - ใช้ pretrained="" เพื่อกันไม่ให้ create_model() พังเพราะ pretrained=None
    - ใช้ load_audio_pretrained() โหลด weights หลังจาก safe_globals apply แล้ว
    """

    def __init__(
        self,
        pretrained_path='checkpoint/music_speech_audioset_epoch_15_esc_89.98.pt',
        sampling_rate=32000,
        amodel="HTSAT-base",
    ):
        super().__init__()

        self.device = "cpu"
        self.precision = "fp32"
        self.amodel = amodel  
        self.tmodel = "roberta"
        self.enable_fusion = False
        self.fusion_type = "aff_2d"

        self.pretrained_path = pretrained_path
        self.sampling_rate = sampling_rate

        # -------------------------------------------------------
        # ❗ FIX สำคัญ: ห้ามใช้ pretrained=None
        # create_model() มี pretrained.lower() → None พังทันที
        # -------------------------------------------------------
        self.model, self.model_cfg = create_model(
            amodel_name=self.amodel,
            tmodel_name=self.tmodel,
            pretrained="",              # ใช้ string ว่างแทน None
            precision=self.precision,
            device=self.device,
            enable_fusion=self.enable_fusion,
            fusion_type=self.fusion_type,
        )

        self.tokenizer = RobertaTokenizer.from_pretrained("roberta-base")

        # Freeze parameters
        for p in self.model.parameters():
            p.requires_grad = False

        self.model.eval()
        self.encoder_type = "CLAP"

    # -------------------------------------------------------
    # ฟังก์ชันโหลด pretrained CLAP (โหลดหลังจาก safe_globals apply แล้ว)
    # -------------------------------------------------------
    def load_audio_pretrained(self, ckpt_path=None):
        ckpt_path = ckpt_path or self.pretrained_path

        print(f"🔄 Loading CLAP pretrained weights: {ckpt_path}")
        try:
            ckpt = torch.load(ckpt_path, map_location="cpu")
            state_dict = ckpt.get("model", ckpt)
            missing, unexpected = self.model.load_state_dict(state_dict, strict=False)
            print("   ✔ Loaded CLAP weights")
            print("   Missing keys:", missing)
            print("   Unexpected keys:", unexpected)
        except Exception as e:
            print(f"   ⚠️ Failed to load CLAP pretrained weights: {e}")
            print("   จะใช้ random weights แทน (คุณภาพจะด้อยลง)")

    # -------------------------------------------------------
    # Utility
    # -------------------------------------------------------

    def batch_to_list(self, batch):
        return [batch[i] for i in range(batch.size(0))]

    # -------------------------------------------------------
    # Audio embedding
    # -------------------------------------------------------
    def _get_audio_embed(self, batch):
        # batch: [B, samples]
        with torch.no_grad():
            assert self.sampling_rate == 32000, "Only support 32000Hz input"

            # Resample to 48k for HTSAT
            batch = torchaudio.functional.resample(
                batch,
                orig_freq=self.sampling_rate,
                new_freq=48000,
            )

            audio_dicts = []
            for waveform in self.batch_to_list(batch):
                audio_dict = {}
                audio_dict = get_audio_features(
                    audio_dict,
                    waveform,
                    480000,  # target len for HTSAT
                    data_truncating="fusion",
                    data_filling="repeatpad",
                    audio_cfg=self.model_cfg["audio_cfg"],
                )
                audio_dicts.append(audio_dict)

            embed = self.model.get_audio_embedding(audio_dicts)
            return embed.detach()

    # -------------------------------------------------------
    # Text embedding
    # -------------------------------------------------------
    def _get_text_embed(self, batch):
        double_batch = False

        if len(batch) == 1:
            batch = batch * 2
            double_batch = True

        with torch.no_grad():
            text_data = self.tokenizer(
                batch,
                padding="max_length",
                truncation=True,
                max_length=512,
                return_tensors="pt",
            )
            # convert batch dims so model.get_text_embedding() ใช้ได้
            text_data = {k: v for k, v in text_data.items()}
            embed = self.model.get_text_embedding(text_data)

        if double_batch:
            embed = embed[0].unsqueeze(0)

        return embed.detach()

    # -------------------------------------------------------
    # Public API: get query embedding
    # -------------------------------------------------------
    def get_query_embed(self, modality, audio=None, text=None, use_text_ratio=0.5, device=None):
        if modality == "audio":
            embed = self._get_audio_embed(audio)
        elif modality == "text":
            embed = self._get_text_embed(text)
        elif modality == "hybird":
            if random.random() > use_text_ratio:
                embed = self._get_audio_embed(audio)
            else:
                embed = self._get_text_embed(text)
        else:
            raise NotImplementedError("Unknown modality type")

        return embed.float()

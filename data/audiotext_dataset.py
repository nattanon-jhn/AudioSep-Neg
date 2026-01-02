import json
import os
import random
from typing import Any, Dict, List, Optional, Tuple

import torch
import torchaudio
import soundfile as sf
from torch.utils.data import Dataset


class AudioTextDataset(Dataset):
    """
    AudioTextDataset (Dual-query ready)

    รองรับ schema ใน json:
      - caption (legacy)
      - caption_pos / caption_neg / prompt_type (AudioSep-Neg)
      - text (fallback)

    Output (per item):
      - waveform: (1, T) float32
      - text: str              (alias of caption_pos)
      - caption_pos: str
      - caption_neg: str       ("" if none)
      - prompt_type: str       ("" if none)
      - modality: "audio_text"
    """

    def __init__(
        self,
        datafiles: List[str],
        sampling_rate: int = 32000,
        max_clip_len: int = 5,
    ):
        all_data_json: List[Dict[str, Any]] = []
        for datafile in datafiles:
            with open(datafile, "r") as fp:
                data_json = json.load(fp)["data"]
                all_data_json.extend(data_json)

        self.all_data_json = all_data_json
        self.sampling_rate = sampling_rate
        self.max_length = max_clip_len * sampling_rate

    def __len__(self):
        return len(self.all_data_json)

    # --------------------------------------------------
    # Audio utils
    # --------------------------------------------------
    def _cut_or_randomcrop(self, waveform: torch.Tensor) -> torch.Tensor:
        """
        waveform: (1, T)
        return  : (1, max_length)
        """
        if waveform.size(1) > self.max_length:
            start = random.randint(0, waveform.size(1) - self.max_length)
            waveform = waveform[:, start: start + self.max_length]
        else:
            padded = torch.zeros(1, self.max_length)
            padded[:, : waveform.size(1)] = waveform
            waveform = padded
        return waveform

    def _read_audio(self, index: int) -> Tuple[Dict[str, Any], torch.Tensor, int]:
        """
        Returns:
          item: json item
          audio: (C, T) float32
          sr: int
        """
        try:
            item = self.all_data_json[index]
            audio_path = item["wav"]

            audio_np, sr = sf.read(audio_path, always_2d=True)
            audio = torch.from_numpy(audio_np.T).float()  # (C, T)

            if audio.size(1) < int(self.sampling_rate * 1.0):
                raise RuntimeError(f"{audio_path} too short")

            return item, audio, sr

        except Exception as e:
            print(f"[Dataset] error loading index={index}: {e}")
            new_index = random.randint(0, len(self.all_data_json) - 1)
            return self._read_audio(new_index)

    @staticmethod
    def _ensure_str(x: Any, fallback: str = "unknown") -> str:
        if x is None:
            return fallback
        if isinstance(x, str):
            s = x.strip()
            return s if len(s) > 0 else fallback
        # ถ้าเป็น list/อย่างอื่น ให้แปลงเป็น string
        s = str(x).strip()
        return s if len(s) > 0 else fallback

    # --------------------------------------------------
    # Main
    # --------------------------------------------------
    def __getitem__(self, index: int) -> Dict[str, Any]:
        item, audio, sr = self._read_audio(index)

        # ---------- mono ----------
        if audio.size(0) > 1:
            audio = audio.mean(dim=0, keepdim=True)  # (1, T)

        # ---------- resample ----------
        if sr != self.sampling_rate:
            audio = torchaudio.functional.resample(
                audio,
                orig_freq=sr,
                new_freq=self.sampling_rate,
            )

        # ---------- crop / pad ----------
        audio = self._cut_or_randomcrop(audio)  # (1, max_length)

        # ---------- captions ----------
        # priority: caption_pos -> caption -> text -> filename stem
        file_stem = os.path.splitext(os.path.basename(item.get("wav", "unknown.wav")))[0]
        caption_pos_raw = item.get("caption_pos") or item.get("caption") or item.get("text")
        caption_pos = self._ensure_str(caption_pos_raw, fallback=file_stem)

        caption_neg_raw = item.get("caption_neg", "")
        caption_neg = self._ensure_str(caption_neg_raw, fallback="")  # allow empty string

        prompt_type_raw = item.get("prompt_type", "")
        prompt_type = self._ensure_str(prompt_type_raw, fallback="")  # allow empty string

        return {
            "waveform": audio,             # (1, T)
            "text": caption_pos,           # ✅ alias for older code
            "caption_pos": caption_pos,    # str (never None)
            "caption_neg": caption_neg,    # str ("" if none)
            "prompt_type": prompt_type,    # str ("" if none)
            "modality": "audio_text",
        }

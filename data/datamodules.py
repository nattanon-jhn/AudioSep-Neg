from typing import Dict, List, Optional, NoReturn, Any

import torch
import lightning.pytorch as pl
from torch.utils.data import DataLoader


class DataModule(pl.LightningDataModule):
    def __init__(
        self,
        train_dataset: object,
        batch_size: int,
        num_workers: int,
    ):
        super().__init__()
        self._train_dataset = train_dataset
        self.num_workers = num_workers
        self.batch_size = batch_size
        self.collate_fn = collate_fn

    def prepare_data(self):
        pass

    def setup(self, stage: Optional[str] = None) -> NoReturn:
        self.train_dataset = self._train_dataset

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            dataset=self.train_dataset,
            batch_size=self.batch_size,
            collate_fn=self.collate_fn,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=False,
            shuffle=True,
        )

    def val_dataloader(self):
        pass

    def test_dataloader(self):
        pass

    def teardown(self):
        pass


def collate_fn(list_data_dict: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Collate audio-text batch.

    Expect per-item keys (from AudioTextDataset):
      waveform: (1, T)
      text, caption_pos, caption_neg, prompt_type: str
      modality: "audio_text"

    Return:
      {
        "audio_text": {
          "waveform": (B,1,T),
          "text": [str...],
          "caption_pos": [str...],
          "caption_neg": [str...],
          "prompt_type": [str...],
          "modality": ["audio_text"...]
        }
      }
    """
    at_list = [d for d in list_data_dict if d.get("modality") == "audio_text"]
    if len(at_list) == 0:
        return {}

    batch: Dict[str, Any] = {}
    keys = at_list[0].keys()

    for key in keys:
        values = [d[key] for d in at_list]

        if isinstance(values[0], torch.Tensor):
            batch[key] = torch.stack(values, dim=0)  # waveform -> (B,1,T)
        else:
            # metadata -> list
            batch[key] = values

    return {"audio_text": batch}

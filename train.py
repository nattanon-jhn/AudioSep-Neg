# train_step2.py
import argparse
import logging
import os
import pathlib
from typing import List, NoReturn, Optional

import torch
import lightning.pytorch as pl
from lightning.pytorch.loggers import TensorBoardLogger

from data.datamodules import DataModule
from data.audiotext_dataset import AudioTextDataset
from utils import create_logging, parse_yaml, get_model_class

from models.audiosep import AudioSep
from data.waveform_mixers import SegmentMixer
from models.clap_encoder import CLAP_Encoder

from losses import get_loss_function
from optimizers.lr_schedulers import get_lr_lambda


# ============================================================
# Trainable params summary
# ============================================================
def print_trainable_summary(model):
    groups = {}
    total = 0
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        key = name.split(".")[0]
        groups[key] = groups.get(key, 0) + p.numel()
        total += p.numel()

    print("\n===== TRAINABLE PARAMETER SUMMARY (STEP 2) =====")
    for k, v in sorted(groups.items(), key=lambda x: -x[1]):
        print(f"{k:<25}: {v:,}")
    print(f"{'TOTAL':<25}: {total:,}\n")


# ============================================================
# Checkpoint callback
# ============================================================
class CheckpointEveryNSteps(pl.Callback):
    def __init__(self, checkpoints_dir: str, save_step_frequency: int):
        super().__init__()
        self.checkpoints_dir = checkpoints_dir
        self.save_step_frequency = int(save_step_frequency)

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        step = trainer.global_step
        if step > 0 and step % self.save_step_frequency == 0:
            os.makedirs(self.checkpoints_dir, exist_ok=True)
            ckpt_path = os.path.join(self.checkpoints_dir, f"step={step}.ckpt")
            trainer.save_checkpoint(ckpt_path)
            print(f"[CKPT] Saved: {ckpt_path}")


# ============================================================
# Directories
# ============================================================
def get_dirs(workspace, filename, config_yaml, devices_num):
    os.makedirs(workspace, exist_ok=True)
    yaml_name = pathlib.Path(config_yaml).stem

    checkpoints_dir = os.path.join(
        workspace, "checkpoints", filename, f"{yaml_name},devices={devices_num}"
    )
    logs_dir = os.path.join(
        workspace, "logs", filename, f"{yaml_name},devices={devices_num}"
    )
    tf_logs_dir = os.path.join(
        workspace, "tf_logs", filename, f"{yaml_name},devices={devices_num}"
    )

    os.makedirs(checkpoints_dir, exist_ok=True)
    os.makedirs(logs_dir, exist_ok=True)
    os.makedirs(tf_logs_dir, exist_ok=True)

    create_logging(logs_dir, filemode="w")
    logging.info(f"Workspace: {workspace}")
    logging.info(f"Config: {config_yaml}")

    return checkpoints_dir, logs_dir, tf_logs_dir


# ============================================================
# DataModule
# ============================================================
def get_data_module(config_yaml, num_workers, batch_size):
    configs = parse_yaml(config_yaml)

    dataset = AudioTextDataset(
        datafiles=configs["data"]["datafiles"],
        sampling_rate=configs["data"]["sampling_rate"],
        max_clip_len=configs["data"]["segment_seconds"],
    )

    return DataModule(
        train_dataset=dataset,
        batch_size=batch_size,
        num_workers=num_workers,
    )


# ============================================================
# Load weights only
# ============================================================
def load_weights_only(pl_model, ckpt_path):
    print(f"[INFO] Loading Step1 weights from: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location="cpu")
    state_dict = ckpt["state_dict"] if isinstance(ckpt, dict) and "state_dict" in ckpt else ckpt
    # AudioSep.load_state_dict forces strict=False
    pl_model.load_state_dict(state_dict)


# ============================================================
# Train (STEP 2)
# ============================================================
def train(args) -> NoReturn:

    pl.seed_everything(42, workers=True)

    workspace = args.workspace
    config_yaml = args.config_yaml
    filename = args.filename

    devices_num = torch.cuda.device_count()
    configs = parse_yaml(config_yaml)

    # ---------------- Data ----------------
    max_mix_num = configs["data"]["max_mix_num"]
    lower_db = configs["data"]["loudness_norm"]["lower_db"]
    higher_db = configs["data"]["loudness_norm"]["higher_db"]

    # ---------------- Model ----------------
    model_type = configs["model"]["model_type"]
    input_channels = configs["model"]["input_channels"]
    output_channels = configs["model"]["output_channels"]
    condition_size = configs["model"]["condition_size"]
    use_text_ratio = configs["model"]["use_text_ratio"]

    # ---------------- Train ----------------
    batch_size = configs["train"]["batch_size_per_device"]
    num_workers = configs["train"]["num_workers"]
    num_nodes = configs["train"]["num_nodes"]
    sync_batchnorm = configs["train"]["sync_batchnorm"]

    loss_type = configs["train"]["loss_type"]
    learning_rate = float(configs["train"]["optimizer"]["learning_rate"])
    lr_lambda_type = configs["train"]["optimizer"]["lr_lambda_type"]
    warm_up_steps = configs["train"]["optimizer"]["warm_up_steps"]
    reduce_lr_steps = configs["train"]["optimizer"]["reduce_lr_steps"]
    save_step_frequency = configs["train"]["save_step_frequency"]

    resume_checkpoint_path = args.resume_checkpoint_path.strip() if args.resume_checkpoint_path else ""

    # ---------------- Dirs ----------------
    checkpoints_dir, logs_dir, tf_logs_dir = get_dirs(
        workspace, filename, config_yaml, devices_num
    )

    # ---------------- TensorBoard ----------------
    tb_logger = TensorBoardLogger(
        save_dir=tf_logs_dir,
        name="",
        version=None,
    )

    # ---------------- DataModule ----------------
    data_module = get_data_module(
        config_yaml=config_yaml,
        batch_size=batch_size,
        num_workers=num_workers,
    )

    # ---------------- Backbone ----------------
    Model = get_model_class(model_type)
    ss_model = Model(
        input_channels=input_channels,
        output_channels=output_channels,
        condition_size=condition_size,
    )

    # ---------------- Loss ----------------
    loss_function = get_loss_function(loss_type)

    # ---------------- Mixer ----------------
    segment_mixer = SegmentMixer(
        max_mix_num=max_mix_num,
        lower_db=lower_db,
        higher_db=higher_db,
    )

    # ---------------- CLAP ----------------
    query_encoder = CLAP_Encoder()

    # ---------------- LR scheduler ----------------
    lr_lambda_func = get_lr_lambda(
        lr_lambda_type=lr_lambda_type,
        warm_up_steps=warm_up_steps,
        reduce_lr_steps=reduce_lr_steps,
    )

    # ---------------- Lightning model (IMPORTANT: match AudioSep Step2 signature) ----------------
    pl_model = AudioSep(
        ss_model=ss_model,
        waveform_mixer=segment_mixer,
        query_encoder=query_encoder,
        loss_function=loss_function,
        learning_rate=learning_rate,
        lr_lambda_func=lr_lambda_func,
        use_text_ratio=use_text_ratio,
        freeze_encoder=True,          # ✅ Step2: freeze encoder blocks in ss_model
        debug_every_n_steps=1000,     # ปรับตามต้องการ
    )

    # (optional) extra freeze by name if your backbone uses "encoder" in name
    for name, p in pl_model.ss_model.named_parameters():
        if "encoder" in name:
            p.requires_grad = False

    # Ensure CLAP frozen (redundant-safe)
    for p in pl_model.query_encoder.parameters():
        p.requires_grad = False
    pl_model.query_encoder.eval()

    print_trainable_summary(pl_model)

    # ---------------- Load Step1 weights ----------------
    if resume_checkpoint_path:
        load_weights_only(pl_model, resume_checkpoint_path)

    # ---------------- Trainer ----------------
    trainer = pl.Trainer(
        accelerator="auto",
        devices="auto",
        num_nodes=num_nodes,
        precision="32-true",
        logger=tb_logger,
        callbacks=[
            CheckpointEveryNSteps(
                checkpoints_dir=checkpoints_dir,
                save_step_frequency=save_step_frequency,
            )
        ],
        max_epochs=-1,
        log_every_n_steps=50,
        sync_batchnorm=sync_batchnorm,
        enable_checkpointing=False,
        enable_progress_bar=True,
        enable_model_summary=True,
        num_sanity_val_steps=0,
    )

    trainer.fit(pl_model, datamodule=data_module, ckpt_path=None)


# ============================================================
# Main
# ============================================================
if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--workspace", type=str, required=True)
    parser.add_argument("--config_yaml", type=str, required=True)
    parser.add_argument(
        "--resume_checkpoint_path",
        type=str,
        required=True,
        help="Step1 checkpoint path (weights-only load)",
    )

    args = parser.parse_args()
    args.filename = pathlib.Path(__file__).stem

    train(args)

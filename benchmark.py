
Folder highlights
Evaluation data from an AudioSep-Neg project includes results for positive, negative, contrastive, and negation audio separation tasks.

# benchmark.py  (ROOT: AudioSep-Neg_Final/benchmark.py)

import sys
import json
import argparse
from pathlib import Path

import torch

# -----------------------------
# Make Python see: ./evaluation/evaluate_esc50.py
# -----------------------------
ROOT = Path(__file__).resolve().parent
EVAL_DIR = ROOT / "evaluation"
sys.path.insert(0, str(EVAL_DIR))  # <-- key fix

# now this works because evaluate_esc50.py is inside ./evaluation/
from evaluate_esc50 import (
    ESC50ValEvaluator,
    _resolve_audiosep_neg_root,
    _ensure_import_audiosep_neg,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint_path", type=str, required=True)

    # optional overrides
    parser.add_argument("--audiosep_neg_root", type=str, default=None)

    # config for audiosep-neg-final
    parser.add_argument("--config_yaml", type=str, default=None)
    parser.add_argument("--clap_ckpt_path", type=str, default=None)

    # your validation set root (ESC50_val)
    parser.add_argument("--val_root", type=str, default=None)
    parser.add_argument("--csv_glob", type=str, default="val_*.csv")

    # debug / speed
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--device", type=str, default=None)

    # logging
    parser.add_argument("--save_json", type=str, default=None)

    args = parser.parse_args()

    device = (
        torch.device(args.device)
        if args.device
        else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    )

    project_root = ROOT  # AudioSep-Neg_Final/
    audiosep_neg_root = _resolve_audiosep_neg_root(args.audiosep_neg_root)
    _ensure_import_audiosep_neg(str(audiosep_neg_root))

    # import build_audiosep from audiosep-neg-final
    from pipeline import build_audiosep  # type: ignore

    # defaults (match audiosep-neg-final repo layout)
    if args.config_yaml is None:
        args.config_yaml = str(audiosep_neg_root / "config" / "audiosep_base_step2.yaml")
    if args.clap_ckpt_path is None:
        args.clap_ckpt_path = str(audiosep_neg_root / "checkpoint" / "music_speech_audioset_epoch_15_esc_89.98.pt")
    if args.val_root is None:
        # match your generation path: <PROJECT_ROOT>/evaluation/data/ESC50_val
        args.val_root = str(project_root / "evaluation" / "data" / "ESC50_val")

    print("========== SETTINGS ==========")
    print(f"device           : {device}")
    print(f"audiosep_neg_root: {audiosep_neg_root}")
    print(f"config_yaml      : {args.config_yaml}")
    print(f"clap_ckpt_path   : {args.clap_ckpt_path}")
    print(f"checkpoint_path  : {args.checkpoint_path}")
    print(f"val_root         : {args.val_root}")
    print(f"csv_glob         : {args.csv_glob}")
    print(f"limit            : {args.limit}")
    print("================================\n")

    # 1) load model
    model = build_audiosep(
        config_yaml=args.config_yaml,
        checkpoint_path=args.checkpoint_path,
        device=device,
        clap_ckpt_path=args.clap_ckpt_path,
    )

    # 2) evaluate
    evaluator = ESC50ValEvaluator(
        val_root=args.val_root,
        audiosep_neg_root=str(audiosep_neg_root),
        csv_glob=args.csv_glob,
        sr=32000,
        mono=True,
    )

    result = evaluator(model=model, device=device, limit=args.limit, verbose=True)

    print("\n======= ESC50_val (Validation) =======")
    print(f"MEAN SDR   : {result.mean_sdr:.4f}")
    print(f"MEAN SDRi  : {result.mean_sdri:.4f}")
    print(f"MEAN SI-SDR: {result.mean_sisdr:.4f}")
    print("----- per prompt_type -----")
    for pt, d in sorted(result.per_type.items(), key=lambda x: x[0]):
        print(
            f"{pt:12s} | n={d['n']:4d} | "
            f"SDR={d['mean_sdr']:.4f} | SDRi={d['mean_sdri']:.4f} | SI-SDR={d['mean_sisdr']:.4f}"
        )
    print("======================================\n")

    if args.save_json:
        out = {
            "mean_sdr": result.mean_sdr,
            "mean_sdri": result.mean_sdri,
            "mean_sisdr": result.mean_sisdr,
            "per_type": result.per_type,
            "settings": vars(args),
        }
        out_path = Path(args.save_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"[OK] saved json: {out_path}")


if __name__ == "__main__":
    main()

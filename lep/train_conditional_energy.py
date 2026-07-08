"""CLI for training observation-conditioned latent energy priors."""

from __future__ import annotations

import argparse
import json
from typing import Optional, Sequence

import torch

from .conditional_energy import train_conditional_energy
from .train_deepsdf import load_yaml


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="Train a conditional latent energy prior.")
    parser.add_argument("--config", default=None)
    parser.add_argument("--checkpoint", required=True, help="Frozen DeepSDF checkpoint with train latents.")
    parser.add_argument("--sdf-root", required=True, help="SDF sample root for train shape observations.")
    parser.add_argument("--out", required=True)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--observation-points", type=int, nargs="+", default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--point-hidden-size", type=int, default=None)
    parser.add_argument("--context-dim", type=int, default=None)
    parser.add_argument("--hidden-dim", type=int, default=None)
    parser.add_argument("--langevin-steps", type=int, default=None)
    args = parser.parse_args(argv)

    config = load_yaml(args.config)
    overrides = {
        "batch_size": args.batch_size,
        "observation_points": args.observation_points,
        "lr": args.lr,
        "point_hidden_size": args.point_hidden_size,
        "context_dim": args.context_dim,
        "hidden_dim": args.hidden_dim,
        "langevin_steps": args.langevin_steps,
    }
    resolved_overrides = {key: value for key, value in overrides.items() if value is not None}
    if resolved_overrides:
        config = dict(config)
        config["conditional_energy"] = {
            **dict(config.get("conditional_energy", {})),
            **resolved_overrides,
        }

    checkpoint_path = train_conditional_energy(
        config=config,
        checkpoint=args.checkpoint,
        sdf_root=args.sdf_root,
        out_dir=args.out,
        device=args.device,
        epochs_override=args.epochs,
    )
    print(json.dumps({"checkpoint": str(checkpoint_path)}, indent=2))


if __name__ == "__main__":
    main()

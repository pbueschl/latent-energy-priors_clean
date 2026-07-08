"""Train a compact DeepSDF autodecoder from per-shape `.npz` SDF samples."""

from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from .deepsdf import DeepSDFAutodecoder, LatentDecoder, clamp_sdf
from .shapenet import load_split


LATENT_REGULARIZERS = {"l2", "none", "norm_hinge"}
TRAIN_HISTORY_FIELDNAMES = ["epoch", "loss", "sdf_l1", "latent_l2", "latent_reg", "latent_norm"]


def _torch_load(path: Path, map_location: Optional[str] = None) -> object:
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


def load_yaml(path: Optional[str]) -> Dict[str, object]:
    if path is None:
        return {}
    with Path(path).open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Config must be a mapping: {path}")
    return data


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def find_sdf_file(sdf_root: str, shape_id: str) -> Path:
    root = Path(sdf_root).expanduser()
    candidates = [
        root / f"{shape_id}.npz",
        root / shape_id / "samples.npz",
        root / shape_id / f"{shape_id}.npz",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    matches = sorted(root.rglob(f"{shape_id}.npz"))
    if matches:
        return matches[0]
    raise FileNotFoundError(f"No SDF .npz found for shape id {shape_id!r} under {root}")


def load_sdf_npz(path: str) -> Tuple[np.ndarray, np.ndarray]:
    with np.load(path) as data:
        point_key = "points" if "points" in data else "xyz" if "xyz" in data else None
        sdf_key = "sdf" if "sdf" in data else "sdf_values" if "sdf_values" in data else None
        if point_key is None or sdf_key is None:
            raise KeyError(f"{path} must contain points/xyz and sdf/sdf_values arrays")
        points = np.asarray(data[point_key], dtype=np.float32)
        sdf = np.asarray(data[sdf_key], dtype=np.float32).reshape(-1)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError(f"Expected points [N, 3] in {path}, got {points.shape}")
    if points.shape[0] != sdf.shape[0]:
        raise ValueError(f"Point/SDF count mismatch in {path}: {points.shape[0]} vs {sdf.shape[0]}")
    return points, sdf


def _sample_arrays(
    points: np.ndarray,
    sdf: np.ndarray,
    count: int,
    rng: np.random.Generator,
) -> Tuple[np.ndarray, np.ndarray]:
    replace = count > points.shape[0]
    indices = rng.choice(points.shape[0], size=count, replace=replace)
    return points[indices], sdf[indices]


class SDFShapeDataset(Dataset):
    """One sampled SDF cloud per shape per epoch, without worker-side caches."""

    def __init__(
        self,
        shape_ids: Sequence[str],
        sdf_paths: Dict[str, Path],
        samples_per_shape: int,
        seed: int,
    ) -> None:
        self.shape_ids = list(shape_ids)
        self.sdf_paths = {shape_id: str(sdf_paths[shape_id]) for shape_id in self.shape_ids}
        self.samples_per_shape = int(samples_per_shape)
        self.seed = int(seed)
        self._epoch = torch.zeros((), dtype=torch.long).share_memory_()

    def __len__(self) -> int:
        return len(self.shape_ids)

    def set_epoch(self, epoch: int) -> None:
        self._epoch.fill_(int(epoch))

    def __getitem__(self, shape_index: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        shape_index = int(shape_index)
        shape_id = self.shape_ids[shape_index]
        epoch = int(self._epoch.item())
        rng = np.random.default_rng(self.seed + epoch * 1_000_003 + shape_index)
        points, sdf = load_sdf_npz(self.sdf_paths[shape_id])
        sampled_points, sampled_sdf = _sample_arrays(points, sdf, self.samples_per_shape, rng)
        return (
            torch.tensor(shape_index, dtype=torch.long),
            torch.from_numpy(sampled_points.astype(np.float32, copy=False)),
            torch.from_numpy(sampled_sdf.astype(np.float32, copy=False)),
        )


def _seed_worker(worker_id: int) -> None:
    worker_seed = (torch.initial_seed() + worker_id) % 2**32
    random.seed(worker_seed)
    np.random.seed(worker_seed)


def _decoder_from_cfg(cfg: Dict[str, object]) -> LatentDecoder:
    return LatentDecoder(
        latent_dim=int(cfg.get("latent_dim", 128)),
        hidden_dim=int(cfg.get("hidden_dim", 256)),
        num_layers=int(cfg.get("num_layers", 8)),
        skip_layers=cfg.get("skip_layers", ()),
        activation=str(cfg.get("activation", "relu")),
    )


def _resolve_latent_training_options(cfg: Dict[str, object]) -> Dict[str, object]:
    latent_l2_weight = float(cfg.get("latent_l2_weight", 1e-4))
    latent_regularizer = str(cfg.get("latent_regularizer", "l2")).strip().lower()
    if latent_regularizer not in LATENT_REGULARIZERS:
        allowed = ", ".join(sorted(LATENT_REGULARIZERS))
        raise ValueError(f"Unsupported latent_regularizer {latent_regularizer!r}; expected one of {allowed}")

    latent_hinge_radius = float(cfg.get("latent_hinge_radius", 1.0))
    if latent_hinge_radius < 0.0:
        raise ValueError("latent_hinge_radius must be non-negative")
    latent_hinge_weight = float(cfg.get("latent_hinge_weight", latent_l2_weight))

    latent_noise_std = float(cfg.get("latent_noise_std", 0.0))
    if latent_noise_std < 0.0:
        raise ValueError("latent_noise_std must be non-negative")

    latent_dropout_p = float(cfg.get("latent_dropout_p", 0.0))
    if not 0.0 <= latent_dropout_p < 1.0:
        raise ValueError("latent_dropout_p must be in [0, 1)")

    cfg["latent_l2_weight"] = latent_l2_weight
    cfg["latent_regularizer"] = latent_regularizer
    cfg["latent_hinge_radius"] = latent_hinge_radius
    if "latent_hinge_weight" in cfg or latent_regularizer == "norm_hinge":
        cfg["latent_hinge_weight"] = latent_hinge_weight
    cfg["latent_noise_std"] = latent_noise_std
    cfg["latent_dropout_p"] = latent_dropout_p
    return {
        "latent_l2_weight": latent_l2_weight,
        "latent_regularizer": latent_regularizer,
        "latent_hinge_radius": latent_hinge_radius,
        "latent_hinge_weight": latent_hinge_weight,
        "latent_noise_std": latent_noise_std,
        "latent_dropout_p": latent_dropout_p,
    }


def _latent_regularization_terms(
    z: torch.Tensor,
    latent_regularizer: str,
    latent_l2_weight: float,
    latent_hinge_radius: float,
    latent_hinge_weight: float,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    latent_norms = z.norm(dim=-1)
    latent_norm = latent_norms.mean()
    latent_l2 = z.pow(2).sum(dim=-1).mean()
    if latent_regularizer == "l2":
        latent_reg = latent_l2
        weighted_reg = float(latent_l2_weight) * latent_reg
    elif latent_regularizer == "none":
        latent_reg = z.new_zeros(())
        weighted_reg = latent_reg
    elif latent_regularizer == "norm_hinge":
        latent_reg = (latent_norms - float(latent_hinge_radius)).clamp_min(0.0).pow(2).mean()
        weighted_reg = float(latent_hinge_weight) * latent_reg
    else:
        allowed = ", ".join(sorted(LATENT_REGULARIZERS))
        raise ValueError(f"Unsupported latent_regularizer {latent_regularizer!r}; expected one of {allowed}")
    return weighted_reg, latent_reg, latent_l2, latent_norm


def _perturb_latents_for_training(
    z: torch.Tensor,
    latent_noise_std: float,
    latent_dropout_p: float,
) -> torch.Tensor:
    z_for_decoder = z
    if latent_dropout_p > 0.0:
        z_for_decoder = F.dropout(z_for_decoder, p=float(latent_dropout_p), training=True)
    if latent_noise_std > 0.0:
        z_for_decoder = z_for_decoder + torch.randn_like(z_for_decoder) * float(latent_noise_std)
    return z_for_decoder


def _history_fieldnames(path: Path) -> List[str]:
    if path.exists() and path.stat().st_size > 0:
        with path.open("r", newline="", encoding="utf-8") as f:
            reader = csv.reader(f)
            try:
                header = next(reader)
            except StopIteration:
                header = []
        if header:
            return header
    return list(TRAIN_HISTORY_FIELDNAMES)


def _train_batch(
    model: DeepSDFAutodecoder,
    optimizer: torch.optim.Optimizer,
    shape_indices: torch.Tensor,
    xyz: torch.Tensor,
    target: torch.Tensor,
    sdf_band: Optional[float],
    latent_regularizer: str,
    latent_l2_weight: float,
    latent_hinge_radius: float,
    latent_hinge_weight: float,
    latent_noise_std: float,
    latent_dropout_p: float,
) -> Tuple[float, float, float, float, float]:
    optimizer.zero_grad(set_to_none=True)
    z = model.latent_codes(shape_indices.long())
    z_for_decoder = _perturb_latents_for_training(z, latent_noise_std, latent_dropout_p)
    pred = model.decoder(z_for_decoder, xyz)
    sdf_l1 = F.l1_loss(clamp_sdf(pred, sdf_band), clamp_sdf(target, sdf_band))
    weighted_reg, latent_reg, latent_l2, latent_norm = _latent_regularization_terms(
        z,
        latent_regularizer=latent_regularizer,
        latent_l2_weight=latent_l2_weight,
        latent_hinge_radius=latent_hinge_radius,
        latent_hinge_weight=latent_hinge_weight,
    )
    loss = sdf_l1 + weighted_reg
    loss.backward()
    optimizer.step()
    return (
        float(loss.detach().cpu()),
        float(sdf_l1.detach().cpu()),
        float(latent_l2.detach().cpu()),
        float(latent_reg.detach().cpu()),
        float(latent_norm.detach().cpu()),
    )


def _manual_epoch_batches(
    train_shape_ids: Sequence[str],
    sdf_paths: Dict[str, Path],
    cache: Dict[str, Tuple[np.ndarray, np.ndarray]],
    rng: np.random.Generator,
    batch_shapes: int,
    samples_per_shape: int,
) -> Iterator[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    order = rng.permutation(len(train_shape_ids))
    for start in range(0, len(order), batch_shapes):
        batch_indices_np = order[start : start + batch_shapes]
        batch_points: List[np.ndarray] = []
        batch_sdf: List[np.ndarray] = []
        for shape_index in batch_indices_np:
            shape_id = train_shape_ids[int(shape_index)]
            if shape_id not in cache:
                cache[shape_id] = load_sdf_npz(str(sdf_paths[shape_id]))
            points, sdf = cache[shape_id]
            sampled_points, sampled_sdf = _sample_arrays(points, sdf, samples_per_shape, rng)
            batch_points.append(sampled_points)
            batch_sdf.append(sampled_sdf)
        yield batch_indices_np, np.stack(batch_points), np.stack(batch_sdf)


def _make_dataloader(
    train_shape_ids: Sequence[str],
    sdf_paths: Dict[str, Path],
    samples_per_shape: int,
    batch_shapes: int,
    num_workers: int,
    seed: int,
    pin_memory: bool,
) -> Tuple[SDFShapeDataset, DataLoader]:
    dataset = SDFShapeDataset(
        train_shape_ids,
        sdf_paths=sdf_paths,
        samples_per_shape=samples_per_shape,
        seed=seed,
    )
    generator = torch.Generator()
    generator.manual_seed(seed)
    loader = DataLoader(
        dataset,
        batch_size=int(batch_shapes),
        shuffle=True,
        num_workers=int(num_workers),
        pin_memory=pin_memory,
        persistent_workers=int(num_workers) > 0,
        worker_init_fn=_seed_worker,
        generator=generator,
    )
    return dataset, loader


def _save_checkpoint(
    path: Path,
    epoch: int,
    model: DeepSDFAutodecoder,
    optimizer: torch.optim.Optimizer,
    train_shape_ids: Sequence[str],
    config: Dict[str, object],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "decoder_state_dict": model.decoder.state_dict(),
            "decoder_config": model.decoder.config(),
            "latent_codes_state_dict": model.latent_codes.state_dict(),
            "train_latents": model.export_latents(),
            "train_shape_ids": list(train_shape_ids),
            "optimizer_state_dict": optimizer.state_dict(),
            "config": config,
        },
        path,
    )


def train_deepsdf(
    config: Dict[str, object],
    sdf_root: str,
    split_file: str,
    out_dir: str,
    device: str = "cpu",
    resume: Optional[str] = None,
    epochs_override: Optional[int] = None,
    num_workers_override: Optional[int] = None,
) -> Path:
    seed = int(config.get("seed", 13))
    set_seed(seed)
    split = load_split(split_file)
    train_shape_ids = [str(shape_id) for shape_id in split.get("train", [])]
    if not train_shape_ids:
        raise ValueError(f"Split has no train ids: {split_file}")

    cfg = dict(config.get("deepsdf", {}))
    if num_workers_override is not None:
        cfg["num_workers"] = int(num_workers_override)
    latent_options = _resolve_latent_training_options(cfg)
    resolved_config = dict(config)
    resolved_config["deepsdf"] = cfg
    decoder = _decoder_from_cfg(cfg)
    model = DeepSDFAutodecoder(
        num_shapes=len(train_shape_ids),
        latent_dim=int(cfg.get("latent_dim", 128)),
        decoder=decoder,
        latent_init_std=float(cfg.get("latent_init_std", 0.01)),
    ).to(device)
    optimizer = torch.optim.Adam(
        [
            {"params": model.decoder.parameters(), "lr": float(cfg.get("lr_decoder", 5e-4))},
            {"params": model.latent_codes.parameters(), "lr": float(cfg.get("lr_latents", 1e-3))},
        ]
    )
    start_epoch = 0
    if resume is not None:
        checkpoint = _torch_load(Path(resume), map_location=device)
        if not isinstance(checkpoint, dict):
            raise ValueError(f"Checkpoint must be a dict: {resume}")
        model.decoder.load_state_dict(checkpoint["decoder_state_dict"])
        model.latent_codes.load_state_dict(checkpoint["latent_codes_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        start_epoch = int(checkpoint.get("epoch", 0))

    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    cache: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
    sdf_paths = {shape_id: find_sdf_file(sdf_root, shape_id) for shape_id in train_shape_ids}
    rng = np.random.default_rng(seed)
    epochs = int(epochs_override or cfg.get("epochs", 1000))
    batch_shapes = int(cfg.get("batch_shapes", 8))
    samples_per_shape = int(cfg.get("samples_per_shape", 2048))
    sdf_band = cfg.get("sdf_clamp", 0.1)
    latent_l2_weight = float(latent_options["latent_l2_weight"])
    latent_regularizer = str(latent_options["latent_regularizer"])
    latent_hinge_radius = float(latent_options["latent_hinge_radius"])
    latent_hinge_weight = float(latent_options["latent_hinge_weight"])
    latent_noise_std = float(latent_options["latent_noise_std"])
    latent_dropout_p = float(latent_options["latent_dropout_p"])
    save_every = int(cfg.get("save_every", 100))
    num_workers = int(cfg.get("num_workers", 0))
    use_dataloader = num_workers > 0
    dataset = None
    loader = None
    if use_dataloader:
        dataset, loader = _make_dataloader(
            train_shape_ids,
            sdf_paths=sdf_paths,
            samples_per_shape=samples_per_shape,
            batch_shapes=batch_shapes,
            num_workers=num_workers,
            seed=seed,
            pin_memory=str(device).startswith("cuda"),
        )
    history_path = out_path / "train_history.csv"

    with history_path.open("a", newline="", encoding="utf-8") as f:
        history_fieldnames = _history_fieldnames(history_path)
        writer = csv.DictWriter(f, fieldnames=history_fieldnames)
        if f.tell() == 0:
            writer.writeheader()

        for epoch in tqdm(range(start_epoch, epochs), desc="train_deepsdf"):
            epoch_loss = 0.0
            epoch_sdf = 0.0
            epoch_l2 = 0.0
            epoch_reg = 0.0
            epoch_norm = 0.0
            steps = 0
            if use_dataloader:
                assert dataset is not None and loader is not None
                dataset.set_epoch(epoch)
                for shape_indices, xyz, target in loader:
                    shape_indices = shape_indices.to(device=device, non_blocking=True)
                    xyz = xyz.to(device=device, dtype=torch.float32, non_blocking=True)
                    target = target.to(device=device, dtype=torch.float32, non_blocking=True)
                    loss_value, sdf_value, l2_value, reg_value, norm_value = _train_batch(
                        model,
                        optimizer,
                        shape_indices,
                        xyz,
                        target,
                        sdf_band,
                        latent_regularizer,
                        latent_l2_weight,
                        latent_hinge_radius,
                        latent_hinge_weight,
                        latent_noise_std,
                        latent_dropout_p,
                    )
                    epoch_loss += loss_value
                    epoch_sdf += sdf_value
                    epoch_l2 += l2_value
                    epoch_reg += reg_value
                    epoch_norm += norm_value
                    steps += 1
            else:
                for batch_indices_np, batch_points, batch_sdf in _manual_epoch_batches(
                    train_shape_ids,
                    sdf_paths,
                    cache,
                    rng,
                    batch_shapes,
                    samples_per_shape,
                ):
                    shape_indices = torch.as_tensor(batch_indices_np, dtype=torch.long, device=device)
                    xyz = torch.as_tensor(batch_points, dtype=torch.float32, device=device)
                    target = torch.as_tensor(batch_sdf, dtype=torch.float32, device=device)
                    loss_value, sdf_value, l2_value, reg_value, norm_value = _train_batch(
                        model,
                        optimizer,
                        shape_indices,
                        xyz,
                        target,
                        sdf_band,
                        latent_regularizer,
                        latent_l2_weight,
                        latent_hinge_radius,
                        latent_hinge_weight,
                        latent_noise_std,
                        latent_dropout_p,
                    )
                    epoch_loss += loss_value
                    epoch_sdf += sdf_value
                    epoch_l2 += l2_value
                    epoch_reg += reg_value
                    epoch_norm += norm_value
                    steps += 1

            row = {
                "epoch": epoch + 1,
                "loss": epoch_loss / max(steps, 1),
                "sdf_l1": epoch_sdf / max(steps, 1),
                "latent_l2": epoch_l2 / max(steps, 1),
                "latent_reg": epoch_reg / max(steps, 1),
                "latent_norm": epoch_norm / max(steps, 1),
            }
            writer.writerow({field: row.get(field, "") for field in history_fieldnames})
            f.flush()
            if save_every > 0 and (epoch + 1) % save_every == 0:
                _save_checkpoint(
                    out_path / "deepsdf_last.pt",
                    epoch + 1,
                    model,
                    optimizer,
                    train_shape_ids,
                    resolved_config,
                )

    final_path = out_path / "deepsdf_final.pt"
    _save_checkpoint(final_path, epochs, model, optimizer, train_shape_ids, resolved_config)
    latents = model.export_latents()
    torch.save(
        {"train_latents": latents, "shape_ids": train_shape_ids, "config": resolved_config},
        out_path / "train_latents.pt",
    )
    np.savez_compressed(
        out_path / "train_latents.npz",
        train_latents=latents.numpy().astype(np.float32),
        shape_ids=np.asarray(train_shape_ids),
    )
    with (out_path / "resolved_config.json").open("w", encoding="utf-8") as f:
        json.dump(resolved_config, f, indent=2, sort_keys=True)
        f.write("\n")
    return final_path


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="Train a compact DeepSDF autodecoder.")
    parser.add_argument("--config", default=None)
    parser.add_argument("--sdf-root", required=True)
    parser.add_argument("--split-file", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--resume", default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    args = parser.parse_args(argv)
    config = load_yaml(args.config)
    final_path = train_deepsdf(
        config=config,
        sdf_root=args.sdf_root,
        split_file=args.split_file,
        out_dir=args.out,
        device=args.device,
        resume=args.resume,
        epochs_override=args.epochs,
        num_workers_override=args.num_workers,
    )
    print(json.dumps({"checkpoint": str(final_path)}, indent=2))


if __name__ == "__main__":
    main()

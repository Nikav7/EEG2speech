import argparse
import os
import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.patches import Patch
from sklearn.manifold import TSNE
from umap import UMAP


plt.style.use("seaborn-v0_8-whitegrid")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Plot UMAP/t-SNE embeddings for eeg_SPSs task folders by split, colored by class id."
    )
    parser.add_argument(
        "--eeg-spss-dir",
        default="eeg_SPSs",
        help="Root folder containing task subfolders (e.g., imagined_speech, attempted_speech), each with train/val/unseentest.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Directory where UMAP/t-SNE plots are saved. Default: same as --eeg-spss-dir.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducible UMAP projections.",
    )
    parser.add_argument(
        "--max-samples-per-split",
        type=int,
        default=250,
        help="Optional cap for number of trial samples per split.",
    )
    parser.add_argument(
        "--tasks",
        nargs="+",
        default=["imagined_speech", "attempted_speech", "listening"],
        help="Task folder names under --eeg-spss-dir to analyze.",
    )
    parser.add_argument(
        "--n-classes",
        type=int,
        default=74,
        help="Number of classes expected in plotting (used for fixed discrete color mapping).",
    )
    parser.add_argument(
        "--events-codes",
        default="events_codes.csv",
        help="Path to events_codes.csv file for class name mapping.",
    )
    return parser.parse_args()


def load_event_names(events_codes_path: str) -> dict:
    """Load event names from events_codes.csv file.
    Returns: dict mapping class_id (int) -> event_name (str).
    """
    id_to_name = {}
    if not os.path.exists(events_codes_path):
        print(f"Warning: {events_codes_path} not found; using class IDs for labels")
        return id_to_name
    
    df = pd.read_csv(events_codes_path, header=None, names=["name", "id", "mark"])
    for _, row in df.iterrows():
        class_id = int(row["id"])
        event_name = row["name"].strip().strip("'")
        id_to_name[class_id] = event_name
    
    return id_to_name


def make_distinct_colormap(n_classes: int):
    """Create a colormap with n_classes distinct colors sampled from hsv."""
    if n_classes <= 1:
        return ListedColormap([plt.get_cmap("hsv")(0.0)])
    base_cmap = plt.get_cmap("hsv")
    colors = [base_cmap(i / float(n_classes - 1)) for i in range(n_classes)]
    return ListedColormap(colors)


def create_discrete_norm_and_cmap(n_classes: int):
    """Create BoundaryNorm and colormap for discrete (no gradient) colors.
    Returns: (norm, cmap) where norm maps integer class indices 0..n_classes-1 to discrete colors.
    """
    cmap = make_distinct_colormap(n_classes)
    boundaries = np.arange(-0.5, n_classes + 0.5, 1.0)
    norm = BoundaryNorm(boundaries, cmap.N)
    return norm, cmap


def add_discrete_colorbar(fig, ax, cmap, norm, class_ids):
    """Attach a compact discrete colorbar to an axis.
    Tick labels show representative class IDs to match attached style.
    """
    if not class_ids:
        return

    n_classes = len(class_ids)
    mappable = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    cbar = fig.colorbar(mappable, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Class")

    # Keep labels readable for many classes.
    if n_classes <= 12:
        tick_idx = np.arange(n_classes)
    else:
        tick_idx = np.unique(np.linspace(0, n_classes - 1, 7, dtype=int))

    cbar.set_ticks(tick_idx)
    cbar.set_ticklabels([str(class_ids[i]) for i in tick_idx])


def parse_class_id(filename: str):
    match = re.search(r"class_(\d+)", filename)
    if not match:
        return None
    return int(match.group(1))


def load_split_trials(split_dir: str):
    if not os.path.isdir(split_dir):
        return np.zeros((0, 0), dtype=np.float64), np.zeros((0,), dtype=np.int32), []

    files = sorted([f for f in os.listdir(split_dir) if f.lower().endswith(".csv")])
    features = []
    labels = []
    kept_files = []
    expected_size = None
    skipped = []

    for file_name in files:
        class_id = parse_class_id(file_name)
        if class_id is None:
            continue

        file_path = os.path.join(split_dir, file_name)
        arr = pd.read_csv(file_path, header=None).to_numpy(dtype=np.float64)
        if arr.size == 0:
            continue

        flat_size = arr.reshape(-1).size
        
        # Set expected size from first valid file
        if expected_size is None:
            expected_size = flat_size
        
        # Skip files with mismatched dimensions
        if flat_size != expected_size:
            skipped.append((file_name, flat_size, expected_size))
            continue

        features.append(arr.reshape(-1))
        labels.append(class_id)
        kept_files.append(file_name)

    if skipped:
        print(f"Warning: {len(skipped)} files skipped in {split_dir} due to dimension mismatch:")
        for fname, got, expected in skipped[:3]:  # Show first 3
            print(f"  {fname}: got {got}, expected {expected}")
        if len(skipped) > 3:
            print(f"  ... and {len(skipped) - 3} more")

    if not features:
        return np.zeros((0, 0), dtype=np.float64), np.zeros((0,), dtype=np.int32), []

    return np.vstack(features), np.asarray(labels, dtype=np.int32), kept_files


def maybe_subsample(x, y, file_names, max_samples, rng):
    if x.shape[0] <= max_samples:
        return x, y, file_names
    idx = rng.choice(x.shape[0], size=max_samples, replace=False)
    idx = np.sort(idx)
    return x[idx], y[idx], [file_names[i] for i in idx]


def fit_umap(x: np.ndarray, seed: int, n_components: int = 2):
    if x.shape[0] < 3:
        return None

    n_neighbors = min(30, max(5, x.shape[0] // 10))
    reducer = UMAP(n_components=n_components, random_state=seed, n_neighbors=n_neighbors, min_dist=0.1)
    return reducer.fit_transform(x)


def fit_tsne(x: np.ndarray, seed: int, n_components: int = 2):
    if x.shape[0] < 3:
        return None

    perplexity = min(30, max(5, x.shape[0] // 10), x.shape[0] - 1)
    if perplexity < 2:
        return None

    model = TSNE(
        n_components=n_components,
        random_state=seed,
        perplexity=perplexity,
        init="pca",
        learning_rate="auto",
    )
    return model.fit_transform(x)


def plot_splits_umap(split_data, out_path: str, task_name: str, class_ids, id_to_name: dict = None):
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    norm, cmap = create_discrete_norm_and_cmap(len(class_ids))
    split_labels = {"train": "Train", "val": "Val", "unseentest": "Test"}

    for ax, split_name in zip(axes, ["train", "val", "unseentest"]):
        x, y_color = split_data[split_name]["x"], split_data[split_name]["y_color"]

        if x.shape[0] < 3:
            ax.text(0.5, 0.5, f"Not enough samples in {split_name}", ha="center", va="center")
            ax.set_title(f"{split_name} (n={x.shape[0]})")
            ax.set_xticks([])
            ax.set_yticks([])
            continue

        z = split_data[split_name]["z_umap2d"]
        scatter = ax.scatter(
            z[:, 0],
            z[:, 1],
            c=y_color,
            cmap=cmap,
            norm=norm,
            s=20,
            alpha=0.85,
            edgecolors="none",
        )
        ax.set_title(f"{split_labels[split_name]} (n={x.shape[0]})")
        ax.set_xlabel("UMAP 1")
        ax.set_ylabel("UMAP 2")
        ax.grid(True, alpha=0.25)
        add_discrete_colorbar(fig, ax, cmap, norm, class_ids)

    fig.suptitle(f"UMAP of {task_name} eeg_SPSs Trials by Split (colored by event)")
    fig.tight_layout(rect=[0, 0, 1, 1])
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_splits_umap_3d(split_data, out_path: str, task_name: str, class_ids, id_to_name: dict = None):
    fig = plt.figure(figsize=(18, 6))
    split_names = ["train", "val", "unseentest"]
    norm, cmap = create_discrete_norm_and_cmap(len(class_ids))
    split_labels = {"train": "Train", "val": "Val", "unseentest": "Test"}

    for i, split_name in enumerate(split_names, start=1):
        ax = fig.add_subplot(1, 3, i, projection="3d")
        x, y_color = split_data[split_name]["x"], split_data[split_name]["y_color"]

        if x.shape[0] < 3:
            ax.text2D(0.2, 0.5, f"Not enough samples in {split_name}", transform=ax.transAxes)
            ax.set_title(f"{split_name} (n={x.shape[0]})")
            continue

        z3 = split_data[split_name]["z_umap3d"]
        scatter = ax.scatter(
            z3[:, 0],
            z3[:, 1],
            z3[:, 2],
            c=y_color,
            cmap=cmap,
            norm=norm,
            s=18,
            alpha=0.85,
            edgecolors="none",
        )
        ax.set_title(f"{split_labels[split_name]} (n={x.shape[0]})")
        ax.set_xlabel("UMAP 1")
        ax.set_ylabel("UMAP 2")
        ax.set_zlabel("UMAP 3")
        add_discrete_colorbar(fig, ax, cmap, norm, class_ids)

    fig.suptitle(f"3D UMAP of {task_name} eeg_SPSs Trials by Split (colored by event)")
    fig.tight_layout(rect=[0, 0, 1, 1])
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_splits_tsne(split_data, out_path: str, task_name: str, class_ids, id_to_name: dict = None):
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    norm, cmap = create_discrete_norm_and_cmap(len(class_ids))
    split_labels = {"train": "Train", "val": "Val", "unseentest": "Test"}

    for ax, split_name in zip(axes, ["train", "val", "unseentest"]):
        x, y_color = split_data[split_name]["x"], split_data[split_name]["y_color"]

        if x.shape[0] < 3:
            ax.text(0.5, 0.5, f"Not enough samples in {split_name}", ha="center", va="center")
            ax.set_title(f"{split_name} (n={x.shape[0]})")
            ax.set_xticks([])
            ax.set_yticks([])
            continue

        z = split_data[split_name]["z_tsne2d"]
        scatter = ax.scatter(
            z[:, 0],
            z[:, 1],
            c=y_color,
            cmap=cmap,
            norm=norm,
            s=20,
            alpha=0.85,
            edgecolors="none",
        )
        ax.set_title(f"{split_labels[split_name]} (n={x.shape[0]})")
        ax.set_xlabel("t-SNE 1")
        ax.set_ylabel("t-SNE 2")
        ax.grid(True, alpha=0.25)
        add_discrete_colorbar(fig, ax, cmap, norm, class_ids)

    fig.suptitle(f"t-SNE of {task_name} eeg_SPSs Trials by Split (colored by event)")
    fig.tight_layout(rect=[0, 0, 1, 1])
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main():
    args = parse_args()

    output_dir = args.output_dir if args.output_dir else args.eeg_spss_dir
    os.makedirs(output_dir, exist_ok=True)

    id_to_name = load_event_names(args.events_codes)
    print(f"Loaded {len(id_to_name)} event names from {args.events_codes}")

    rng = np.random.default_rng(args.seed)
    for task_name in args.tasks:
        task_dir = os.path.join(args.eeg_spss_dir, task_name)
        if not os.path.isdir(task_dir):
            print(f"Skipping task '{task_name}': folder not found at {task_dir}")
            continue

        split_data = {}
        for split_name in ["train", "val", "unseentest"]:
            split_dir = os.path.join(task_dir, split_name)
            x, y, files = load_split_trials(split_dir)
            x, y, files = maybe_subsample(x, y, files, args.max_samples_per_split, rng)
            z_umap2d = fit_umap(x, args.seed, n_components=2)
            z_umap3d = fit_umap(x, args.seed, n_components=3)
            z_tsne2d = fit_tsne(x, args.seed, n_components=2)

            split_data[split_name] = {
                "x": x,
                "y": y,
                "y_color": np.zeros_like(y),
                "z_umap2d": z_umap2d,
                "z_umap3d": z_umap3d,
                "z_tsne2d": z_tsne2d,
                "files": files,
            }

        class_ids = sorted(
            {
                int(c)
                for split_name in ["train", "val", "unseentest"]
                for c in split_data[split_name]["y"]
            }
        )

        if len(class_ids) > args.n_classes:
            all_y = np.concatenate(
                [split_data[split_name]["y"] for split_name in ["train", "val", "unseentest"] if split_data[split_name]["y"].size > 0]
            )
            unique_ids, counts = np.unique(all_y, return_counts=True)
            order = np.argsort(-counts)
            keep_ids = set(int(unique_ids[i]) for i in order[: args.n_classes])

            print(
                f"{task_name}: keeping top {args.n_classes} classes by frequency "
                f"(from {len(class_ids)} total classes)"
            )

            for split_name in ["train", "val", "unseentest"]:
                y = split_data[split_name]["y"]
                if y.size == 0:
                    continue

                keep_mask = np.isin(y, list(keep_ids))
                split_data[split_name]["x"] = split_data[split_name]["x"][keep_mask]
                split_data[split_name]["y"] = y[keep_mask]
                split_data[split_name]["files"] = [
                    file_name for file_name, keep in zip(split_data[split_name]["files"], keep_mask) if keep
                ]

                if split_data[split_name]["z_umap2d"] is not None:
                    split_data[split_name]["z_umap2d"] = split_data[split_name]["z_umap2d"][keep_mask]
                if split_data[split_name]["z_umap3d"] is not None:
                    split_data[split_name]["z_umap3d"] = split_data[split_name]["z_umap3d"][keep_mask]
                if split_data[split_name]["z_tsne2d"] is not None:
                    split_data[split_name]["z_tsne2d"] = split_data[split_name]["z_tsne2d"][keep_mask]

            class_ids = sorted(keep_ids)

        if class_ids:
            class_to_idx = {class_id: idx for idx, class_id in enumerate(class_ids)}
            for split_name in ["train", "val", "unseentest"]:
                y = split_data[split_name]["y"]
                split_data[split_name]["y_color"] = np.asarray(
                    [class_to_idx[int(class_id)] for class_id in y], dtype=np.int32
                )

        umap2d_path = os.path.join(output_dir, f"{task_name}_umap_splits.png")
        umap3d_path = os.path.join(output_dir, f"{task_name}_umap3d_splits.png")
        tsne_path = os.path.join(output_dir, f"{task_name}_tsne_splits.png")

        plot_splits_umap(split_data, umap2d_path, task_name, class_ids, id_to_name)
        plot_splits_umap_3d(split_data, umap3d_path, task_name, class_ids, id_to_name)
        plot_splits_tsne(split_data, tsne_path, task_name, class_ids, id_to_name)

        summary_rows = []
        for split_name in ["train", "val", "unseentest"]:
            y = split_data[split_name]["y"]
            summary_rows.append(
                {
                    "task": task_name,
                    "split": split_name,
                    "n_trials": int(y.shape[0]),
                    "n_classes": int(np.unique(y).size) if y.size > 0 else 0,
                }
            )
        summary_df = pd.DataFrame(summary_rows)
        summary_path = os.path.join(output_dir, f"{task_name}_summary.csv")
        summary_df.to_csv(summary_path, index=False)

        print("Saved UMAP 2D plot:", umap2d_path)
        print("Saved UMAP 3D plot:", umap3d_path)
        print("Saved t-SNE plot:", tsne_path)
        print("Saved summary:", summary_path)


if __name__ == "__main__":
    main()

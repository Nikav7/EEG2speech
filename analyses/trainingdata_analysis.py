import argparse
import os
import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.patches import Patch
from sklearn.manifold import TSNE
#from umap import UMAP


plt.style.use("seaborn-v0_8-whitegrid")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Plot UMAP/t-SNE embeddings for eeg folders by split, colored by class id."
    )
    parser.add_argument(
        "--eeg-source-dir",
        default=os.path.join("non_augmented", "raw_pre_augmentation"), #"raw_pre_augmentation" "csp_post_augmentation" "raw_post_augmentation_no_csp"
        help="Root folder containing task subfolders (imagined_speech, attempted_speech, listening), each with train/val/test.",
    )
    parser.add_argument(
        "--subject-id",
        nargs="+",
        type=int,
        default=[16, 17, 18, 19],
        help="Subject ID for the EEG data.",
    )
    parser.add_argument(
        "--output-dir",
        default=os.path.join("plots","1619", "pre_aug_expREPRO_1-13"), # "plots/subjs16-19_cspcls1-13" "plots/subjs16-19_raw_pre_augmentation"
        help="Directory where UMAP/t-SNE plots are saved.."
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
        default=1600,
        help="Optional cap for number of trial samples per split.",
    )
    parser.add_argument(
        "--tasks",
        nargs="+",
        default=["attempted_speech", "imagined_speech", "listening"],
        help="EEG tasks to analyze.",
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
    parser.add_argument(
        "--feature-label",
        default="EEG after augmentation and CSP.",
        help="Label describing the feature/data type used in plot titles.",
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
    #BoundaryNorm requires at least two boundaries (one color region).
    safe_n_classes = max(1, int(n_classes))
    cmap = make_distinct_colormap(safe_n_classes)
    boundaries = np.arange(-0.5, safe_n_classes + 0.5, 1.0)
    norm = BoundaryNorm(boundaries, cmap.N)
    return norm, cmap


def add_discrete_colorbar(fig, ax, cmap, norm, class_ids, id_to_name: dict = None):
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
    if id_to_name:
        tick_labels = [f"{class_ids[i]}: {id_to_name.get(int(class_ids[i]), str(class_ids[i]))}" for i in tick_idx]
    else:
        tick_labels = [str(class_ids[i]) for i in tick_idx]
    cbar.set_ticklabels(tick_labels)


def parse_class_id(filename: str):
    # both names like class_12_*.csv and names like label012_*.csv.
    patterns = [r"class_(\d+)", r"label(\d+)"]
    for pattern in patterns:
        match = re.search(pattern, filename)
        if match:
            return int(match.group(1))
    return None


SPLITS = ["train", "val", "test"]
SPLIT_LABELS = {"train": "Train", "val": "Val", "test": "Test"}


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


# def fit_umap(x: np.ndarray, seed: int, n_components: int = 2):
#     if x.shape[0] < 3:
#         return None

#     n_neighbors = min(30, max(5, x.shape[0] // 10))
#     reducer = UMAP(n_components=n_components, random_state=seed, n_neighbors=n_neighbors, min_dist=0.1)
#     return reducer.fit_transform(x)


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
        init="random",
        learning_rate="auto",
    )
    return model.fit_transform(x)


def plot_splits_umap(split_data, out_path: str, task_name: str, class_ids, id_to_name: dict = None):
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    norm, cmap = create_discrete_norm_and_cmap(len(class_ids))

    for ax, split_name in zip(axes, SPLITS):
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
        ax.set_title(f"{SPLIT_LABELS[split_name]} (n={x.shape[0]})")
        ax.set_xlabel("UMAP 1")
        ax.set_ylabel("UMAP 2")
        ax.grid(True, alpha=0.25)
        add_discrete_colorbar(fig, ax, cmap, norm, class_ids)

    fig.suptitle(f"UMAP of {task_name} eeg Trials by Split (colored by event)")
    fig.tight_layout(rect=[0, 0, 1, 1])
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_splits_umap_3d(split_data, out_path: str, task_name: str, class_ids, id_to_name: dict = None):
    fig = plt.figure(figsize=(18, 6))
    split_names = SPLITS
    norm, cmap = create_discrete_norm_and_cmap(len(class_ids))

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
        ax.set_title(f"{SPLIT_LABELS[split_name]} (n={x.shape[0]})")
        ax.set_xlabel("UMAP 1")
        ax.set_ylabel("UMAP 2")
        ax.set_zlabel("UMAP 3")
        add_discrete_colorbar(fig, ax, cmap, norm, class_ids)

    fig.suptitle(f"3D UMAP of {task_name} eeg Trials by Split (colored by event)")
    fig.tight_layout(rect=[0, 0, 1, 1])
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _build_legend_layout(n_items: int):
    ncol = min(10, max(1, n_items))
    nrows = int(np.ceil(n_items / ncol))
    bottom_pad = 0.12 + max(0, nrows - 1) * 0.035
    return ncol, bottom_pad


def plot_tsne_splits(
    split_data,
    out_path: str,
    title: str,
    color_mode: str,
    seed: int,
    class_ids=None,
    id_to_name: dict = None,
    condition_names: dict = None,
    title_fontsize: int = 18,
    legend_fontsize: int = 15,
    subplot_title_fontsize: int = 15,
    axis_label_fontsize: int = 11,
):
    fig, axes = plt.subplots(1, 3, figsize=(20, 7))

    if color_mode == "event":
        cmap = make_distinct_colormap(len(class_ids))
        norm = BoundaryNorm(np.arange(-0.5, max(1, len(class_ids)) + 0.5, 1.0), cmap.N)
        legend_handles = []
        for idx, class_id in enumerate(class_ids):
            name = id_to_name.get(int(class_id), str(int(class_id))) if id_to_name else str(int(class_id))
            legend_handles.append(Patch(facecolor=cmap(idx), edgecolor="none", label=f"{int(class_id)}: {name}"))
    elif color_mode == "condition":
        cond_ids = sorted({int(v) for split_name in SPLITS for v in split_data[split_name]["y_condition"]})
        cmap = plt.get_cmap("tab10", max(2, len(cond_ids)))
        cond_to_idx = {cond_id: i for i, cond_id in enumerate(cond_ids)}
        norm = BoundaryNorm(np.arange(-0.5, len(cond_ids) + 0.5, 1.0), cmap.N)
        legend_handles = []
        for cond_id in cond_ids:
            cond_name = condition_names.get(cond_id, f"condition_{cond_id}") if condition_names else f"condition_{cond_id}"
            legend_handles.append(Patch(facecolor=cmap(cond_to_idx[cond_id]), edgecolor="none", label=cond_name))
    elif color_mode == "subject":
        subj_ids = sorted({int(v) for split_name in SPLITS for v in split_data[split_name]["y_subject"]})
        cmap = plt.get_cmap("tab20", max(2, len(subj_ids)))
        subj_to_idx = {subj_id: i for i, subj_id in enumerate(subj_ids)}
        norm = BoundaryNorm(np.arange(-0.5, len(subj_ids) + 0.5, 1.0), cmap.N)
        legend_handles = []
        for subj_id in subj_ids:
            legend_handles.append(Patch(facecolor=cmap(subj_to_idx[subj_id]), edgecolor="none", label=f"subj{subj_id}"))
    else:
        raise ValueError(f"Unsupported color_mode={color_mode}")

    for ax, split_name in zip(axes, SPLITS):
        x = split_data[split_name]["x"]
        n_split = x.shape[0]

        if n_split < 3:
            ax.text(0.5, 0.5, f"Not enough samples in {split_name}", ha="center", va="center")
            ax.set_title(f"{SPLIT_LABELS[split_name]} (n={n_split})", fontsize=subplot_title_fontsize)
            ax.set_xticks([])
            ax.set_yticks([])
            continue

        z = fit_tsne(x, seed, n_components=2)
        if z is None:
            ax.text(0.5, 0.5, f"t-SNE failed in {split_name}", ha="center", va="center")
            ax.set_title(f"{SPLIT_LABELS[split_name]} (n={n_split})", fontsize=subplot_title_fontsize)
            ax.set_xticks([])
            ax.set_yticks([])
            continue

        if color_mode == "event":
            y_plot = split_data[split_name]["y_color"]
        elif color_mode == "condition":
            y_plot = np.asarray([cond_to_idx[int(v)] for v in split_data[split_name]["y_condition"]], dtype=np.int32)
        else:
            y_plot = np.asarray([subj_to_idx[int(v)] for v in split_data[split_name]["y_subject"]], dtype=np.int32)

        ax.scatter(
            z[:, 0],
            z[:, 1],
            c=y_plot,
            cmap=cmap,
            norm=norm,
            s=14,
            alpha=0.8,
            edgecolors="none",
        )
        ax.set_title(f"{SPLIT_LABELS[split_name]} (n={n_split})", fontsize=subplot_title_fontsize)
        ax.set_xlabel("t-SNE 1", fontsize=axis_label_fontsize)
        ax.set_ylabel("t-SNE 2", fontsize=axis_label_fontsize)
        ax.grid(True, alpha=0.25)

    ncol, bottom_pad = _build_legend_layout(len(legend_handles))
    fig.legend(
        handles=legend_handles,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.01),
        ncol=ncol,
        fontsize=legend_fontsize,
        frameon=False,
    )
    fig.suptitle(title, fontsize=title_fontsize)
    fig.tight_layout(rect=[0, bottom_pad, 1, 1])
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_sample_counts_by_condition(
    split_data,
    out_path: str,
    condition_names: dict,
    subject_ids,
    split_name: str,
    color_by: str = None,
    id_to_name: dict = None,
    
):
    """Plot stacked sample counts per condition, colored by subject or class."""
    if color_by not in {"subject", "class"}:
        raise ValueError("color_by must be either 'subject' or 'class'")

    condition_ids = sorted(condition_names)
    subject_ids = sorted(int(subject_id) for subject_id in subject_ids)

    condition_to_idx = {condition_id: idx for idx, condition_id in enumerate(condition_ids)}
    y_condition = np.asarray(split_data.get("y_condition", []), dtype=np.int32)
    y_subject = np.asarray(split_data.get("y_subject", []), dtype=np.int32)
    y_class = np.asarray(split_data.get("y", []), dtype=np.int32)

    if color_by == "subject":
        category_ids = subject_ids
        category_to_idx = {category_id: idx for idx, category_id in enumerate(category_ids)}
        category_labels = [f"Subject {category_id}" for category_id in category_ids]
        title_suffix = "by subject"
    else:
        category_ids = sorted(int(class_id) for class_id in np.unique(y_class))
        category_to_idx = {category_id: idx for idx, category_id in enumerate(category_ids)}
        category_labels = [
            f"Class {class_id}: {id_to_name.get(class_id, class_id) if id_to_name else class_id}"
            for class_id in category_ids
        ]
        title_suffix = "by class"

    counts = np.zeros((len(condition_ids), len(category_ids)), dtype=np.int32)

    color_values = y_subject if color_by == "subject" else y_class
    for condition_id, color_value in zip(y_condition, color_values):
        condition_idx = condition_to_idx.get(int(condition_id))
        category_idx = category_to_idx.get(int(color_value))
        if condition_idx is not None and category_idx is not None:
            counts[condition_idx, category_idx] += 1

    fig, ax = plt.subplots(figsize=(max(9, len(condition_ids) * 1.2), 7))
    x = np.arange(len(condition_ids))
    bottoms = np.zeros(len(condition_ids), dtype=np.int32)
    cmap = plt.get_cmap("tab20", max(1, len(category_ids)))

    for category_idx, category_label in enumerate(category_labels):
        values = counts[:, category_idx]
        ax.bar(
            x,
            values,
            bottom=bottoms,
            color=cmap(category_idx),
            label=category_label,
            edgecolor="white",
            linewidth=0.5,
        )
        bottoms += values

    condition_totals = counts.sum(axis=1)
    condition_labels = [
        f"{condition_names[condition_id]} (n={total})"
        for condition_id, total in zip(condition_ids, condition_totals)
    ]
    ax.set_xticks(x)
    ax.set_xticklabels(condition_labels)
    ax.set_xlabel("Condition")
    ax.set_ylabel(f"Number of samples (total={int(condition_totals.sum())})")
    ax.set_ylim(0, 2500)
    ax.set_title(f"Number of samples per condition {title_suffix} ({SPLIT_LABELS[split_name]})")
    ax.grid(axis="y", alpha=0.25)
    ax.set_axisbelow(True)
    ax.legend(title="Subjects", loc="upper left", bbox_to_anchor=(1.01, 1), frameon=False)
    fig.tight_layout()
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def save_class_subject_counts_csv(split_data, out_path: str, subject_ids):
    """Save a class-by-subject sample-count table for one split."""
    subject_ids = sorted(int(subject_id) for subject_id in subject_ids)
    class_ids = sorted(int(class_id) for class_id in np.unique(split_data.get("y", [])))
    counts = np.zeros((len(class_ids), len(subject_ids)), dtype=np.int32)

    class_to_idx = {class_id: idx for idx, class_id in enumerate(class_ids)}
    subject_to_idx = {subject_id: idx for idx, subject_id in enumerate(subject_ids)}
    y_class = np.asarray(split_data.get("y", []), dtype=np.int32)
    y_subject = np.asarray(split_data.get("y_subject", []), dtype=np.int32)

    for class_id, subject_id in zip(y_class, y_subject):
        class_idx = class_to_idx.get(int(class_id))
        subject_idx = subject_to_idx.get(int(subject_id))
        if class_idx is not None and subject_idx is not None:
            counts[class_idx, subject_idx] += 1

    counts_df = pd.DataFrame(
        counts,
        index=[f"class_{class_id}" for class_id in class_ids],
        columns=[f"subject_{subject_id}" for subject_id in subject_ids],
    )
    counts_df.index.name = "class"
    counts_df.to_csv(out_path)


def save_class_condition_counts_csv(
    split_data,
    out_path: str,
    subject_id: int,
    condition_names: dict,
):
    """Save a class-by-condition sample-count table for one subject and split."""
    condition_ids = sorted(int(condition_id) for condition_id in condition_names)
    class_ids = sorted(int(class_id) for class_id in np.unique(split_data.get("y", [])))
    counts = np.zeros((len(class_ids), len(condition_ids)), dtype=np.int32)

    class_to_idx = {class_id: idx for idx, class_id in enumerate(class_ids)}
    condition_to_idx = {condition_id: idx for idx, condition_id in enumerate(condition_ids)}
    y_class = np.asarray(split_data.get("y", []), dtype=np.int32)
    y_subject = np.asarray(split_data.get("y_subject", []), dtype=np.int32)
    y_condition = np.asarray(split_data.get("y_condition", []), dtype=np.int32)

    for class_id, current_subject_id, condition_id in zip(y_class, y_subject, y_condition):
        if int(current_subject_id) != int(subject_id):
            continue
        class_idx = class_to_idx.get(int(class_id))
        condition_idx = condition_to_idx.get(int(condition_id))
        if class_idx is not None and condition_idx is not None:
            counts[class_idx, condition_idx] += 1

    counts_df = pd.DataFrame(
        counts,
        index=[f"class_{class_id}" for class_id in class_ids],
        columns=[condition_names[condition_id] for condition_id in condition_ids],
    )
    counts_df.index.name = "class"
    counts_df.to_csv(out_path)


def main():
    args = parse_args()

    feature_label = args.feature_label
    tsne_title_fontsize = 20
    tsne_legend_fontsize = 11
    tsne_subplot_title_fontsize = 15
    tsne_axis_label_fontsize = 14

    output_dir = args.output_dir if args.output_dir else args.eeg_source_dir
    os.makedirs(output_dir, exist_ok=True)

    id_to_name = load_event_names(args.events_codes)
    print(f"Loaded {len(id_to_name)} event names from {args.events_codes}")

    rng = np.random.default_rng(args.seed)

    root_name = os.path.basename(os.path.normpath(args.eeg_source_dir)).lower()
    root_is_subject_dir = root_name.startswith("subj")

    condition_name_to_idx = {name: idx for idx, name in enumerate(args.tasks)}
    condition_idx_to_name = {idx: name for name, idx in condition_name_to_idx.items()}
    cumulative_split = {
        split_name: {
            "x_parts": [],
            "y_parts": [],
            "y_subject_parts": [],
            "y_condition_parts": [],
            "files": [],
        }
        for split_name in SPLITS
    }

    for task_name in args.tasks:
        task_condition_idx = condition_name_to_idx[task_name]
        if root_is_subject_dir:
            task_dirs = []
            for subj_id in args.subject_id:
                task_dir = os.path.join(args.eeg_source_dir, task_name)
                if os.path.isdir(task_dir):
                    task_dirs.append((subj_id, task_dir))
        else:
            task_dirs = []
            for subj_id in args.subject_id:
                task_dir = os.path.join(args.eeg_source_dir, f"subj{subj_id}", task_name)
                if os.path.isdir(task_dir):
                    task_dirs.append((subj_id, task_dir))

        if not task_dirs:
            print(
                f"for subject IDs {args.subject_id}"
                f"under {args.eeg_source_dir}"
                f"for subject IDs {args.subject_id}"
            )
            continue

        split_data = {
            split_name: {
                "x_parts": [],
                "y_parts": [],
                "subject_parts": [],
                "condition_parts": [],
                "files": [],
            }
            for split_name in SPLITS
        }

        print(f"{task_name}: loading {len(task_dirs)} subject folder(s)")
        for subj_id, task_dir in task_dirs:
            for split_name in SPLITS:
                split_dir = os.path.join(task_dir, split_name)
                x, y, files = load_split_trials(split_dir)
                x, y, files = maybe_subsample(x, y, files, args.max_samples_per_split, rng)

                if y.size == 0:
                    continue

                split_data[split_name]["x_parts"].append(x)
                split_data[split_name]["y_parts"].append(y)
                split_data[split_name]["subject_parts"].append(np.full(y.shape, subj_id, dtype=np.int32))
                split_data[split_name]["condition_parts"].append(np.full(y.shape, task_condition_idx, dtype=np.int32))
                split_data[split_name]["files"].extend([f"subj{subj_id}/{file_name}" for file_name in files])

        for split_name in SPLITS:
            x_parts = split_data[split_name].pop("x_parts")
            y_parts = split_data[split_name].pop("y_parts")
            subject_parts = split_data[split_name].pop("subject_parts")
            condition_parts = split_data[split_name].pop("condition_parts")

            if x_parts:
                x = np.vstack(x_parts)
                y = np.concatenate(y_parts)
                y_subject = np.concatenate(subject_parts)
                y_condition = np.concatenate(condition_parts)
            else:
                x = np.zeros((0, 0), dtype=np.float64)
                y = np.zeros((0,), dtype=np.int32)
                y_subject = np.zeros((0,), dtype=np.int32)
                y_condition = np.zeros((0,), dtype=np.int32)

            split_data[split_name] = {
                "x": x,
                "y": y,
                "y_subject": y_subject,
                "y_condition": y_condition,
                "y_color": np.zeros_like(y),
                "files": split_data[split_name]["files"],
            }

        class_ids = sorted(
            {
                int(c)
                for split_name in SPLITS
                for c in split_data[split_name]["y"]
            }
        )

        if len(class_ids) > args.n_classes:
            all_y = np.concatenate(
                [split_data[split_name]["y"] for split_name in SPLITS if split_data[split_name]["y"].size > 0]
            )
            unique_ids, counts = np.unique(all_y, return_counts=True)
            order = np.argsort(-counts)
            keep_ids = set(int(unique_ids[i]) for i in order[: args.n_classes])

            print(
                f"{task_name}: keeping top {args.n_classes} classes by frequency "
                f"(from {len(class_ids)} total classes)"
            )

            for split_name in ["train", "val", "test"]:
                y = split_data[split_name]["y"]
                if y.size == 0:
                    continue

                keep_mask = np.isin(y, list(keep_ids))
                split_data[split_name]["x"] = split_data[split_name]["x"][keep_mask]
                split_data[split_name]["y"] = y[keep_mask]
                split_data[split_name]["files"] = [
                    file_name for file_name, keep in zip(split_data[split_name]["files"], keep_mask) if keep
                ]
                split_data[split_name]["y_subject"] = split_data[split_name]["y_subject"][keep_mask]
                split_data[split_name]["y_condition"] = split_data[split_name]["y_condition"][keep_mask]

            class_ids = sorted(keep_ids)

        if class_ids:
            class_to_idx = {class_id: idx for idx, class_id in enumerate(class_ids)}
            for split_name in ["train", "val", "test"]:
                y = split_data[split_name]["y"]
                split_data[split_name]["y_color"] = np.asarray(
                    [class_to_idx[int(class_id)] for class_id in y], dtype=np.int32
                )


        tsne_file = os.path.join(output_dir, f"tsne_{task_name}_by_split.png")
        plot_tsne_splits(
            split_data=split_data,
            out_path=tsne_file,
            title=f"{feature_label} visualized with t-SNE {task_name} by split - Subject " + ", ".join(str(s) for s in args.subject_id),
            color_mode="event",
            seed=args.seed,
            class_ids=class_ids,
            id_to_name=id_to_name,
            title_fontsize=tsne_title_fontsize,
            legend_fontsize=tsne_legend_fontsize,
            subplot_title_fontsize=tsne_subplot_title_fontsize,
            axis_label_fontsize=tsne_axis_label_fontsize,
        )

        for split_name in SPLITS:
            if split_data[split_name]["x"].shape[0] == 0:
                continue
            cumulative_split[split_name]["x_parts"].append(split_data[split_name]["x"])
            cumulative_split[split_name]["y_parts"].append(split_data[split_name]["y"])
            cumulative_split[split_name]["y_subject_parts"].append(split_data[split_name]["y_subject"])
            cumulative_split[split_name]["y_condition_parts"].append(split_data[split_name]["y_condition"])
            cumulative_split[split_name]["files"].extend(split_data[split_name]["files"])

        summary_rows = []
        for split_name in ["train", "val", "test"]:
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

        #print("Saved UMAP 2D plot:", umap2d_path)
        print("Saved t-SNE plot:", tsne_file)
        print("Saved summary:", summary_path)

    cumulative_ready = {}
    for split_name in SPLITS:
        if cumulative_split[split_name]["x_parts"]:
            cumulative_ready[split_name] = {
                "x": np.vstack(cumulative_split[split_name]["x_parts"]),
                "y": np.concatenate(cumulative_split[split_name]["y_parts"]),
                "y_subject": np.concatenate(cumulative_split[split_name]["y_subject_parts"]),
                "y_condition": np.concatenate(cumulative_split[split_name]["y_condition_parts"]),
                "files": cumulative_split[split_name]["files"],
            }
        else:
            cumulative_ready[split_name] = {
                "x": np.zeros((0, 0), dtype=np.float64),
                "y": np.zeros((0,), dtype=np.int32),
                "y_subject": np.zeros((0,), dtype=np.int32),
                "y_condition": np.zeros((0,), dtype=np.int32),
                "files": [],
            }

        cumulative_ready[split_name]["y_color"] = np.zeros_like(cumulative_ready[split_name]["y"])

        sample_counts_path = os.path.join(output_dir, f"sample_counts_by_condition_{split_name}.png")
        plot_sample_counts_by_condition(
            split_data=cumulative_ready[split_name],
            out_path=sample_counts_path,
            condition_names=condition_idx_to_name,
            subject_ids=args.subject_id,
            split_name=split_name,
            color_by="subject",
            id_to_name=id_to_name,
        )
        print("Saved sample-count plot:", sample_counts_path)

        class_subject_counts_path = os.path.join(
            output_dir, f"sample_counts_by_class_subject_{split_name}.csv"
        )
        save_class_subject_counts_csv(
            split_data=cumulative_ready[split_name],
            out_path=class_subject_counts_path,
            subject_ids=args.subject_id,
        )
        print("Saved class-subject count summary:", class_subject_counts_path)

        for subject_id in args.subject_id:
            class_condition_counts_path = os.path.join(
                output_dir,
                f"sample_counts_by_class_condition_subj{subject_id}_{split_name}.csv",
            )
            save_class_condition_counts_csv(
                split_data=cumulative_ready[split_name],
                out_path=class_condition_counts_path,
                subject_id=subject_id,
                condition_names=condition_idx_to_name,
            )
            print("Saved subject class-condition summary:", class_condition_counts_path)

    cumulative_has_samples = any(cumulative_ready[split_name]["x"].shape[0] > 0 for split_name in SPLITS)
    if cumulative_has_samples:
        cumulative_path = os.path.join(output_dir, "tsne_all_conditions_by_split.png")
        plot_tsne_splits(
            split_data=cumulative_ready,
            out_path=cumulative_path,
            title=f"{feature_label} with t-SNE by split (colored by condition) - Subject " + ", ".join(str(s) for s in args.subject_id),
            color_mode="condition",
            seed=args.seed,
            condition_names=condition_idx_to_name,
            title_fontsize=tsne_title_fontsize,
            legend_fontsize=tsne_legend_fontsize,
            subplot_title_fontsize=tsne_subplot_title_fontsize,
            axis_label_fontsize=tsne_axis_label_fontsize,
        )
        print("Saved cumulative t-SNE plot:", cumulative_path)

        cumulative_subject_path = os.path.join(output_dir, "tsne_all_conditions_by_split_by_subject.png")
        plot_tsne_splits(
            split_data=cumulative_ready,
            out_path=cumulative_subject_path,
            title=f"{feature_label} with t-SNE by split (colored by subject) - Subject " + ", ".join(str(s) for s in args.subject_id),
            color_mode="subject",
            seed=args.seed,
            title_fontsize=tsne_title_fontsize,
            legend_fontsize=tsne_legend_fontsize,
            subplot_title_fontsize=tsne_subplot_title_fontsize,
            axis_label_fontsize=tsne_axis_label_fontsize,
        )
        print("Saved cumulative subject t-SNE plot:", cumulative_subject_path)

if __name__ == "__main__":
    main()

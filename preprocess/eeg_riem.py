import os
import numpy as np
import pandas as pd
import mne
from pyriemann.estimation import Covariances
from pyriemann.tangentspace import TangentSpace
from pyriemann.geometry.mean import mean_riemann
from pyriemann.geometry.tangentspace import tangent_space
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from preprocess_utils import *


CONDITIONS = {1: 'attempted_speech', 2: 'imagined_speech', 3: 'listening'}

# Windowing setup
WIN_MS = 125 
STRIDE_MS = 125 


#Plots
def plot_tsne(X_emb, y_class, y_cond, y_subject):
    print("Running t-SNE…")
    X_tsne = TSNE(n_components=2, perplexity=30, random_state=42).fit_transform(X_emb)
    fig, axes = plt.subplots(1, 3, figsize=(24, 7))
    sc1 = axes[0].scatter(X_tsne[:, 0], X_tsne[:, 1], c=y_cond,  cmap='tab10',   s=4, alpha=0.7)
    sc2 = axes[1].scatter(X_tsne[:, 0], X_tsne[:, 1], c=y_class, cmap='Spectral', s=4, alpha=0.7)
    subj_unique = np.unique(y_subject)
    subj_idx = {s: i for i, s in enumerate(subj_unique)}
    subj_colors = np.array([subj_idx[s] for s in y_subject])
    cmap_subj = plt.cm.get_cmap('tab20', max(len(subj_unique), 2))
    sc3 = axes[2].scatter(X_tsne[:, 0], X_tsne[:, 1], c=subj_colors, cmap=cmap_subj, s=4, alpha=0.7)
    plt.colorbar(sc1, ax=axes[0]).set_label('Condition')
    plt.colorbar(sc2, ax=axes[1]).set_label('Class (word code 1-74)')
    plt.colorbar(sc3, ax=axes[2]).set_label('Subject index')
    norm    = plt.Normalize(vmin=y_cond.min(), vmax=y_cond.max())
    handles = [plt.Line2D([0], [0], marker='o', color='w',
               markerfacecolor=plt.cm.tab10(norm(i)), markersize=8, label=n)
               for i, n in CONDITIONS.items()]
    handles_subj = [plt.Line2D([0], [0], marker='o', color='w',
                   markerfacecolor=cmap_subj(i / max(len(subj_unique) - 1, 1)), markersize=7,
                   label=f'subj{int(s)}')
                   for i, s in enumerate(subj_unique)]
    axes[0].legend(handles=handles, loc='best', fontsize=8)
    axes[2].legend(handles=handles_subj, loc='best', fontsize=8)
    n = len(X_emb)
    axes[0].set_title(f't-SNE 2D – Condition (n={n})')
    axes[1].set_title(f't-SNE 2D – Class (74 words) (n={n})')
    axes[2].set_title(f't-SNE 2D – Subject (n={n})')
    plt.suptitle(f'Riemannian Tangent Space + t-SNE  |  {n} samples', fontsize=13)
    plt.tight_layout()
    plt.savefig(f'riemannian_tsne2d_cond_class_subj.png', dpi=600, bbox_inches='tight')
    #plt.show()


def plot_tsne_imagined_by_split(X_emb, y_cond, y_split, y_subject, imagined_cond=2):
    """t-SNE of the imagined-speech condition only; one subplot per split, coloured by subject."""
    mask = y_cond == imagined_cond
    X, split, subj = X_emb[mask], y_split[mask], y_subject[mask]
    print(f"Running t-SNE on imagined speech ({len(X)} samples)…")
    X_tsne = TSNE(n_components=2, perplexity=30, random_state=42).fit_transform(X)
    subj_unique = np.unique(subj)
    cmap = plt.cm.get_cmap('tab20', max(len(subj_unique), 2))
    fig, axes = plt.subplots(1, 3, figsize=(24, 7), sharex=True, sharey=True)
    for ax, name in zip(axes, ['train', 'val', 'test']):
        m = split == name
        for i, s in enumerate(subj_unique):
            ms = m & (subj == s)
            ax.scatter(X_tsne[ms, 0], X_tsne[ms, 1], color=cmap(i), s=6, alpha=0.7, label=f'subj{int(s)}')
        ax.set_title(f'{name} (n={int(m.sum())})')
    axes[0].legend(loc='best', fontsize=8)
    plt.suptitle(f'Imagined speech – Riemannian t-SNE by split, coloured by subject  |  {len(X)} samples', fontsize=13)
    plt.tight_layout()
    plt.savefig(f'riemannian_tsne2d_imagined_by_split_subj.png', dpi=600, bbox_inches='tight')
    #plt.show()


# def plot_umap3d(X_emb, y_class, y_cond, out_dir='new_plots'):
#     print("Running UMAP 3D…")
#     X_3d = UMAP(n_components=3, n_neighbors=15, min_dist=0.1, metric='euclidean').fit_transform(X_emb)
#     os.makedirs(out_dir, exist_ok=True)
#     for label_arr, cmap, title, fname in [
#         (y_cond,  'tab10',   'Condition',       'riemannian_umap3d_cond'),
#         (y_class, 'Spectral','Class (74 words)', 'riemannian_umap3d_class'),
#     ]:
#         fig = plt.figure(figsize=(10, 8))
#         ax  = fig.add_subplot(111, projection='3d')
#         sc  = ax.scatter(X_3d[:, 0], X_3d[:, 1], X_3d[:, 2], c=label_arr, cmap=cmap, s=4, alpha=0.7)
#         fig.colorbar(sc, ax=ax, pad=0.1).set_label(title)
#         if label_arr is y_cond:
#             norm    = plt.Normalize(vmin=y_cond.min(), vmax=y_cond.max())
#             handles = [plt.Line2D([0], [0], marker='o', color='w',
#                        markerfacecolor=plt.cm.tab10(norm(i)), markersize=8, label=n)
#                        for i, n in CONDITION_NAMES.items()]
#             ax.legend(handles=handles, loc='best', fontsize=8)
#         ax.set_title(f'UMAP 3D – {title}')
#         ax.set_xlabel('UMAP-1'); ax.set_ylabel('UMAP-2'); ax.set_zlabel('UMAP-3')
#         plt.suptitle('Riemannian Tangent Space + UMAP 3D', fontsize=13)
#         plt.tight_layout()
#         plt.savefig(f'{out_dir}/{fname}.png', dpi=300, bbox_inches='tight')
#         plt.show()



def _save_embedding_csvs(emb, labels, gidx, subjects, output_dir, cond_name, split):
    """Save one CSV (n_windows x feat_dim) per trial, same layout as the raw splits."""
    for i, (e, lab, g, s) in enumerate(zip(emb, labels, gidx, subjects)):
        out_dir = os.path.join(output_dir, f'subj{int(s)}', cond_name, split)
        os.makedirs(out_dir, exist_ok=True)
        fname = f'label{int(lab):03d}_samplegidx{int(g):05d}_row{i:06d}.csv'
        np.savetxt(os.path.join(out_dir, fname), e, delimiter=',', fmt='%.16g')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--subjects", nargs="+", type=int, default=[15, 16, 17, 18, 19])
    parser.add_argument("--output-dir", default="riem_embeddings_aug40")
    parser.add_argument("--rawdata-dir", default="eegdata_rawsplits/rawsplits_tog_riem/raw_post_augmentation40")

    args = parser.parse_args()
    OUTPUT, RAWDATA = args.output_dir, args.rawdata_dir

    tsne_X, tsne_cls, tsne_cond, tsne_subj, tsne_split = [], [], [], [], []
    for cond, cond_name in CONDITIONS.items():
        # Pool all subjects per split so one reference mean is fit on the pooled train set
        pooled = {s: {'x': [], 'y': [], 'idx': [], 'subj': []} for s in ('train', 'val', 'test')}
        for subject in args.subjects:
            x_tr, y_tr, x_val, y_val, x_ts, y_ts, i_tr, i_val, i_ts = load_split_data(
                RAWDATA, subject, condition=cond_name)
            for split, x, y, idx in (('train', x_tr, y_tr, i_tr), ('val', x_val, y_val, i_val), ('test', x_ts, y_ts, i_ts)):
                if len(x) == 0:
                    continue
                pooled[split]['x'].append(x)
                pooled[split]['y'].append(y)
                pooled[split]['idx'].append(idx)
                pooled[split]['subj'].append(np.full(len(x), subject))
        pooled = {s: {k: np.concatenate(v) for k, v in d.items()} for s, d in pooled.items() if d['x']}
        if 'train' not in pooled:
            print(f"{cond_name}: no training data, skipping")
            continue

        embeddings, _ = compute_riemannian_embeddings(
            {s: d['x'] for s, d in pooled.items()},
            output_dir=os.path.join(OUTPUT, cond_name),
            sfreq=SFREQ, win_ms=WIN_MS, stride_ms=STRIDE_MS,
        )
        for split, emb in embeddings.items():
            d = pooled[split]
            _save_embedding_csvs(emb, d['y'], d['idx'], d['subj'], OUTPUT, cond_name, split)
            tsne_X.append(emb.mean(axis=1))  # average windows for visualisation
            tsne_cls.append(d['y'])
            tsne_subj.append(d['subj'])
            tsne_cond.append(np.full(len(emb), cond))
            tsne_split.append(np.full(len(emb), split))

    if tsne_X:
        X_all, cls_all = np.concatenate(tsne_X), np.concatenate(tsne_cls)
        cond_all, subj_all = np.concatenate(tsne_cond), np.concatenate(tsne_subj)
        plot_tsne(X_all, cls_all, cond_all, subj_all)
        plot_tsne_imagined_by_split(X_all, cond_all, np.concatenate(tsne_split), subj_all)



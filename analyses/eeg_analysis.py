import os

import matplotlib.pyplot as plt
import mne
import numpy as np
import pandas as pd
from sklearn.manifold import TSNE
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.patches import Patch
from trainingdata_analysis import make_distinct_colormap, load_event_names, plot_tsne_splits

# Load EEG data and events for subjects:
subjects = [15, 16, 17, 18, 19]
DATA_DIR = 'C:/Users/hssn_/Desktop/EEG2speech/clean_data01-120Hz'
SFREQ = 250.0
EVENT_SFREQ = 250.0  
COND3_FILTER_BAND = (0.25, 40.0)

CONDITION_BASE = {1: 100, 2: 200, 3: 400}
CONDITION_NAMES = {1: 'Imagined speech', 2: 'Listening', 3: 'Attempted speech'}
# Time windows per condition for cisualization in the time domain (in seconds)
COND_TWIN= {1: (-0.5, 2.5), 2: (-0.5, 2.5), 3: (-0.5, 3.5)}

COND_TWIN_tsne= {1: (0.0, 2.0), 2: (0.0, 2.0), 3: (0.2, 2.2)}

tsne = True  

event_names = load_event_names('events_codes.csv')


def load_data(subjects, data_dir=DATA_DIR):
    event_sfreq = EVENT_SFREQ
    event_df = pd.read_csv('events_codes.csv', header=None, names=['word', 'code', 'type'])
    code_to_name = dict(zip(event_df['code'], event_df['word'].str.strip("'")))
    raw_all, markers_all = {}, {}

    for subject in subjects:
        eeg_file = os.path.join(data_dir, f'clean_eeg_subj{subject}.npy')
        evts_file = os.path.join(data_dir, f'events_subj{subject}.npy')
        ch_file = os.path.join(data_dir, 'channel_names.csv')
        if not all(os.path.exists(f) for f in [eeg_file, evts_file, ch_file]):
            print(f"Subject {subject}: missing files, skipping")
            continue

        ch_names = pd.read_csv(ch_file)['Channel'].tolist()
        eog_chs = {'EOG1', 'EOG2', 'EOG3'}
        ch_types = ['eog' if ch in eog_chs else 'eeg' for ch in ch_names]
        info = mne.create_info(ch_names=ch_names, sfreq=SFREQ, ch_types=ch_types)

        raw = mne.io.RawArray(np.load(eeg_file), info)
        raw.set_montage('standard_1020')
        raw_all[subject] = raw

        markers = np.load(evts_file)[:-1]
        if event_sfreq != SFREQ:
            scale = SFREQ / event_sfreq
            markers = markers.copy()
            markers[:, 0] = np.round(markers[:, 0] * scale).astype(markers.dtype)
            markers[:, 0] = np.clip(markers[:, 0], 0, len(raw) - 1)
        markers_all[subject] = markers

    print(f"Loaded {len(raw_all)} subjects")
    return raw_all, markers_all, code_to_name


# Epoching

def extract_epochs(raw_all, markers_all):
    epochs_all = {c: {} for c in CONDITION_BASE}

    for subject in raw_all:
        markers = markers_all[subject].copy()

        # Merge imagined speech: recode 300-series -> 100-series (same word offsets)
        for i in range(len(markers)):
            if 300 <= markers[i, 2] < 400:
                markers[i, 2] = 100 + (markers[i, 2] - 300)

        # Recode event-50 -> 400 + word_code from preceding imagined-speech event
        for i in range(len(markers)):
            if markers[i, 2] == 50:
                prev_imag = [j for j in range(i) if 100 <= markers[j, 2] < 200]
                if prev_imag:
                    markers[i, 2] = 400 + (markers[prev_imag[-1], 2] - 100)

        codes = np.unique(markers[:, 2])
        for cond, base in CONDITION_BASE.items():
            cond_codes = [c for c in codes if base <= c < base + 100]
            if not cond_codes:
                continue
            if tsne:
                tmin, tmax = COND_TWIN_tsne[cond]
            else:
                tmin, tmax = COND_TWIN[cond]
            
            epochs_all[cond][subject] = mne.Epochs(
                raw_all[subject],
                markers,
                event_id={f'e{c}': int(c) for c in cond_codes},
                tmin=tmin,
                tmax=tmax,
                picks='eeg',
                baseline=None,
                preload=True,
                reject=None,
                flat=None,
            )

            epochs_all[cond][subject].drop_log_stats()
    

    return epochs_all


def add_evoked_if_nonempty(epochs_dict, evoked_dict, subject, skip_message=None):
    ep = epochs_dict[subject]
    if ep is not None and len(ep) > 0:
        evoked_dict[subject] = ep.average()
    elif skip_message:
        print(skip_message.format(subject=subject))


def n_epochs_before_drop(ep):
    dropped_target = sum(
        1 for reasons in ep.drop_log if reasons and not all(r == 'IGNORED' for r in reasons)
    )
    return len(ep) + dropped_target


def print_evoked_summary(label, epochs_dict, evoked_dict):
    all_subjects = sorted(epochs_dict.keys())
    contributing_subjects = sorted(evoked_dict.keys())
    total_original = sum(n_epochs_before_drop(epochs_dict[s]) for s in all_subjects)
    total_kept = sum(len(epochs_dict[s]) for s in all_subjects)

    print(
        f"{label}: original={total_original}, used={total_kept} "
        f"across {len(contributing_subjects)}/{len(all_subjects)} subjects"
    )
    if all_subjects:
        per_subject = ', '.join(
            [f"S{s}={n_epochs_before_drop(epochs_dict[s])}->{len(epochs_dict[s])}" for s in all_subjects]
        )
        print(f"  per-subject (original->used): {per_subject}")
    else:
        print('  per-subject (original->used): none')

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


def _build_legend_layout(n_items: int):
    ncol = min(10, max(1, n_items))
    nrows = int(np.ceil(n_items / ncol))
    bottom_pad = 0.12 + max(0, nrows - 1) * 0.035
    return ncol, bottom_pad


def prepare_tsne_data(epochs_all):
    """Converts nested MNE Epochs dict into flattened 2D arrays & subject IDs per condition."""
    tsne_data = {}
    for cond, subjs_dict in epochs_all.items():
        x_list, y_subj_list = [], []
        for subj_id, ep in subjs_dict.items():
            if len(ep) == 0:
                continue
            data = ep.get_data()  # Shape: (n_epochs, n_channels, n_times)
            n_epochs = data.shape[0]
            
            # Flatten channel x time per epoch into 1D feature vector
            data_2d = data.reshape(n_epochs, -1)
            
            x_list.append(data_2d)
            y_subj_list.extend([subj_id] * n_epochs)

        if x_list:
            tsne_data[cond] = {
                "x": np.vstack(x_list),
                "y_subject": np.array(y_subj_list),
            }
        else:
            tsne_data[cond] = {
                "x": np.empty((0, 0)),
                "y_subject": np.array([]),
            }
    return tsne_data

import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm
from matplotlib.patches import Patch
import numpy as np


def plot_tsne(
    eeg_data,
    title: str,
    seed: int,
    title_fontsize: int = 18,
    legend_fontsize: int = 13,  # Reduced default to fit 74 classes
    subplot_title_fontsize: int = 15,
    axis_label_fontsize: int = 11,
    events_ids=None,
    color_mode: str = "subject",  # "subject" or "event"
):
    fig, axes = plt.subplots(1, 3, figsize=(20, 7))

    # Get unique subject IDs
    subj_ids = sorted(
        {
            int(v)
            for cond in CONDITION_BASE
            for v in eeg_data[cond]["y_subject"]
        }
    )
    n_subjs = len(subj_ids)

  
    if color_mode == "event":
        class_codes = list(events_ids.keys())
        id_to_name = events_ids

        n_classes = len(class_codes)
        code_to_idx = {int(code): i for i, code in enumerate(class_codes)}

        cmap = make_distinct_colormap(n_classes)

        legend_handles = []
        for idx, class_id in enumerate(class_codes):
            name = (
                id_to_name.get(int(class_id), str(int(class_id)))
                if isinstance(id_to_name, dict)
                else str(int(class_id))
            )
            legend_handles.append(
                Patch(
                    facecolor=cmap(idx),
                    edgecolor="none",
                    label=f"{int(class_id)}: {name}",
                )
            )

    elif color_mode == "subject":
        cmap = plt.get_cmap("tab10", n_subjs)
        subj_colors = {subj_id: cmap(i) for i, subj_id in enumerate(subj_ids)}
        legend_handles = [
            Patch(
                facecolor=subj_colors[subj_id],
                edgecolor="none",
                label=f"subj{subj_id}",
            )
            for subj_id in subj_ids
        ]

  
    for ax, cond in zip(axes, CONDITION_BASE):
        x = eeg_data[cond]["x"]
        n_split = x.shape[0]

        z = fit_tsne(x, seed, n_components=2)
        if z is None:
            ax.text(
                0.5, 0.5, f"t-SNE failed in {cond}", ha="center", va="center"
            )
            ax.set_title(
                f"{CONDITION_NAMES[cond]} (n={n_split})",
                fontsize=subplot_title_fontsize,
            )
            ax.set_xticks([])
            ax.set_yticks([])
            continue

        # Generate sample-level color array based on selected color_mode
        if color_mode == "subject":
            y_sub = eeg_data[cond]["y_subject"]
            point_colors = [subj_colors[int(s)] for s in y_sub]
        #else:  # event
            #y_evt = eeg_data[cond]["y_event"]
            #point_colors = [cmap(code_to_idx[int(e)]) for e in y_evt]

        ax.scatter(
            z[:, 0],
            z[:, 1],
            c=point_colors,
            s=14,
            alpha=0.8,
            edgecolors="none",
        )
        ax.set_title(
            f"{CONDITION_NAMES[cond]} (n={n_split})",
            fontsize=subplot_title_fontsize,
        )
        ax.set_xlabel("t-SNE 1", fontsize=axis_label_fontsize)
        ax.set_ylabel("t-SNE 2", fontsize=axis_label_fontsize)
        ax.grid(True, alpha=0.25)

    if color_mode == "event" and len(legend_handles) > 20:
        # place legend outside to the right in multiple columns
        fig.legend(
            handles=legend_handles,
            loc="center left",
            bbox_to_anchor=(1.01, 0.5),
            ncol=3,  # columns
            fontsize=legend_fontsize - 3,
            frameon=False,
        )
        bottom_pad = 0.05
    else:
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
    fig.tight_layout(rect=[0, bottom_pad, 1, 0.95])
    fig.savefig(f"{title}.png", dpi=220, bbox_inches="tight")

    return fig, axes

def plot_tsne_(
    eeg_data,
    title: str,
    seed: int,
    title_fontsize: int = 18,
    legend_fontsize: int = 15,
    subplot_title_fontsize: int = 15,
    axis_label_fontsize: int = 11,
    events_ids = None,
    color_mode: str = "subject"  # or "event"
):
    fig, axes = plt.subplots(1, 3, figsize=(20, 7))

    subj_ids = sorted({int(v) for cond in CONDITION_BASE for v in eeg_data[cond]["y_subject"]})
    n_subjs = len(subj_ids)
    id_to_name = load_event_names('events_codes.csv') if events_ids is None else events_ids
    subj_to_idx = {subj_id: i for i, subj_id in enumerate(subj_ids)}

    if color_mode == "event":
        cmap = make_distinct_colormap(len(events_ids))
        norm = BoundaryNorm(np.arange(-0.5, max(1, len(events_ids)) + 0.5, 1.0), cmap.N)
        legend_handles = []
        for idx, class_id in enumerate(events_ids):
            name = id_to_name.get(int(class_id), str(int(class_id))) if id_to_name else str(int(class_id))
            legend_handles.append(Patch(facecolor=cmap(idx), edgecolor="none", label=f"{int(class_id)}: {name}"))
    elif color_mode == "subject":
        cmap = plt.get_cmap("tab10", n_subjs)
        colors = {subj_id: cmap(i) for i, subj_id in enumerate(subj_ids)}
        legend_handles = [
        Patch(facecolor=colors[subj_id], edgecolor="none", label=f"subj{subj_id}")
                for subj_id in subj_ids
            ]


        for ax, cond in zip(axes, CONDITION_BASE):
            x = eeg_data[cond]["x"]
            n_split = x.shape[0]

            z = fit_tsne(x, seed, n_components=2)
            if z is None:
                ax.text(0.5, 0.5, f"t-SNE failed in {cond}", ha="center", va="center")
                ax.set_title(f"{CONDITION_NAMES[cond]} (n={n_split})", fontsize=subplot_title_fontsize)
                ax.set_xticks([])
                ax.set_yticks([])
                continue


            ax.scatter(
                z[:, 0],
                z[:, 1],
                c=colors,  # Pass precalculated RGB/RGBA matrix directly
                s=14,
                alpha=0.8,
                edgecolors="none",
            )
            ax.set_title(f"{CONDITION_NAMES[cond]} (n={n_split})", fontsize=subplot_title_fontsize)
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
        fig.savefig(f"{title}.png", dpi=220, bbox_inches="tight")
    
    return fig, axes


def main():
    raw_all, markers_all, code_to_name = load_data(subjects, data_dir=DATA_DIR)
    raw_filt = {
        s: raw_all[s].copy().filter(l_freq=COND3_FILTER_BAND[0], h_freq=COND3_FILTER_BAND[1], picks='eeg')
        for s in raw_all
    }

    epochs_all = extract_epochs(raw_all, markers_all)
    epochs_allfilt = extract_epochs(raw_filt, markers_all)

    # Condition-specific epoch dictionaries
    epochs_100_all = epochs_all[1]  # Imagined speech
    epochs_200_all = epochs_all[2]  # Listening
    epochs_300_all = epochs_all[3]  # Attempted speech
    epochs_300_allfilt = epochs_allfilt[3]  # Attempted speech filtered
   
    print(f"\nLoaded and epoched data for {len(raw_all)} subjects")

    events=load_event_names('events_codes.csv')
    formatted_tsne_data = prepare_tsne_data(epochs_all)
    plot_tsne(
        eeg_data=formatted_tsne_data,
        title="Raw EEG data (projected with t-SNE) across conditions",
        seed=3,
        events_ids=events,
        color_mode="subject",  # "event"
    )

    # PLOT EVOKED RESPONSES AND TOPOPLOTS
    # Time points for the topoplots
    # time_points12 = [0.20, 0.25, 0.45, 0.65, 0.85, 1.0, 1.25, 1.65, 2.0]
    # time_points3 = [0.25, 0.45, 0.65, 0.85, 1.0, 1.25, 1.5, 1.75, 2.0, 2.25, 2.5, 2.75]

    # # Create evoked responses (averaged across epochs) for each subject
    # evoked_200_all = {}
    # evoked_100_all = {}
    # evoked_300_all = {}
    # evoked_300_allfilt = {}
    # evoked_372_allfilt = {}

    # for subject in epochs_200_all.keys():
    #     add_evoked_if_nonempty(epochs_200_all, evoked_200_all, subject)

    # for subject in epochs_100_all.keys():
    #     add_evoked_if_nonempty(epochs_100_all, evoked_100_all, subject)

    # for subject in epochs_300_all.keys():
    #     add_evoked_if_nonempty(
    #         epochs_300_all,
    #         evoked_300_all,
    #         subject,
    #         skip_message='Skipping attempted evoked average for subject {subject}: no valid attempted epochs',
    #     )

    # for subject in epochs_300_allfilt.keys():
    #     add_evoked_if_nonempty(
    #         epochs_300_allfilt,
    #         evoked_300_allfilt,
    #         subject,
    #         skip_message='Skipping attempted evoked average for subject {subject}: no valid attempted epochs',
    #     )

    # for subject in epochs_300_allfilt.keys():
    #     stim72_epochs = epochs_300_allfilt[subject]['e472'] if 'e472' in epochs_300_allfilt[subject].event_id else None
    #     if stim72_epochs is not None and len(stim72_epochs) > 0:
    #         evoked_372_allfilt[subject] = stim72_epochs.average()
    

    # print('\n' + '=' * 60)
    # print('EVOKED AVERAGING SUMMARY')
    # print('=' * 60)
    # print_evoked_summary('LISTENING', epochs_200_all, evoked_200_all)
    # print_evoked_summary('IMAGINED SPEECH', epochs_100_all, evoked_100_all)
    # print_evoked_summary('ATTEMPTED SPEECH', epochs_300_all, evoked_300_all)
    # print_evoked_summary('ATTEMPTED SPEECH FILTERED', epochs_300_allfilt, evoked_300_allfilt)
    # print_evoked_summary('ATTEMPTED SPEECH FILTERED (stim 72)', epochs_300_allfilt, evoked_372_allfilt)
    # print('=' * 60 + '\n')

    # # Grand average across all subjects for each condition
    # if len(evoked_200_all) > 0:
    #     total_segments_200 = sum(len(epochs_200_all[subject]) for subject in epochs_200_all.keys())
    #     print(
    #         f'Grand Average LISTENING uses {total_segments_200} total segments across {len(evoked_200_all)} subjects'
    #     )
    #     grand_avg_200 = mne.grand_average(list(evoked_200_all.values()), interpolate_bads=False, drop_bads=True)
    #     grand_avg_200.plot_joint(
    #         times=time_points12,
    #         title=f'Grand Average - LISTENING - N={len(evoked_200_all)} subjects',
    #         show=False,
    #     )
    #     plt.savefig('grand_average_listening.png', dpi=600)

    # if len(evoked_100_all) > 0:
    #     total_segments_100 = sum(len(epochs_100_all[subject]) for subject in epochs_100_all.keys())
    #     print(
    #         f'Grand Average IMAGINED SPEECH uses {total_segments_100} total segments across {len(evoked_100_all)} subjects'
    #     )
    #     grand_avg_100 = mne.grand_average(list(evoked_100_all.values()), interpolate_bads=False, drop_bads=True)
    #     grand_avg_100.plot_joint(
    #         times=time_points12,
    #         title=f'Grand Average - IMAGINED SPEECH - N={len(evoked_100_all)} subjects',
    #         show=False,
    #     )
    #     plt.savefig('grand_average_imagined_speech.png', dpi=600)

    # if len(evoked_300_all) > 0:
    #     total_segments_300 = sum(len(epochs_300_all[subject]) for subject in epochs_300_all.keys())
    #     print(
    #         f'Grand Average ATTEMPTED SPEECH uses {total_segments_300} total segments across {len(evoked_300_all)} subjects'
    #     )
    #     grand_avg_300 = mne.grand_average(list(evoked_300_all.values()), interpolate_bads=False, drop_bads=True)
    #     grand_avg_300.plot_joint(
    #         times=time_points3,
    #         title=f'Grand Average - ATTEMPTED SPEECH - N={len(evoked_300_all)} subjects',
    #         show=False,
    #     )
    #     plt.savefig('grand_average_attempted_speech.png', dpi=600)

    # if len(evoked_300_allfilt) > 0:
    #     total_segments_300f = sum(len(epochs_300_allfilt[subject]) for subject in epochs_300_allfilt.keys())
    #     print(
    #         f'Grand Average ATTEMPTED SPEECH FILTERED uses {total_segments_300f} total segments across {len(evoked_300_allfilt)} subjects'
    #     )

    #     grand_avg_300_filt = mne.grand_average(
    #         list(evoked_300_allfilt.values()),
    #         interpolate_bads=False,
    #         drop_bads=True,
    #     )
    #     grand_avg_300_filt.plot_joint(
    #         times=time_points3,
    #         title=f'Grand Average - ATTEMPTED SPEECH FILTERED - N={len(evoked_300_allfilt)} subjects',
    #         show=False,
    #     )

    #     total_segments_300f = sum(len(epochs_300_allfilt[subject]) for subject in epochs_300_allfilt.keys())
    #     print(
    #         f'Grand Average ATTEMPTED SPEECH FILTERED uses {total_segments_300f} total segments across {len(evoked_300_allfilt)} subjects'
    #     )

    #     grand_avg_372 = mne.grand_average(
    #         list(evoked_300_allfilt.values()),
    #         interpolate_bads=False,
    #         drop_bads=True,
    #     )
    #     grand_avg_372.plot_joint(
    #         times=time_points3,
    #         title=f'Grand Average - ATTEMPTED SPEECH FILTERED - N={len(evoked_300_allfilt)} subjects',
    #         show=False,
    #     )
    #     plt.savefig('grand_average_attempted_speech_0p25_40Hz.png', dpi=600)

    # if len(evoked_372_allfilt) > 0:
    #     total_segments_372 = sum(len(epochs_300_allfilt[subject]['e472']) for subject in evoked_372_allfilt.keys())
    #     print(
    #         f'Grand Average ATTEMPTED SPEECH FILTERED (stim 72) uses {total_segments_372} total segments across {len(evoked_372_allfilt)} subjects'
    #     )
    #     grand_avg_372 = mne.grand_average(list(evoked_372_allfilt.values()), interpolate_bads=False, drop_bads=True)
    #     grand_avg_372.plot_joint(
    #         times=time_points3,
    #         title=f'Grand Average - ATTEMPTED SPEECH FILTERED (stim 72) - N={len(evoked_372_allfilt)} subjects',
    #         show=False,
    #     )
    #     plt.savefig('grand_average_attempted_speech_stim72_0p25_40Hz.png', dpi=600)


if __name__ == '__main__':
    main()
    
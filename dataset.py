import csv
import os
import re
import numpy as np
import torch
from torch.utils.data import Dataset
from typing import Dict, List, Optional, Tuple

epsilon = np.finfo(float).eps


class myDataset(Dataset):
    def __init__(
        self,
        mode,
        data="./",
        task="imagined_speech",
        recon="Y_mel",
        audio_mel_path: Optional[str] = None,
        subjects: Optional[List[int]] = None,
        eeg_csp_path: Optional[str] = None,
        split: Optional[str] = None,
        speech_type: Optional[str] = None,
    ):
        self.sample_rate = 16000
        self.n_classes = 74
        self.mode = mode
        self.iter = iter
        self.savedata = data
        self.task = task
        self.recon = recon
        self.audio_mel_path = audio_mel_path
        self.subjects = [int(s) for s in subjects] if subjects else None
        self.eeg_csp_path = eeg_csp_path
        self.split = split
        self.speech_type = speech_type
        self.max_audio = 32768.0
        self.audio_mels = {}
        self.debug_lengths = os.environ.get("NTDATASET_DEBUG_LENGTHS", "0") == "1"
        self.debug_lengths_max = int(os.environ.get("NTDATASET_DEBUG_LENGTHS_MAX", "200"))
        self._debug_print_count = 0

        self._use_split_layout = bool(self.eeg_csp_path and self.split and self.speech_type and self.subjects)
        if any([self.eeg_csp_path, self.split, self.speech_type, self.subjects]) and not self._use_split_layout:
            raise ValueError(
                "When using the subject/split/speech_type layout, provide eeg_csp_path, split, speech_type, and subjects."
            )

        if self._use_split_layout:
            self.eeg_file_paths, self._resolved_split_dirs = self._collect_eeg_split_file_paths()
            self.eeg_files = [os.path.basename(path) for path in self.eeg_file_paths]
            self.audio_files = self._index_audio_files() if self.audio_mel_path else {}
            if self.audio_mel_path:
                self._load_audio_mels()
            self.lenth = len(self.eeg_file_paths)
            self.lenthtest = self.lenth
            self.lenthval = self.lenth
        else:
            self._resolved_split_dirs = []
            self.eeg_file_paths = []
            self.eeg_files = []
            self.audio_files = {}
            self.lenth = len(os.listdir(self.savedata + '/train/Y/'))
            self.lenthtest = len(os.listdir(self.savedata + '/test/Y/'))
            self.lenthval = len(os.listdir(self.savedata + '/val/Y/'))

    def __len__(self):
        if self._use_split_layout:
            return self.lenth
        if self.mode == 2:
            return self.lenthval
        elif self.mode == 1:
            return self.lenthtest
        else:
            return self.lenth

    def __getitem__(self, idx):
        '''
        :param idx:
        :return:
        '''

        if self._use_split_layout:
            file_name = self.eeg_file_paths[idx]
            input, avg_input, std_input = self._load_feature_data(file_name)

            class_code = self._extract_class_code_from_filename(os.path.basename(file_name))
            if self.audio_mels:
                if class_code not in self.audio_mels:
                    raise KeyError(f"No audio mel file cached for class code {class_code}.")
                target, avg_target, std_target = self._normalize_array(self.audio_mels[class_code])
            else:
                target = input.copy()
                avg_target, std_target = avg_input, std_input

            target_cl = np.asarray([class_code], dtype=np.float32)

        else:
            if self.mode == 2:
                forder_name = self.savedata + '/val/'
            elif self.mode == 1:
                forder_name = self.savedata + '/test/'
            else:
                forder_name = self.savedata + '/train/'

            # tasks
            allFileList = os.listdir(forder_name + self.task + "/")
            allFileList.sort()
            file_name = forder_name + self.task + '/' + allFileList[idx]

            if self.task.find('mel') != -1:
                input, avg_input, std_input = self.read_data(file_name)
            else:  # EEG
                input, avg_input, std_input = self.read_data(file_name)

            # recon target
            allFileList = os.listdir(forder_name + self.recon + "/")
            allFileList.sort()
            file_name = forder_name + self.recon + '/' + allFileList[idx]

            if self.recon.find('mel') != -1:
                target, avg_target, std_target = self.read_data(file_name)
            else:  # EEG
                target, avg_target, std_target = self.read_data(file_name)

            # target label
            allFileList = os.listdir(forder_name + "Y/")
            allFileList.sort()
            file_name = forder_name + 'Y/' + allFileList[idx]

            target_cl, _, _ = self.read_raw_data(file_name)
            target_cl = np.squeeze(target_cl)

        # to tensor
        input = torch.tensor(input, dtype=torch.float32)
        target = torch.tensor(target, dtype=torch.float32)
        target_cl = torch.tensor(target_cl, dtype=torch.float32)

        if self.debug_lengths and self._debug_print_count < self.debug_lengths_max:
            self._log_sequence_lengths(idx, file_name, input, target, target_cl)
            self._debug_print_count += 1

        return input, target, target_cl, (avg_target, std_target, avg_input, std_input)

    def _log_sequence_lengths(self, idx, file_name, input_tensor, target_tensor, target_cl_tensor):
        def _seq_len(tensor):
            if tensor.ndim == 0:
                return 1
            return int(tensor.shape[0])

        cls_text = "unknown"
        try:
            cls_val = int(float(target_cl_tensor.view(-1)[0].item()))
            cls_text = str(cls_val)
        except Exception:
            pass

        print(
            "[NTDataset] idx={} file={} class={} in_len={} in_shape={} tgt_len={} tgt_shape={}".format(
                idx,
                os.path.basename(str(file_name)),
                cls_text,
                _seq_len(input_tensor),
                tuple(input_tensor.shape),
                _seq_len(target_tensor),
                tuple(target_tensor.shape),
            )
        )

    
    def _collect_eeg_split_file_paths(self) -> Tuple[List[str], List[str]]:
        """Discover EEG feature files (.npy/.csv) from: <root>/subjXX/<condition>/<split>."""
        candidate_dirs: List[str] = []

        if not self.subjects:
            raise ValueError("subjects must be provided (e.g., [16, 17]).")
        if not self.speech_type:
            raise ValueError(
                "speech_type must be provided (e.g., imagined_speech or attempted_speech)."
            )

        for subject_id in self.subjects:
            split_dir = os.path.join(
                self.eeg_csp_path,
                f"subj{subject_id}",
                self.speech_type,
                self.split,
            )
            if os.path.isdir(split_dir):
                candidate_dirs.append(split_dir)

        if not candidate_dirs:
            raise FileNotFoundError(
                f"Could not resolve split directory for split '{self.split}' under '{self.eeg_csp_path}'. "
                f"Expected '<path>/subjXX/{self.speech_type}/{self.split}/'."
            )

        file_paths: List[str] = []
        for eeg_dir in candidate_dirs:
            for fname in sorted(os.listdir(eeg_dir)):
                if fname.lower().endswith(".npy") or fname.lower().endswith(".csv"):
                    file_paths.append(os.path.join(eeg_dir, fname))

        return file_paths, candidate_dirs

    def _index_audio_files(self) -> Dict[int, str]:
        if not self.audio_mel_path:
            return {}

        if not os.path.exists(self.audio_mel_path):
            raise FileNotFoundError(f"Audio mel directory not found: {self.audio_mel_path}")

        audio_files: Dict[int, str] = {}
        for audio_file in sorted(os.listdir(self.audio_mel_path)):
            if not (audio_file.endswith('_logmel.csv') or audio_file.endswith('_mel.csv')):
                continue
            stem = audio_file.replace('audio', '')
            stem = stem.replace('_logmel.csv', '').replace('_mel.csv', '')
            audio_idx = int(stem)
            audio_files[audio_idx] = os.path.join(self.audio_mel_path, audio_file)
        return audio_files
    
    def _load_audio_mels(self):
        """Load all audio mel files into memory cache."""
        self.audio_mels = {}
        for audio_idx, audio_path in self.audio_files.items():
            self.audio_mels[audio_idx] = self._read_csv_data(audio_path)
            #mel = self._read_csv_data(audio_path)
            #self.audio_mels[audio_idx] = mel
            #print(f"[NTDataset] loaded audio mel idx={audio_idx} file={os.path.basename(audio_path)} shape={mel.shape}")

    @staticmethod
    def _read_csv_data(file_name):
        with open(file_name, 'r', newline='') as f:
            lines = csv.reader(f)
            data = []
            for line in lines:
                data.append(line)

        return np.array(data).astype(np.float32)

    @staticmethod
    def _extract_class_code_from_filename(file_name: str) -> int:
        base_name = os.path.basename(file_name).replace('.csv', '').replace('.npy', '')
        match = re.search(r'label(\d+)', base_name, flags=re.IGNORECASE)
        if match:
            return int(match.group(1))

        match = re.search(r'class[_-]?(\d+)', base_name, flags=re.IGNORECASE)
        if match:
            return int(match.group(1))

        numeric_parts = re.findall(r'\d+', base_name)
        if numeric_parts:
            return int(numeric_parts[0])

        raise ValueError(f"Cannot parse class code from filename: {file_name}")

    def _load_feature_data(self, file_name: str):
        if file_name.lower().endswith('.npy'):
            data = np.load(file_name).astype(np.float32)
        else:
            data = self._read_csv_data(file_name)
        return self._normalize_array(data)

    def _normalize_array(self, data: np.ndarray):
        max_ = np.max(data).astype(np.float32)
        min_ = np.min(data).astype(np.float32)
        avg = (max_ + min_) / 2
        std = max((max_ - min_) / 2, epsilon)
        normalized = np.array((data - avg) / std).astype(np.float32)
        return normalized, avg, std
            
    def read_vector_data(self, file_name,n_classes):
        data = self._read_csv_data(file_name)
        (r,c) = data.shape
        data = np.reshape(data,(n_classes,r//n_classes,c))
        
        data, avg, std = self._normalize_array(data)

        return data, avg, std
    

    def read_data(self, file_name):
        data = self._read_csv_data(file_name)
        return self._normalize_array(data)


    def read_raw_data(self, file_name):
        data = self._read_csv_data(file_name)
        avg = np.array([0]).astype(np.float32)
        std = np.array([1]).astype(np.float32)

            
        return data, avg, std



# EEG2speech
This repository contains the code used at all stages of my Master Thesis project with the final aim of of generating Mel speech spectrograms from EEG features extracted from data recorderd with a 32 channels setting on the task of imagined and attempted speech.

The data collection paradigm is custom and it includes 3 tasks in total: listening, imagined speech and attempted speech during or after the presentation of same stimulus/word, presented as synthetic audios (listening) and text on screen (while imagining and before attempting speech).

Different models are implied to perform the main task --> models\

The main experiment in train.py runs in a GAN framework with 2 dilated convolutional networks for the Generator and the Discriminator respectively outputting and taking as input full Mel spectrograms, leveraging the power of 2 pre-trained models: one for vocoding (Hi-fi GAN Universal V1[^2]) and one for wavs embedding and Automatic Speech Recognition (wav2vec-base960hours[^3])

The main GAN framework and the approach is directly inspired by [^1].

The EEG features used to train the models have been extracted training Common Spatial Filters (CSP) in Python on imagined and attempted speech with splits train/val/test, keeping test out from filters' training. The final data used for training the GAN are just trials from imagined speech, passed with the filters mentioned above.

## Train

The project is run in a Python venv with Python 3.11.9. CUDA V12.1.66 for parallel computing.
The model is trained in parallel on 1 GPU (NVIDIA GeForce GTX 1650 with Max-Q Design) and 1 CPU (AMD Ryzen 7 5800HS, 8 cores, 16 threads). One epoch with all the gradients flowing through all modules takes approx 1000 s.

### Requirements

```
pip install requirements-minimal.txt

```

### To Train:

```
python train.py

```

Default Arguments and details

```
    parser.add_argument('--max_epochs', type=int, default=1000)
    parser.add_argument('--vocoder_pre', type=str, default='UNIVERSAL_V1/g_02500000', help='pretrained vocoder file path')
    parser.add_argument('--vocoder_type', type=str, default='hifigan', choices=['hifigan', 'hifigan_16k', 'griffinlim'], help='vocoder backend to synthesize waveform from mel')
    parser.add_argument('--trained_model', type=str, default=None, help='trained model for G & D folder path')
    parser.add_argument('--model_config', type=str, default='./models', help='config for G & D folder path')
    parser.add_argument('--dataLoc', type=str, default=dataDir)
    parser.add_argument('--eeg_csp_path', type=str, default=dataDir, help='root EEG directory (contains subjXX folders)')
    parser.add_argument('--audio_mel_path', type=str, default=audioDir, help='audio mel spectrograms directory with audio*_logmel.csv')
    parser.add_argument('--audio_wav_path', type=str, default=audioWavDir, help='audio waveform directory with audio*.wav')
    parser.add_argument('--config', type=str, default='./config_params.json')
    parser.add_argument('--logDir', type=str, default=logDir)
    parser.add_argument('--resume', type=bool, default=True)
    parser.add_argument('--pretrain', type=bool, default=False)
    parser.add_argument('--prefreeze', type=bool, default=False)
    parser.add_argument('--gpuNum', nargs='+', type=int, default=[0])
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--debug_batch_trace', type=int, choices=[0, 1], default=0, help='Print the last successful training stage for each batch')
    parser.add_argument('--debug_cuda_sync', type=int, choices=[0, 1], default=0, help='Synchronize CUDA after each debug stage to localize silent kernel failures')
    parser.add_argument('--debug_cuda_memory', type=int, choices=[0, 1], default=0, help='Print allocated and reserved CUDA memory at each debug stage')
    parser.add_argument('--sub', nargs='+', type=int, default=[18])
    parser.add_argument('--task', type=str, default='imagined_speech')

    parser.add_argument('--stt_chunk_size', type=int, default=4)
    parser.add_argument('--stt_backbone', type=str, default='wav2vecFT', choices=['wav2vecFT','wav2vec2_base_960h', 'wav2vec2_large_960h', 'hubert_large'])
    parser.add_argument('--ctc_device', type=str, default='cpu', choices=['cuda', 'cpu'], help='device for vocoder+STT+CTC branch')
    parser.add_argument('--cer_every_n_batches', type=int, default=4)
    parser.add_argument('--compute_cer_in_val', type=bool, default=False)

```




[^1]: Towards Voice Reconstruction from EEG during Imagined Speech, AAAI Conference on Artificial Intelligence (AAAI). *Y.-E. Lee, S.-H. Lee, S.-H Kim, and S.-W. Lee*. 2023.  
[Paper|GitHub](https://arxiv.org/abs/2301.07173 , https://github.com/youngeun1209/NeuroTalk)
[^2]: HiFi-GAN: Generative Adversarial Networks for Efficient and High Fidelity Speech Synthesis. *Jungil Kong, Jaehyeon Kim, Jaekyoung Bae*. 2020.  [Paper|GitHub] (https://arxiv.org/abs/2010.05646 , https://github.com/jik876/hifi-gan)  
[^3]: wav2vec 2.0: A Framework for Self-Supervised Learning of Speech Representations. *Alexei Baevski, Henry Zhou, Abdelrahman Mohamed, Michael Auli*. 2020.  
[Paper] (https://arxiv.org/abs/2006.11477)

To add line breaks within a footnote, add 2 spaces to the end of a line.  
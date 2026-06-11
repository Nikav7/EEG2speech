import numpy as np
from scipy.signal import lfilter, filtfilt
from scipy.stats import norm

class HEAR:
    def __init__(self, fs, D, is_causal=True, mag_art_mu=3.0, mag_est_win=0.25, epsilon=1e-8):
        self.fs = fs
        self.is_causal = is_causal
        self.mag_art_mu = mag_art_mu
        self.mag_est_win = mag_est_win
        self.D = D
        self.epsilon = epsilon  # Small constant to avoid division by zero

        # Compute exponential smoothing factor so that 'mag_est_win * fs' famples have 'p' % of the weights 
        p = 0.9
        self.exp_lambda = (1 - p) ** (1 / (self.mag_est_win * fs))

        self.state_havg = None
        self.state_ref_mag = None
        self.mag_art_sigma = 1.0

    def train(self, data):
        n_chans, _ = data.shape
        #bi_data = data.reshape(-1, train_data.shape[1] * train_data.shape[2])
        # Create averaging filter based on exponential smoothing factor
        b = [1 - self.exp_lambda]
        a = [1, -self.exp_lambda]

        if not self.is_causal:
            # get the envelope of the error-Signal (smoothed)
            # Forward filter
            data_mag = lfilter(b, a, data ** 2, axis=1)
            # Backward filter (zero-phase distortion)
            data_mag = np.sqrt(np.flip(lfilter(b, a, np.flip(data_mag, axis=1), axis=1), axis=1))
        else:
            # Forward filter #causal #get the envelope of the signal (smoothed)
            data_mag = np.sqrt(lfilter(b, a, data ** 2, axis=1))

        self.state_ref_mag = np.mean(data_mag, axis=1)
        self.state_havg = (b, a)

    def apply(self, data):
        n_chans, _ = data.shape

        # Check if the model is calibrated to the correct number of channels
        if self.state_ref_mag is None or n_chans != self.state_ref_mag.shape[0]:
           raise ValueError("Model is not trained for the correct number of channels")

        b, a = self.state_havg
        if self.is_causal:
            data_mag = np.sqrt(lfilter(b, a, data ** 2, axis=1))
        else:
            data_mag = lfilter(b, a, data ** 2, axis=1)
            data_mag = np.sqrt(np.flip(lfilter(b, a, np.flip(data_mag, axis=1), axis=1), axis=1))

        ths_mu = self.state_ref_mag * self.mag_art_mu

        # Avoid division by zero by adding epsilon
        x = (data_mag - ths_mu[:, None]) / (self.mag_art_sigma * (self.state_ref_mag[:, None] + self.epsilon))

        # Query the cdf (comulative distribution function) of a Gaussian distribution
        p_art_ext = norm.cdf(x)
        p_art = np.max(p_art_ext, axis=0)

        varargout = [p_art]

        if np.isnan(self.D).any() or not self.D.ndim == 2 or self.D.shape[0] != self.D.shape[1] or n_chans != self.D.shape[0]:
            if len(varargout) > 1:
                raise ValueError('Expecting corrected output without a valid channel interpolation matrix.')
        else:
            # estimate the probability that an artifact contaminated channel cannot be corrected by its neighbors (confidence)
            p_art_ext_D = self.D @ p_art_ext
            varargout.append(np.max(p_art_ext * p_art_ext_D, axis=1))

            # Correction step (corrected data)
            data_c = p_art_ext * (self.D @ data) + (1 - p_art_ext) * data
            varargout.append(data_c)

        #return varargout if len(varargout) > 1 else varargout[0]
        return varargout
    
import librosa
import numpy as np
from scipy import signal


# These pre-processing functions are referred from https://github.com/keithito/tacotron

_mel_basis = None

NUM_MELS = 80
NUM_FREQ = 1024
SAMPLE_RATE = 20000
FRAME_LENGTH_MS = 50.
FRAME_SHIFT_MS = 12.5
PREEMPHASIS = 0.97
MIN_LEVEL_DB = -100
REF_LEVEL_DB = 20

MAX_ITERS = 200
GRIFFIN_LIM_ITERS = 60
POWER = 1.5


def load_wav(path):
    return np.asarray(librosa.load(path, sr=SAMPLE_RATE)[0], dtype=np.float32)


def save_wav(wav, path):
    wav *= 32767 / max(0.01, np.max(np.abs(wav)))
    librosa.output.write_wav(path, wav.astype(np.int16), SAMPLE_RATE)


def _linear_to_mel(spectrogram):
    global _mel_basis
    if _mel_basis is None:
        _mel_basis = _build_mel_basis()
    return np.dot(_mel_basis, spectrogram)


def _build_mel_basis():
    n_fft = (NUM_FREQ - 1) * 2
    return librosa.filters.mel(SAMPLE_RATE, n_fft, n_mels=NUM_MELS)


def _normalize(S):
    return np.clip((S - MIN_LEVEL_DB) / -MIN_LEVEL_DB, 0, 1)


def _denormalize(S):
    return (np.clip(S, 0, 1) * -MIN_LEVEL_DB) + MIN_LEVEL_DB


def _stft_parameters():
    n_fft = (NUM_FREQ - 1) * 2
    hop_length = int(FRAME_SHIFT_MS / 1000 * SAMPLE_RATE)
    win_length = int(FRAME_LENGTH_MS / 1000 * SAMPLE_RATE)
    return n_fft, hop_length, win_length


def _amp_to_db(x):
    return 20 * np.log10(np.maximum(1e-5, x))


def _db_to_amp(x):
    return np.power(10.0, x * 0.05)


def preemphasis(x):
    return signal.lfilter([1, -PREEMPHASIS], [1], x)


def inv_preemphasis(x):
    return signal.lfilter([1], [1, -PREEMPHASIS], x)


def spectrogram(y):
    D = _stft(preemphasis(y))
    S = _amp_to_db(np.abs(D)) - REF_LEVEL_DB
    return _normalize(S)


def inv_spectrogram(spectrogram):
    '''Converts spectrogram to waveform using librosa'''

    S = _denormalize(spectrogram)
    S = _db_to_amp(S + REF_LEVEL_DB)  # Convert back to linear

    return inv_preemphasis(_griffin_lim(S ** POWER))  # Reconstruct phase


def _griffin_lim(S):
    '''librosa implementation of Griffin-Lim
    Based on https://github.com/librosa/librosa/issues/434
    '''
    angles = np.exp(2j * np.pi * np.random.rand(*S.shape))
    S_complex = np.abs(S).astype(np.complex)
    y = _istft(S_complex * angles)
    for i in range(GRIFFIN_LIM_ITERS):
        angles = np.exp(1j * np.angle(_stft(y)))
        y = _istft(S_complex * angles)
    return y


def _istft(y):
    _, hop_length, win_length = _stft_parameters()
    return librosa.istft(y, hop_length=hop_length, win_length=win_length)


def melspectrogram(y):
    D = _stft(preemphasis(y))
    S = _amp_to_db(_linear_to_mel(np.abs(D)))
    return _normalize(S)


def _stft(y):
    n_fft, hop_length, win_length = _stft_parameters()
    return librosa.stft(y=y, n_fft=n_fft, hop_length=hop_length, win_length=win_length)


def find_endpoint(wav, threshold_db=-40, min_silence_sec=0.8):
    window_length = int(SAMPLE_RATE * min_silence_sec)
    hop_length = int(window_length / 4)
    threshold = _db_to_amp(threshold_db)
    for x in range(hop_length, len(wav) - window_length, hop_length):
        if np.max(wav[x:x + window_length]) < threshold:
            return x + hop_length
    return len(wav)

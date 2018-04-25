import io
import os

import numpy as np
import torch
from torch.utils.data import Dataset

from audio import find_endpoint
from audio import inv_spectrogram, save_wav


class AudioDataset(Dataset):
    def __init__(self, path, args):
        self.args = args
        self.path = path
        self.fnames = []

        for f in os.listdir(path):
            if '-spec-' in f:
                self.fnames.append(f)

    def __len__(self):
        return len(self.fnames)

    def __getitem__(self, idx):
        spec = np.load(os.path.join(self.path, self.fnames[idx]))
        return spec


def collate_fn_audio(batch):
    len_lists = [len(elt) for elt in batch]
    max_lens = max(len_lists)
    padded = []
    for d in batch:
        org_shape = list(d.shape[:])
        pad_shape = list(d.shape[:])
        pad_shape[0] = max_lens - org_shape[0]

        padding = np.zeros(pad_shape, dtype=np.float32)
        padded.append(np.concatenate((d, padding), axis=0))

    return torch.from_numpy(np.stack(padded, axis=0).astype(np.float32))


def audio_writer(dir_path, spec_output, name_head, use_cuda=True):
    for file_idx, a_spec_output in enumerate(spec_output):
        if file_idx > 20:
            break

        if use_cuda:
            nparr = a_spec_output.cpu().data.numpy()
        else:
            nparr = a_spec_output.data.numpy()

        wav = inv_spectrogram(nparr)
        wav = wav[:find_endpoint(wav)]
        out = io.BytesIO()
        save_wav(wav, out)

        path = os.path.join(dir_path, '%s_%d.wav' % (name_head, file_idx))
        with open(path, 'wb') as f:
            f.write(out.getvalue())
        f.close()
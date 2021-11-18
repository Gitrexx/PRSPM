import torch
import collections
import os
import soundfile as sf
import librosa
from torch.utils.data import DataLoader, Dataset
import numpy as np
from joblib import Parallel, delayed

LOGICAL_DATA_ROOT = './LA'
PHISYCAL_DATA_ROOT = './PA'
COMBINE_DATA_ROOT = 'test_materials'


ASVFile = collections.namedtuple('ASVFile',
                                 ['speaker_id', 'file_name', 'path', 'sys_id', 'key'])


class ASVDataset(Dataset):
    """ Utility class to load  train/dev datatsets """

    def __init__(self, transform=None,is_train=True, sample_size=None,is_logical=True, feature_name=None, is_eval=False,eval_part=0):
        if is_logical:
            data_root = COMBINE_DATA_ROOT
            track = 'SA'
        else:
            data_root = PHISYCAL_DATA_ROOT
            track = 'PA'

        if is_eval:
            # data_root = os.path.join('eval_data', data_root)
            print("data_root:", data_root)
        assert feature_name is not None, 'must provide feature name'
        self.track = track
        self.is_logical = is_logical
        self.prefix = 'ASVspoof2019_{}'.format(track)
        self.sysid_dict = {
            '-': 0,  # bonafide speech
            'SS_1': 1,  # Wavenet vocoder
            'SS_2': 2,  # Conventional vocoder WORLD
            'SS_4': 3,  # Conventional vocoder MERLIN
            'US_1': 4,  # Unit selection system MaryTTS
            'VC_1': 5,  # Voice conversion using neural networks
            'VC_4': 6,  # transform function-based voice conversion
            # For PA:
            'AA': 7,
            'AB': 8,
            'AC': 9,
            'BA': 10,
            'BB': 11,
            'BC': 12,
            'CA': 13,
            'CB': 14,
            'CC': 15,
            'A01': 16,
            'A02': 17,
            'A03': 18,
            'A04': 19,
            'A05': 20,
            'A06': 21,
            'A07': 22,
            'A08': 23,
            'A09': 24,
            'A10': 25,
            'A11': 26,
            'A12': 27,
            'A13': 28,
            'A14': 29,
            'A15': 30,
            'A16': 31,
            'A17': 32,
            'A18': 33,
            'A19': 34,
        }
        self.is_eval = is_eval
        self.sysid_dict_inv = {v: k for k, v in self.sysid_dict.items()}
        self.data_root = data_root
        self.dset_name = 'eval' if is_eval else 'train' if is_train else 'dev'
        self.protocols_fname = 'eval.trl' if is_eval else 'train.trn' if is_train else 'dev.trl'
        self.protocols_dir = os.path.join(self.data_root,
                                          '{}_cm_protocols/'.format(self.prefix))
        self.files_dir = os.path.join(self.data_root, '{}_{}'.format(
            self.prefix, self.dset_name), 'flac')
        self.protocols_fname = os.path.join(self.protocols_dir,
                                            'ASVspoof2019.{}.cm.{}.txt'.format(track,
                                                                               self.protocols_fname))  # 'ASVspoof2019.{PA}.cm.{eval.trl}.txt'
        self.cache_fname = 'cache_{}_{}_{}_{}.npy'.format(self.dset_name, track, feature_name,eval_part) if is_eval else 'cache_{}_{}_{}_{}.npy'.format(self.dset_name, track, feature_name, part)
        self.transform = transform
        if os.path.exists(self.cache_fname):
            print("data exist")
            self.data_x, self.data_y, self.data_sysid, self.files_meta = torch.load(self.cache_fname)
            print('Dataset loaded from cache ', self.cache_fname)
        else:
            self.files_meta = self.parse_protocols_file(self.protocols_fname)
            self.data = list(map(self.read_file, self.files_meta))
            self.data_x, self.data_y, self.data_sysid = map(list, zip(*self.data))
            if self.transform:
                self.data_x = Parallel(n_jobs=4, prefer='threads')(delayed(self.transform)(x) for x in self.data_x)
            torch.save((self.data_x, self.data_y, self.data_sysid, self.files_meta), self.cache_fname)
            print('Dataset saved to cache ', self.cache_fname)
        self.length = len(self.data_x)  # 29700
        print("length", self.length)

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        x = self.data_x[idx]
        y = self.data_y[idx]
        return x, y, self.files_meta[idx]

    def read_file(self, meta):
        data_x, sample_rate = sf.read(meta.path)
        data_y = meta.key
        return data_x, float(data_y), meta.sys_id

    def _parse_line(self, line):
        tokens = line.strip().split(' ')

        return ASVFile(speaker_id=tokens[0],
                       file_name=tokens[1],
                       path=os.path.join(self.files_dir, tokens[1] + '.flac'),
                       sys_id=self.sysid_dict[tokens[3]],
                       key=int(tokens[4] == 'bonafide'))

    def parse_protocols_file(self, protocols_fname):
        lines = open(protocols_fname).readlines()
        files_meta = map(self._parse_line, lines)
        return list(files_meta)


if __name__ == '__main__':
    dev_set = ASVDataset(is_eval=True, is_logical=False, transform=None,
                         feature_name='spect', part=4)



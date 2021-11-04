import argparse
import sys
import os

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import DataLoader

import librosa
import random


from models import ResNet_18
from torch import nn
# from tensorboardX import SummaryWriter
import soundfile as sf
import librosa



# from scipy.optimize import brentq
# from scipy.interpolate import interp1d
# from sklearn.metrics import roc_curve

model_path = 'SA.pth'
data_path = 'test_materials/'
testfile = 'zuiniub.flac'


os.environ['CUDA_VISIBLE_DEVICES'] = '0'

def setseed():
    print("set seed ok")
    seed = 0
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def pad(x, max_len=64000):
    x_len = x.shape[0]
    if x_len >= max_len:
        return x[:max_len]
    # need to pad
    num_repeats = (max_len / x_len)+1
    x_repeat = np.repeat(x, num_repeats)
    padded_x = x_repeat[:max_len]
    return padded_x


def get_log_spectrum(x):
    s = librosa.core.stft(x, n_fft=2048, win_length=2048, hop_length=512)
    a = np.abs(s)
    feat = librosa.power_to_db(a)
    return feat

def evaluate(dataset,model,device):
    data_loader = DataLoader(dataset,batch_size=1, shuffle=False)
    model.eval()
    score_list = []
    for batch_x in data_loader:
        batch_x = batch_x.to(device)
        batch_out = model(batch_x)
        score_list.extend(batch_out.tolist())
    return score_list

def ASVDataset(filepath):
    data_x, sample_rate = sf.read(filepath)
    # number_of_samples = round(len(data_x) * float(16000) / sample_rate)
    # data_x = sps.resample(data_x, number_of_samples)
    print(data_x.shape)
    # data_x = data_x[:,0]
    data_x = pad(data_x)
    print(data_x.shape)
    data_x = librosa.util.normalize(data_x)
    data_x = get_log_spectrum(data_x)
    print(sample_rate)
    return Tensor(data_x).unsqueeze(dim=0)

def spoofdetect(filepath,modelpath):
    setseed()

    model_cls = ResNet_18
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # device = 'cuda'
    model = model_cls().to(device)
    model.load_state_dict(torch.load(modelpath,map_location=torch.device('cpu')))
    dataset = ASVDataset(filepath)
    return evaluate(dataset,model,device)[0]

def spoofdetect_stack(filepath,model1_path,model2_path):
    setseed()
    model_cls = ResNet_18
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model1 = model_cls().to(device)
    model2 = model_cls().to(device)
    model1.load_state_dict(torch.load(model1_path, map_location=torch.device('cpu')))
    model2.load_state_dict(torch.load(model2_path, map_location=torch.device('cpu')))
    dataset = ASVDataset(filepath)
    temp = evaluate(dataset,model1,device)[0]
    result = [0,0]
    if temp[0]>temp[1]:
        result[0] = temp[0]
    else:
        temp = evaluate(dataset,model2,device)[0]
        if temp[0]>temp[1]:
            result[0] = max(temp[0],result[0])
        else:
            result[1]=temp[1]
    return result






# if __name__ == '__main__':
#     setseed()
#
#     model_cls = ResNet_18
#
#     print(torch.cuda.is_available())
#     device = 'cuda' if torch.cuda.is_available() else 'cpu'
#     # device = 'cuda'
#     print(device)
#     model = model_cls().to(device)
#     model.load_state_dict(torch.load(model_path,map_location=torch.device('cpu')))
#     dataset = ASVDataset(data_path, testfile)
#     print(evaluate(dataset,model,device))

    # for file in filelist:
    #     dataset = ASVDataset(data_path, file)
    #     print(file,evaluate(dataset, model, device)[0][1])
import argparse
import sys
import os
import numpy as np
import torch
from torch import Tensor
from torch.utils.data import DataLoader
from utils.DataAugmentation import *
import librosa
import random
import matplotlib.pyplot as plt
from utils.model import ResNet_18
import soundfile as sf
import librosa
import scipy.signal as sps



model_path = ''
data_path = ''
testfile = ''


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

    data_x = volume_augment(data_x)
    data_x = pad(data_x)
    data_x = librosa.util.normalize(data_x)
    data_x = get_log_spectrum(data_x)
    return Tensor(data_x).unsqueeze(dim=0)

def spoofdetect(filepath,modelpath):
    setseed()
    model_cls = ResNet_18
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = model_cls().to(device)
    model.load_state_dict(torch.load(modelpath))
    dataset = ASVDataset(filepath)
    return evaluate(dataset,model,device)[0]

def spoofdetect_voting(filepath,modelpath_PA,modelpath_LA, modelpath_SA):
    setseed()
    model_cls_PA = ResNet_18
    model_cls_LA = ResNet_18
    model_cls_SA = ResNet_18
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # device = 'cuda'
    model_PA = model_cls_PA().to(device)
    model_LA = model_cls_LA().to(device)
    model_SA = model_cls_SA().to(device)
    model_PA.load_state_dict(torch.load(modelpath_PA))
    model_LA.load_state_dict(torch.load(modelpath_LA))
    model_SA.load_state_dict(torch.load(modelpath_SA))
    dataset = ASVDataset(filepath)
    return evaluate_volting(dataset,model_PA,model_SA,model_LA,device)[0]
    # return evaluate(dataset,model,device)[0]

def evaluate_volting(dataset,model_PA,model_SA,model_LA,device):
    data_loader = DataLoader(dataset,batch_size=1, shuffle=False)
    model_PA.eval()
    model_LA.eval()
    model_SA.eval()
    score_list = []
    for batch_x in data_loader:
        batch_x = batch_x.to(device)
        batch_out_PA = model_PA(batch_x)
        batch_out_LA = model_LA(batch_x)
        batch_out_SA = model_SA(batch_x)

        batch_score_s = batch_out_SA

        batch_score_p = batch_out_PA

        batch_score_l = batch_out_LA

        # batch_score_s = (batch_out_SA[:, 1] - batch_out_SA[:, 0]
        #                ).data.cpu().numpy().ravel()
        #
        # batch_score_p = (batch_out_PA[:, 1] - batch_out_PA[:, 0]
        #                ).data.cpu().numpy().ravel()
        #
        # batch_score_l = (batch_out_LA[:, 1] - batch_out_LA[:, 0]
        #                ).data.cpu().numpy().ravel()

        _, batch_pred_p = batch_out_PA.max(dim=1)
        _, batch_pred_l = batch_out_LA.max(dim=1)
        _, batch_pred_s = batch_out_SA.max(dim=1)

        print("result:",batch_pred_s,batch_pred_l,batch_pred_p)

        if batch_pred_s == batch_pred_p and batch_pred_p == batch_pred_l:
          batch_score = (batch_score_s + batch_score_l + batch_score_p)/3
        elif batch_pred_s == batch_pred_p:
          batch_score = (batch_score_p + batch_score_s)/2
        elif batch_pred_s == batch_pred_l:
          batch_score = (batch_score_l + batch_score_s)/2
        elif batch_score_p == batch_score_l:
          batch_score = (batch_score_p+batch_score_l)/2

        score_list.extend(batch_score.tolist())
    return score_list



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
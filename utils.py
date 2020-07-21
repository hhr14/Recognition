import torch
import os
from model import xvecTDNN
import random
import numpy as np
import librosa


def load_data(hparams):
    f = open(hparams.train_txt_dir, 'r')
    mel_list = []
    language_id = []
    time = []
    while True:
        line = f.readline()
        if line != '':
            mel_list.append(line.split(' ')[0])
            language_id.append(line.split(' ')[1])
            time.append(line.split(' ')[2])
        else:
            break
    size = len(mel_list)
    index = [i for i in range(size)]
    random.shuffle(index)
    mel_list = np.array(mel_list)[index]
    language_id = np.array(language_id)[index]
    time = np.array(time)[index]
    return [mel_list[:int(size * 0.9)], language_id[:int(size * 0.9)], time[:int(size * 0.9)]],\
           [mel_list[int(size * 0.9):], language_id[int(size * 0.9):], time[int(size * 0.9):]]


def load_recent_model(model_path):
    """
    返回最近的权重文件
    :param path:
    :return:
    """
    model_list = os.listdir(model_path)
    recent_epoch = -1
    recent_file = None
    for model_file in model_list:
        epoch = int(((model_file.split('_')[-1]).split('-')[0])[1:])
        if epoch > recent_epoch:
            recent_epoch = epoch
            recent_file = model_file
    print('recent file', recent_file)
    if recent_file is None:
        return None
    else:
        return os.path.join(model_path, recent_file), recent_epoch


def load_best_model(model_path, hparams):
    """
    返回acc最好的权重文件
    :param path:
    :return:
    """
    model_list = os.listdir(model_path)
    best_acc = 0
    best_file = None
    for model_file in model_list:
        model_acc = float((model_file.split('-')[-1])[:-3])
        if hparams.load_epoch is not None:
            epoch = int(((model_file.split('_')[-1]).split('-')[0])[1:])
            if epoch != hparams.load_epoch:
                continue
        if model_acc > best_acc:
            best_acc = model_acc
            best_file = model_file
    print('best_file', best_file)
    return os.path.join(model_path, best_file)


def create_model(hparams, mode='train'):
    if hparams.model == 'xvecTDNN':
        mymodel = xvecTDNN(hparams)
    else:
        raise ValueError('model not exist!')

    if mode == 'train':
        if load_recent_model(hparams.model_save_path) is not None:
            model_path, recent_epoch = load_recent_model(hparams.model_save_path)
            checkpoint = torch.load(model_path)
            if hparams.model == 'xvecTDNN':
                mymodel.load_state_dict(checkpoint['model_state'])
            return mymodel, recent_epoch
        else:
            return mymodel, 0
    else:
        best_file_path = load_best_model(hparams.model_save_path, hparams)
        checkpoint = torch.load(best_file_path)
        if hparams.model == 'xvecTDNN':
            mymodel.load_state_dict(checkpoint['model_state'])
        return mymodel


def get_predict_file_list(predict_path):
    if os.path.isfile(predict_path):
        return [predict_path]
    else:
        result = []
        predict_folder = os.listdir(predict_path)
        predict_folder.sort()
        for predict_file in predict_folder:
            result.append(os.path.join(predict_path, predict_file))
        return result


def feature_extract(wav_path, mode='mel_spectrum'):
    if mode == 'mel_spectrum':
        pass
    else:
        raise ValueError('Unlegal feature extracting method!')


def get_melspectrum(hparams, wav_path):
    try:
        y, sr = librosa.load(wav_path)
    except Exception:
        print('Error open', wav_path)
        return None
    else:
        mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=hparams.n_mels, n_fft=hparams.frame_length,
                                             hop_length=hparams.frame_shift)
        logmel = librosa.power_to_db(mel ** 2)
        # 这里可以看一下时长，如果平均时长比较长 可以采用这种方法 否则最好全局normalize
        return librosa.util.normalize(logmel, axis=1)

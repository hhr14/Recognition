import librosa
import argparse
import os
import numpy as np
from hparams import get_params
from utils import get_melspectrum


def preprocess(hparams):
    train_label_list = os.listdir(hparams.input_dir)
    output_txt = open(hparams.output_dir, mode='w')
    for i in range(len(train_label_list)):
        print('processing', train_label_list[i], '......')
        train_path = os.path.join(hparams.input_dir, train_label_list[i], 'train')
        dev_path = os.path.join(hparams.input_dir, train_label_list[i], 'dev')
        label = int(train_label_list[i].split('-')[0][1:])
        train_file_list = os.listdir(train_path)
        dev_file_list = os.listdir(dev_path)
        for m in range(len(train_file_list)):
            wav_name = train_file_list[m].split('.')[0]
            wav_path = os.path.join(train_path, train_file_list[m])
            mel_spec = get_melspectrum(hparams, wav_path)
            if mel_spec is None:
                continue
            output_path = os.path.join(hparams.mel_output_dir, wav_name + '.mel')
            np.save(output_path, mel_spec)
            output_txt.write(output_path + ' ' + str(label) + ' ' + str(mel_spec.shape[1]) + '\n')
        for n in range(len(dev_file_list)):
            wav_name = dev_file_list[n].split('.')[0]
            wav_path = os.path.join(dev_path, dev_file_list[n])
            mel_spec = get_melspectrum(hparams, wav_path)
            if mel_spec is None:
                continue
            output_path = os.path.join(hparams.mel_output_dir, wav_name + '.mel')
            np.save(output_path, mel_spec)
            output_txt.write(output_path + ' ' + str(label) + ' ' + str(mel_spec.shape[1]) + '\n')


if __name__ == "__main__":
    preprocess(get_params())


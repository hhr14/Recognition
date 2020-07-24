import os
import numpy as np
from hparams import get_params
from utils import get_melspectrum, VadSplit


def preprocess(hparams):
    output_txt = open(hparams.train_data_txt_dir, mode='w')

    if hparams.use_vad is True:
        vad_process(hparams)
        train_proc_list = os.listdir(hparams.train_proc_dir)
        for i in range(len(train_proc_list)):
            wav_name = train_proc_list[i].split('.')[0]
            wav_path = os.path.join(hparams.train_proc_dir, train_proc_list[i])
            mel_spec = get_melspectrum(hparams, wav_path)
            label = int(train_proc_list[i].split('-')[0][1:])
            if mel_spec is None:
                continue
            output_path = os.path.join(hparams.mel_output_dir, wav_name + '.mel')
            np.save(output_path, mel_spec)
            output_txt.write(output_path + ' ' + str(label) + ' ' + str(mel_spec.shape[1]) + '\n')
        return

    train_label_list = os.listdir(hparams.train_data_dir)
    for i in range(len(train_label_list)):
        print('processing', train_label_list[i], '......')
        train_path = os.path.join(hparams.train_data_dir, train_label_list[i], 'train')
        dev_path = os.path.join(hparams.train_data_dir, train_label_list[i], 'dev')
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


def vad_process(hparams):
    train_label_list = os.listdir(hparams.train_data_dir)
    for i in range(len(train_label_list)):
        print('processing', train_label_list[i], '......')
        train_path = os.path.join(hparams.train_data_dir, train_label_list[i], 'train')
        dev_path = os.path.join(hparams.train_data_dir, train_label_list[i], 'dev')
        label = int(train_label_list[i].split('-')[0][1:])
        train_file_list = os.listdir(train_path)
        dev_file_list = os.listdir(dev_path)
        for m in range(len(train_file_list)):
            wav_path = os.path.join(train_path, train_file_list[m])
            vs = VadSplit(hparams, wav_path, hparams.train_proc_dir)
            vs.output_segment()
        for n in range(len(dev_file_list)):
            wav_path = os.path.join(dev_path, dev_file_list[n])
            vs = VadSplit(hparams, wav_path, hparams.train_proc_dir)
            vs.output_segment()


if __name__ == "__main__":
    preprocess(get_params())


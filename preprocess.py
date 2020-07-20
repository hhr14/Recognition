import librosa
import argparse
import os
import numpy as np


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
    parser = argparse.ArgumentParser(description="preprocess hparams")
    parser.add_argument('--input_dir', type=str, default='data/train')
    parser.add_argument('--output_dir', type=str, default='data/train.txt')
    parser.add_argument('--mel_output_dir', type=str, default='data/mel')
    parser.add_argument('--n_mels', type=int, default=24)
    parser.add_argument('--frame_length', type=int, default=400)
    # 按照原文说法 这里帧长应该是400 即25ms
    parser.add_argument('--frame_shift', type=int, default=160)
    preprocess(parser.parse_args())


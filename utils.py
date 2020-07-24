import torch
import os
from model import xvecTDNN
import random
import numpy as np
import librosa
import webrtcvad
import sys
import collections
import contextlib
import wave


def load_data(hparams):
    f = open(hparams.train_data_txt_dir, 'r')
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
        mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=hparams.n_mels,
                                             n_fft=hparams.mel_frame_length,
                                             hop_length=hparams.mel_frame_shift)
        logmel = librosa.power_to_db(mel ** 2)
        # 这里可以看一下时长，如果平均时长比较长 可以采用这种方法 否则最好全局normalize
        return librosa.util.normalize(logmel, axis=1)


class Frame(object):
    """Represents a "frame" of audio data."""
    def __init__(self, bytes, timestamp, duration):
        self.bytes = bytes
        self.timestamp = timestamp
        self.duration = duration


class VadSplit:
    def __init__(self, hparams, wav_path, output_path):
        self.vad_mode = hparams.vad_mode
        self.frame_length = hparams.vad_frame_dur
        self.buffer_length = hparams.vad_buffer_length
        self.wav_path = wav_path
        self.output_path = output_path
        self.audio, self.sample_rate = self.read_wave()

    def read_wave(self):
        try:
            with contextlib.closing(wave.open(self.wav_path, 'rb')) as wf:
                num_channels = wf.getnchannels()
                assert num_channels == 1
                sample_width = wf.getsampwidth()
                assert sample_width == 2
                sample_rate = wf.getframerate()
                assert sample_rate in (8000, 16000, 32000, 48000)
                pcm_data = wf.readframes(wf.getnframes())
                return pcm_data, sample_rate
        except Exception:
            print('Error open', self.wav_path)
            return None, None

    def write_wave(self, path, audio):
        with contextlib.closing(wave.open(path, 'wb')) as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(self.sample_rate)
            wf.writeframes(audio)

    def frame_generator(self):
        """Generates audio frames from PCM audio data.
        Takes the desired frame duration in milliseconds, the PCM data, and
        the sample rate.
        Yields Frames of the requested duration.
        """
        # 因为wave库是以byte方式读入数据，而给定音频是每个数据是16-bit，也就是一个数据就要占2byte
        # 因此需要乘2来获取对应时长的数据
        n = int(self.sample_rate * (self.frame_length / 1000.0) * 2)
        offset = 0
        timestamp = 0.0
        duration = (float(n) / self.sample_rate) / 2.0
        while offset + n < len(self.audio):
            yield Frame(self.audio[offset:offset + n], timestamp, duration)
            timestamp += duration
            offset += n

    def vad_collector(self, vad, frames):
        num_padding_frames = self.buffer_length
        ring_buffer = collections.deque(maxlen=num_padding_frames)
        triggered = False

        voiced_frames = []
        for frame in frames:
            is_speech = vad.is_speech(frame.bytes, self.sample_rate)

            # sys.stdout.write('1' if is_speech else '0')
            if not triggered:
                ring_buffer.append((frame, is_speech))
                num_voiced = len([f for f, speech in ring_buffer if speech])
                # If we're NOTTRIGGERED and more than 90% of the frames in
                # the ring buffer are voiced frames, then enter the
                # TRIGGERED state.
                if num_voiced > 0.9 * ring_buffer.maxlen:
                    triggered = True
                    # sys.stdout.write('+(%s)' % (ring_buffer[0][0].timestamp,))
                    # We want to yield all the audio we see from now until
                    # we are NOTTRIGGERED, but we have to start with the
                    # audio that's already in the ring buffer.
                    for f, s in ring_buffer:
                        voiced_frames.append(f)
                    ring_buffer.clear()
            else:
                # We're in the TRIGGERED state, so collect the audio data
                # and add it to the ring buffer.
                voiced_frames.append(frame)
                ring_buffer.append((frame, is_speech))
                num_unvoiced = len([f for f, speech in ring_buffer if not speech])
                # If more than 90% of the frames in the ring buffer are
                # unvoiced, then enter NOTTRIGGERED and yield whatever
                # audio we've collected.
                if num_unvoiced > 0.9 * ring_buffer.maxlen:
                    # sys.stdout.write('-(%s)' % (frame.timestamp + frame.duration))
                    triggered = False
                    yield b''.join([f.bytes for f in voiced_frames])
                    ring_buffer.clear()
                    voiced_frames = []
        # if triggered:
        #     sys.stdout.write('-(%s)' % (frame.timestamp + frame.duration))
        # sys.stdout.write('\n')
        # If we have any leftover voiced audio when we run out of input,
        # yield it.
        if voiced_frames:
            yield b''.join([f.bytes for f in voiced_frames])

    def output_segment(self):
        if self.audio is None:
            return
        vad = webrtcvad.Vad(self.vad_mode)
        frames = self.frame_generator()
        frames = list(frames)
        segments = self.vad_collector(vad, frames)
        wav_name = os.path.basename(self.wav_path).split('.')[0]
        output_base_name = os.path.join(self.output_path, wav_name)
        for i, segment in enumerate(segments):
            path = output_base_name + '-%d.wav' % (i,)
            self.write_wave(path, segment)

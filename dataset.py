from torch.utils.data import Dataset
import numpy as np


class WindowDataset(Dataset):
    def __init__(self, mel_list, time_list, output, step_per_epoch, hparams):
        self.mel_list = mel_list
        self.time_list = time_list
        self.output = output
        self.hparams = hparams
        self.batch_size = hparams.batch_size
        self.window_size = hparams.window_size
        self.step_per_epoch = step_per_epoch
        self.sample_list = []
        self.build_random_list()

    def __getitem__(self, item):

        random_index = np.random.randint(0, len(self.sample_list))
        id = self.sample_list[random_index][0]
        begin = self.sample_list[random_index][1]
        #  now input is : [filter-bank(128), Time]
        mel_input = np.load(self.mel_list[id] + '.npy')[:, begin: begin + self.window_size]
        language_id_output = int(self.output[id]) - 1
        return {'mel_input': mel_input, 'language_id_output': language_id_output}


    def __len__(self):
        # len must be real length of dataset, not size / batch_size
        return self.step_per_epoch * self.batch_size

    def build_random_list(self):
        for i in range(len(self.time_list)):
            if int(self.time_list[i]) > self.window_size:
                for j in range(int(self.time_list[i]) - self.window_size):
                    self.sample_list.append([i, j])
        print("sample_list length:", len(self.sample_list))
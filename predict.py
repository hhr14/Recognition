import numpy as np
import torch
import utils
from hparams import get_params
import os


def predict(hparams, predict_file_list, mymodel, device):
    # use model.eval() to ban dropout or BN
    out_csv = open(hparams.predict_output, 'w')
    out_csv.write('id,label\n')
    mymodel.eval()
    for i in range(len(predict_file_list)):
        print('predict ' + predict_file_list[i], ' .....')
        mel_data = utils.get_melspectrum(hparams, predict_file_list[i])
        if mel_data is None:
            continue
        lan_result = np.zeros(hparams.lan_num)
        count = 0
        begin = 0
        end = begin + hparams.window_size
        while end <= mel_data.shape[1]:
            mel_input = mel_data[:, begin: end]
            mel_input = torch.from_numpy(mel_input[np.newaxis, :]).to(device)
            lan_output = mymodel(mel_input.float())
            lan_result += (lan_output.cpu().detach().numpy()).reshape(hparams.lan_num)
            count += 1
            begin += hparams.predict_step
            end += hparams.predict_step
        predict_lan = np.argmax(lan_result)
        out_csv.write(os.path.basename(predict_file_list[i]) + ',L' +
                      format(predict_lan + 1, '0>3d') + '\n')


if __name__ == "__main__":
    hparams = get_params()
    print(hparams)
    os.environ['CUDA_VISIBLE_DEVICES'] = hparams.gpu
    device = torch.device("cuda")
    mymodel = utils.create_model(hparams, mode='predict')
    mymodel = mymodel.float()
    mymodel.to(device)
    if hparams.model == 'xvecTDNN':
        predict(hparams, utils.get_predict_file_list(hparams.predict_input), mymodel, device)
    else:
        raise ValueError('Unlegal model')

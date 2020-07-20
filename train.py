import numpy as np
import utils
from hparams import get_params
import os
import torch
from dataset import WindowDataset
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim


def train(train_data, validate_data, mymodel, device, recent_epoch):
    # use model.eval() when validate.
    model_name = hparams.model_save_path + 'b' + str(hparams.batch_size) + \
                 '_lr' + str(hparams.lr) + '_'
    train_generator = WindowDataset(train_data[0], train_data[2], train_data[1],
                                    hparams.train_step, hparams)
    train_loader = DataLoader(dataset=train_generator, batch_size=hparams.batch_size,
                              shuffle=True, num_workers=4)
    validate_generator = WindowDataset(validate_data[0], validate_data[2], validate_data[1],
                                    hparams.validate_step, hparams)
    validate_loader = DataLoader(dataset=validate_generator, batch_size=hparams.batch_size,
                                 shuffle=False, num_workers=4)
    optimizer = optim.Adam(mymodel.parameters(), lr=hparams.lr)
    loss_func = nn.CrossEntropyLoss()
    # pytorch crossentropy 自带 softmax
    for e in range(hparams.epochs):
        print('\n\nepoch:', e)
        epoch_ave_loss_list = {'train': [], 'eval': []}
        for mode in ['train', 'eval']:
            if mode == 'train':
                mymodel.train()
                loader = train_loader
            else:
                mymodel.eval()
                loader = validate_loader
            num_correct = 0
            total = 0
            for b, data in enumerate(loader):
                mel_input = data['mel_input'].to(device)
                label = data['language_id_output'].to(device)
                # label [batch]
                # output [batch, C]
                optimizer.zero_grad()
                output = mymodel(mel_input.float())
                loss = loss_func(output, label)
                pred_language = output.argmax(dim=1)
                num_correct += torch.eq(pred_language, label).sum().item()
                total += label.size(0)
                epoch_ave_loss_list[mode].append(loss.item())
                if mode == 'train':
                    loss.backward()
                    optimizer.step()
                    print('step:', b, 'loss:', loss.item())
            epoch_ave_loss = np.mean(epoch_ave_loss_list[mode])
            print(mode + '_loss:', epoch_ave_loss, mode + '_language_acc:',
                  num_correct / total)

        if (e + 1) % hparams.check_point_distance == 0:
            epoch = e + 1 + recent_epoch
            mymodel_name = model_name + 'e' + str(epoch) + '-' + \
                           format(np.mean(epoch_ave_loss_list['eval']), '.4f') + '.pt'
            torch.save({'model_state': mymodel.state_dict()},
                       mymodel_name)


if __name__ == "__main__":
    hparams = get_params()
    print(hparams)
    os.environ['CUDA_VISIBLE_DEVICES'] = hparams.gpu
    device = torch.device("cuda")
    mymodel, recent_epoch = utils.create_model(hparams)
    mymodel = mymodel.float()
    mymodel.to(device)
    print(mymodel)
    # summary(mymodel, [(hparams.window_size, hparams.refer_size),
    #                   (hparams.window_size, hparams.content_size)], batch_size=1, device='cuda')
    train_data, validate_data = utils.load_data(hparams)
    if hparams.model == 'xvecTDNN':
        train(train_data, validate_data, mymodel, device, recent_epoch)
    else:
        raise ValueError('Unlegal Error')

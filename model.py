import torch
import torch.nn as nn
from torch.nn import functional as F


class xvecTDNN(nn.Module):
    """
    由于语种跟语言很有关系 而语言又是一个上下文强相关的 可能需要加入一些上下文模型 如LSTM
    """
    def __init__(self, hparams, **kwargs):
        super(xvecTDNN, self).__init__()

        self.grads = {}
        self.hparams = hparams
        p_dropout = hparams.dropout

        self.tdnn1 = nn.Conv1d(in_channels=hparams.feature_dim, out_channels=512, kernel_size=5, dilation=1)
        self.bn_tdnn1 = nn.BatchNorm1d(512, momentum=0.1)
        self.dropout_tdnn1 = nn.Dropout(p=p_dropout)

        self.tdnn2 = nn.Conv1d(in_channels=512, out_channels=512, kernel_size=5, dilation=2)
        self.bn_tdnn2 = nn.BatchNorm1d(512, momentum=0.1)
        self.dropout_tdnn2 = nn.Dropout(p=p_dropout)

        self.tdnn3 = nn.Conv1d(in_channels=512, out_channels=512, kernel_size=7, dilation=3)
        self.bn_tdnn3 = nn.BatchNorm1d(512, momentum=0.1)
        self.dropout_tdnn3 = nn.Dropout(p=p_dropout)

        self.tdnn4 = nn.Conv1d(in_channels=512, out_channels=512, kernel_size=1, dilation=1)
        self.bn_tdnn4 = nn.BatchNorm1d(512, momentum=0.1)
        self.dropout_tdnn4 = nn.Dropout(p=p_dropout)

        self.tdnn5 = nn.Conv1d(in_channels=512, out_channels=1500, kernel_size=1, dilation=1)
        self.bn_tdnn5 = nn.BatchNorm1d(1500, momentum=0.1)
        self.dropout_tdnn5 = nn.Dropout(p=p_dropout)

        self.fc1 = nn.Linear(3000, 512)
        self.bn_fc1 = nn.BatchNorm1d(512, momentum=0.1)
        self.dropout_fc1 = nn.Dropout(p=p_dropout)

        self.fc2 = nn.Linear(512, 512)
        self.bn_fc2 = nn.BatchNorm1d(512, momentum=0.1)
        self.dropout_fc2 = nn.Dropout(p=p_dropout)

        self.fc3 = nn.Linear(512, hparams.lan_num)

    def forward(self, x):
        """
        :param mel_input: mel spectrum. [batch, 128, window_size]
        :return:
        """
        # Note: x must be (batch_size, feat_dim, chunk_len)

        x = self.dropout_tdnn1(self.bn_tdnn1(F.relu(self.tdnn1(x))))
        x = self.dropout_tdnn2(self.bn_tdnn2(F.relu(self.tdnn2(x))))
        x = self.dropout_tdnn3(self.bn_tdnn3(F.relu(self.tdnn3(x))))
        x = self.dropout_tdnn4(self.bn_tdnn4(F.relu(self.tdnn4(x))))
        x = self.dropout_tdnn5(self.bn_tdnn5(F.relu(self.tdnn5(x))))

        x.register_hook(self.save_grad('x_5'))
        #  上述代码可以将该层的输出x在反向传播到这里的时候输出它的梯度，同理可以把所有层的梯度都加入到grads中查看

        if self.training:
            shape = x.size()
            noise = torch.cuda.FloatTensor(shape) if torch.cuda.is_available() else torch.FloatTensor(shape)
            torch.randn(shape, out=noise)
            x += noise * self.hparams.eps
        #  加入noise这一步很重要 能让x中的零元素变成非零 从而求导的时候不会出现NaN
        #  求导的时候出现NaN的本质原因是求std之前有一些特殊情况下 某一个维度特征在所有时间步上都等于均值
        #  导致那个时间步求方差会等于0，而这在反向求导的时候，由于std 函数外层是开根号，求导的结果是
        #  1/ (2 * 根号) 就导致除零错误 从而反向梯度爆炸，加入一些随机噪声可以基本避免这种及其特殊的情况
        #  发生，而输出的时候由于不进行反向求导 所以不需要进行噪声的补正 不过加上影响也不大

        stats = torch.cat((x.mean(dim=2), x.std(dim=2)), dim=1)

        # 本质就是每个时间步都会预测出一个固定长度的embedding 最后是对不同时间步的embedding作 均值和方差
        x = self.dropout_fc1(self.bn_fc1(F.relu(self.fc1(stats))))
        x = self.dropout_fc2(self.bn_fc2(F.relu(self.fc2(x))))
        x = self.fc3(x)
        return x

    def save_grad(self, name):
        def hook(grad):
            self.grads[name] = grad

        return hook

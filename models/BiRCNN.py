import torch
import torch.nn as nn



class BiRCNN(nn.Module):

    def __init__(self, num_classes=34):
        super(BiRCNN, self).__init__()
        self.cnn1 = nn.Sequential(
            nn.Conv1d(in_channels=8, out_channels=128, kernel_size=36,
                      stride=1, padding=0, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv1d(in_channels=128, out_channels=128, kernel_size=36,
                      stride=1, padding=0, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv1d(in_channels=128, out_channels=128, kernel_size=36,
                      stride=1, padding=0, bias=False),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=16, stride=16, padding=0),
            nn.Dropout(p=0.5, inplace=True)
        )
        self.cnn2 = nn.Sequential(
            nn.Conv1d(in_channels=8, out_channels=128, kernel_size=36,
                      stride=1, padding=0, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv1d(in_channels=128, out_channels=128, kernel_size=36,
                      stride=1, padding=0, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv1d(in_channels=128, out_channels=128, kernel_size=36,
                      stride=1, padding=0, bias=False),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=16, stride=16, padding=0),
            nn.Dropout(p=0.5, inplace=True)
        )
        self.rnn1 = nn.LSTM(input_size=2048, hidden_size=128, num_layers=2, bidirectional=True, dropout=0.25)
        self.rnn2 = nn.LSTM(input_size=2048, hidden_size=128, num_layers=2, bidirectional=True, dropout=0.25)
        self.rnn3 = nn.LSTM(input_size=11, hidden_size=128, num_layers=2, bidirectional=True, dropout=0.25)
        self.fc = nn.Linear(in_features=256 * 3, out_features=num_classes)

    def forward(self, x, HRV):
        out = x
        out_sep_1 = []
        out_sep_2 = []
        for i in range(out.shape[0]):
            sep = out[i]
            sep_1 = self.cnn1(sep)
            sep_2 = self.cnn2(sep)
            sep_1 = sep_1.reshape(sep_1.shape[0], -1)
            sep_2 = sep_2.reshape(sep_2.shape[0], -1)
            sep_1 = sep_1.unsqueeze(0)
            sep_2 = sep_2.unsqueeze(0)
            out_sep_1.append(sep_1)
            out_sep_2.append(sep_2)

        out_1 = torch.cat(out_sep_1, dim=0).permute(1, 0, 2)  # out_1 => [Lb, b, 2048]
        out_2 = torch.cat(out_sep_2, dim=0).permute(1, 0, 2)  # out_2 => [Lb, b, 2048]
        out_rnn, (out_h, out_c) = self.rnn1(out_1)
        out_1 = torch.cat([out_h[-1, :, :], out_h[-2, :, :]], dim=1)  # out_1 => [b, 256]
        out_rnn, (out_h, out_c) = self.rnn2(out_2)
        out_2 = torch.cat([out_h[-1, :, :], out_h[-2, :, :]], dim=1)  # out_2 => [b, 256]

        HRV = HRV.permute(2, 0, 1)  # HRV => [2, 16, 11]
        out_rnn, (out_h, out_c) = self.rnn3(HRV)
        out_3 = torch.cat([out_h[-1, :, :], out_h[-2, :, :]], dim=1)  # out_3 => [b, 256]

        out = torch.cat([out_1, out_2, out_3], dim=1)  # out => [b, 768]
        out = self.fc(out)  # out => [b, 20]

        print(out.shape)

        return out



if __name__ == '__main__':
    '''
    这里按照原文的意思，是将连续的Lb段heartbeat输入网络
    每段hearbeat是根据R-peak分割的，并fix length到定长
    这里我假设将[8, 5000]分割成6段心跳（实际有几段我不知道这里是假设，后续算出来改就好）
    大致分割流程：根据R-peak分割出若干不等长hearbeat，根据原文的g(x)函数将每个hearbeat规整为721，舍去首位180，变为361
    这个361就和原文保持一致吧，主要是5000分割了几段心跳
    HRV我也不知道咋算的...
    '''
    input = torch.randn(16, 6, 8, 361)
    HRV = torch.randn(16, 11, 2)
    BiRCNN = BiRCNN()
    output = BiRCNN(input, HRV)
    # print(output.shape)
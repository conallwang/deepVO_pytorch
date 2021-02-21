import torch
import torch.nn as nn

def conv(batch_norm, in_channel, out_channel, ks=3, sd=1):
    if batch_norm:
        return nn.Sequential(
            nn.Conv2d(in_channel, out_channel, ks, sd, (ks-1)//2, bias=False),
            nn.BatchNorm2d(out_channel),
            nn.ReLU()
        )
    else:
        return nn.Sequential(
            nn.Conv2d(in_channel, out_channel, ks, sd, (ks-1)//2, bias=False),
            nn.ReLU()
        )

def fc(activation, in_feature, out_feature):
    if activation:
        return nn.Sequential(
            nn.Linear(in_feature, out_feature),
            nn.ReLU()
        )
    else:
        return nn.Linear(in_feature, out_feature)

class DeepVO(nn.Module):
    def __init__(self, batch_norm=False):
        super().__init__()

        self.batch_norm = batch_norm

        self.conv1 = conv(self.batch_norm, 6, 64, 7, 2)
        self.conv2 = conv(self.batch_norm, 64, 128, 5, 2)
        self.conv3 = conv(self.batch_norm, 128, 256, 5, 2)
        self.conv3_1 = conv(self.batch_norm, 256, 256, 3, 1)
        self.conv4 = conv(self.batch_norm, 256, 512, 3, 2)
        self.conv4_1 = conv(self.batch_norm, 512, 512, 3, 1)
        self.conv5 = conv(self.batch_norm, 512, 512, 3, 2)
        self.conv5_1 = conv(self.batch_norm, 512, 512, 3, 1)
        self.conv6 = conv(self.batch_norm, 512, 1024, 3, 2)

        self.lstm = nn.LSTM(input_size=1024*20*6, hidden_size=1000, num_layers=2, batch_first=True)

        self.fc1 = fc(True, 1000, 128)
        self.fc2 = fc(False, 128, 6)
    
    def forward(self, x1, x2):
        N, T, C, H, W = x1.shape        # batch, seq, channel, height, width
        x1 = x1.reshape(-1, C, H, W)
        x2 = x2.reshape(-1, C, H, W)    

        x = torch.cat([x1, x2], dim=1)  # [N, T, 6, 384, 1280]
        x = self.conv1(x)               # [N, T, 64, 192, 640]
        x = self.conv2(x)               
        x = self.conv3_1(self.conv3(x))
        x = self.conv4_1(self.conv4(x))
        x = self.conv5_1(self.conv5(x))
        x = self.conv6(x)               # [N, T, 1024, 6, 20]

        x = x.view(N, -1, 1024*6*20)

        x, _ = self.lstm(x)

        x = x.reshape(-1, 1000)
        x = self.fc1(x)
        x = self.fc2(x)

        return x



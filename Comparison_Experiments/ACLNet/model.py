import torch.nn as nn
import torch
import torch.nn.functional as F
import torchvision.models as models
from functools import partial
from ConvLSTM import ConvLSTM
from config import *
# from ACLNet.ConvLSTM import ConvLSTM
# from ACLNet.config import *

class ACLNet(nn.Module):
    def __init__(self):
        super(ACLNet,self).__init__()
        self.Vgg_feats=models.vgg16(pretrained=True).features[0:30]

        self.maxpool_atn1_1 = nn.MaxPool2d((2, 2), stride=(2, 2))
        self.conv_atn1_1 = nn.Conv2d(512, 64, 1)
        self.conv_atn1_2 = nn.Conv2d(64, 128, 3, padding=(1, 1))
        self.maxpool_atn1_2 = nn.MaxPool2d((2, 2), stride=(2, 2))
        self.conv_atn1_3 = nn.Conv2d(128, 64, 1)
        self.conv_atn1_4 = nn.Conv2d(64, 128, 3, padding=(1, 1))
        self.conv_atn1_5 = nn.Conv2d(128, 1, 1)
        self.upsampling1 = nn.UpsamplingNearest2d(scale_factor=4)

        self.convLSTM = ConvLSTM(input_channels=512, hidden_channels=[256], kernel_size=3, step=num_frames,
                        effective_step=[4]).cuda()
        self.conv_atn2_1 = nn.Conv2d(256, 1, 1)
        self.upsampling2_1 = nn.UpsamplingNearest2d(scale_factor=4)
        self.upsampling2_2 = nn.UpsamplingNearest2d(scale_factor=2)


    def forward(self,x):
        out_batch=[]
        # [batch, num, c, h, w]->[batch, c, h, w]
        for img_seq  in torch.unbind(x, dim=1):
            feats =self.Vgg_feats(img_seq)

            attention = self.maxpool_atn1_1(feats)
            attention = F.relu(self.conv_atn1_1(attention))
            attention = F.relu(self.conv_atn1_2(attention))
            attention = self.maxpool_atn1_2(attention)
            attention = F.relu(self.conv_atn1_3(attention))
            attention = F.relu(self.conv_atn1_4(attention))
            attention = torch.sigmoid(self.conv_atn1_5(attention))
            attention = self.upsampling1(attention)

            # f_attention = attention.repeat(1, 512, 1, 1)
            f_attention = attention.repeat(1, 512, 1, 1).cuda()
            m_outs = f_attention * feats
            outs = feats + m_outs

            outs = outs.type(torch.FloatTensor)
            
            outs = self.convLSTM(outs.cuda())[0][0]
            outs = torch.sigmoid(self.conv_atn2_1(outs))
            outs = self.upsampling2_1(outs)
            out_batch.append(outs)
        out_batch = torch.stack(out_batch, dim=1)
        # attention = self.upsampling2_2(attention)
        # return [outs, attention]
        return out_batch


class ACLLoss(nn.Module):
    def __init__(self):
        super(ACLLoss, self).__init__()
        return

    # def forward(self, y_pred, y_sal, y_fix):
    def forward(self, y_pred, y_sal):
        y_pred = F.normalize(y_pred, p=1, dim=[2, 3])
        y_sal = F.normalize(y_sal, p=1, dim=[2, 3])
        loss_kl = self.kl_divergence(y_sal, y_pred)
        loss_cc = self.correlation_coefficient(y_sal, y_pred)
        # loss_nss = self.nss(y_sal, y_fix)
        # loss = 10 * loss_kl + loss_cc + loss_nss
        # return loss, loss_kl, loss_cc, loss_nss
        loss =  10*loss_kl + 0.1*loss_cc   # 不应该是减去cc吗
        return loss

    def kl_divergence(self, y_sal, y_pred):
        loss = torch.sum(y_sal * torch.log((y_sal + 1e-7) / (y_pred + 1e-7)))
        return loss

    def correlation_coefficient(self, y_sal, y_pred):
        N = y_pred.size()[2] * y_pred.size()[3]
        sum_prod = torch.sum(y_sal * y_pred, dim=[2, 3])
        sum_x = torch.sum(y_sal, dim=[2, 3])
        sum_y = torch.sum(y_pred, dim=[2, 3])
        sum_x_square = torch.sum(y_sal**2, dim=[2, 3]) + 1e-7
        sum_y_square = torch.sum(y_pred**2, dim=[2, 3]) + 1e-7
        num = sum_prod - ((sum_x * sum_y) / N)
        den = torch.sqrt((sum_x_square - sum_x**2 / N) * (sum_y_square - sum_y**2 / N))
        loss = torch.sum(-2 * num / den)  #
        return loss

    def nss(self, y_fix, y_pred):
        y_pred = F.layer_norm(y_pred, normalized_shape=y_pred.size()[2:])
        loss = -torch.sum(((torch.sum(y_fix * y_pred, dim=[2, 3])) / (torch.sum(y_fix, dim=[2, 3]))))
        return loss

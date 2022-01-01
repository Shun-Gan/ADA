import torch.nn as nn
import torch, sys
import torch.nn.functional as F
import torchvision.models as models
from functools import partial
# from ConvLSTM import ConvLSTM
# from config import *
# from LearnableGaussianFilter import GaussianLayer
from HWS.ConvLSTM import ConvLSTM
from HWS.config import *
from HWS.LearnableGaussianFilter import GaussianLayer


class HWSNet(nn.Module):
    def __init__(self):
        super(HWSNet,self).__init__()
        # self.CNN_feats=models.alexnet(pretrained=True).features[0:30]
        # self.CNN_feats=nn.Sequential(*list(models.alexnet(pretrained=True).children()))[0]
        self.CNN_feats=models.alexnet(pretrained=True).features[0:10]
        # for name, params in self.CNN_feats.named_parameters():
        #     print(name)

        # self.upsampling = nn.UpsamplingNearest2d(scale_factor=2)
        self.upsampling = nn.UpsamplingBilinear2d(scale_factor=2)
        self.conv1 = nn.Conv2d(256, 128, 1)
        self.dropout1 = nn.Dropout2d(p=0.2)
        self.conv2 = nn.Conv2d(128, 128, 1)
        self.dropout2 = nn.Dropout2d(p=0.2)
        self.conv3 = nn.Conv2d(128, 64, 1)
        self.dropout3 = nn.Dropout2d(p=0.2)

        self.convLSTM = ConvLSTM(input_channels=64, hidden_channels=[32], kernel_size=3, step=num_frames,
                        effective_step=[4]).cuda()
        self.conv4 = nn.Conv2d(32, 1, 1)           
        self.smooth = GaussianLayer(max_sigma=1.5)
        # self.upsampling2 = nn.UpsamplingNearest2d(scale_factor=2)


    def forward(self,x):
        out_batch=[]
        # [8, 8, 3, 360, 640]
        # [batch, num, c, h, w]->[batch, c, h, w]
        for img_seq  in torch.unbind(x, dim=1):
            feats1 =self.CNN_feats(img_seq)
            feats = self.upsampling(feats1)

            feats = F.relu(self.conv1(feats))
            feats = self.dropout1(feats)
            feats = F.relu(self.conv2(feats))
            feats = self.dropout2(feats)
            feats = F.relu(self.conv3(feats))
            feats = self.dropout3(feats)
            feats = self.convLSTM(feats)[0][0]
            feats = self.conv4(feats)
            feats = self.smooth(feats)
            feats = torch.sigmoid(feats)
            # feats = self.upsampling2(feats)
            out_batch.append(feats)
        out_batch = torch.stack(out_batch, dim=1)
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

def _pointwise_loss(lambd, input, target, size_average=True, reduce=True):
    d = lambd(input, target)
    if not reduce:
        return d
    return torch.mean(d) if size_average else torch.sum(d)

class KLDLoss(nn.Module):
    def __init__(self):
        super(KLDLoss, self).__init__()

    def KLD(self, inp, trg):
        inp = inp/torch.sum(inp)
        trg = trg/torch.sum(trg)
        eps = sys.float_info.epsilon

        return torch.sum(trg*torch.log(eps+torch.div(trg,(inp+eps))))

    def forward(self, inp, trg):
        return _pointwise_loss(lambda a, b: self.KLD(a, b), inp, trg)

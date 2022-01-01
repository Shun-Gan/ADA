import torch.nn as nn
import torch
import torchvision.models as models

class MLNet(nn.Module):
    
    def __init__(self,prior_size):
        super(MLNet, self).__init__()
        # loading pre-trained vgg16 model and         
        # removing last max pooling layer
        features = list(models.vgg16(pretrained = True).features)[:-1]
        
        # making same spatial size
        # by calculation :) 
        # in pytorch there was problem outputing same size in maxpool2d
        features[23].stride = 1
        features[23].kernel_size = 5
        features[23].padding = 2
                
        self.features = nn.ModuleList(features).eval() 
        # adding dropout layer
        self.fddropout = nn.Dropout2d(p=0.2)
        # adding convolution layer to down number of filters 1280 ==> 64
        self.int_conv = nn.Conv2d(1280,64,kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.int_batch = nn.BatchNorm2d(64)
        self.int_relu = nn.ReLU(inplace=True)
        self.pre_final_conv = nn.Conv2d(64,1,kernel_size=(1, 1), stride=(1, 1) ,padding=(0, 0))
        self.final_batch = nn.BatchNorm2d(1)
        self.final_relu = nn.ReLU(inplace=True)
        # prior initialized to ones
        self.prior = nn.Parameter(torch.ones((1,1,prior_size[0],prior_size[1]), requires_grad=True))
        
        # bilinear upsampling layer
        self.bilinearup = torch.nn.UpsamplingBilinear2d(scale_factor=2)

        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        
        results = []
        for ii,model in enumerate(self.features):
            x = model(x)
            if ii in {16,23,29}:
                results.append(x)
        
        # concat to get 1280 = 512 + 512 + 256
        x = torch.cat((results[0],results[1],results[2]),1) 
        
        # adding dropout layer with dropout set to 0.5 (default)
        x = self.fddropout(x)
        
        # 64 filters convolution layer
        x = self.int_conv(x)
        x = self.int_batch(x)
        x = self.int_relu(x)
        # 1*1 convolution layer
        x = self.pre_final_conv(x)
        # x = self.final_batch(x)
        # x = self.final_relu(x)

        upscaled_prior = self.bilinearup(self.prior)
        # print ("upscaled_prior shape: {}".format(upscaled_prior.shape))

        # dot product with prior
        x = x * upscaled_prior
        # x = torch.nn.functional.relu(x,inplace=True)
        x = self.sigmoid(x)
        return x


import os, glob, torch, cv2
import pandas as pd
import numpy as np
from model import SODModel
from PIL import Image 
import torchvision.transforms as transforms

# processing = transforms.Compose([
#     transforms.ToPILImage(), # string path= > image data
#     transforms.Resize((256, 256),
#                 interpolation=Image.LANCZOS),
#     transforms.ToTensor(),
#     transforms.Normalize(mean=[0.485, 0.456, 0.406],
#                             std=[0.229, 0.224, 0.225])
#     ])

normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])

import utils
def metric_compute(pred, sal):
    # the mean of metrics in every batch 
    metric_list = ['kld', 'cc', 'sim']
    num_ =len(metric_list)
    metrics = torch.zeros(num_, dtype=float)
    for ii in range(num_):
        metric_name = metric_list[ii]
        if metric_name=='kld':
            metrics[ii] = utils.new_kld(pred, sal).mean()
        elif metric_name=='cc':
            metrics[ii] = utils.new_cc(pred, sal).mean()
        elif metric_name=='sim':
            metrics[ii] = utils.new_sim(pred, sal).mean()
        elif metric_name=='emd':
            metrics[ii] = utils.emd_loss(pred, sal).mean()
    return metrics

class Example_prediction():
    def __init__(self, ):
        self.root = './runs/'
        self.img_path = '../inference/'
        self.img_size = [256, 256]
        self.num_frames = 32
        self.net = SODModel()
        self.write_video=True
        self.net = self.net.cuda()
        self.folders = sorted(os.listdir(self.root))

    def read_img_seq(self, imgs):
        img_seq=[]
        for img_path in imgs:
            img = cv2.imread(img_path)[:, :, ::-1]

            img = cv2.resize(img, tuple((int(self.img_size[1]), int(self.img_size[0]))), 
                interpolation=cv2.INTER_CUBIC)
            img = img.astype('float32')/255.
            img = img.transpose(2, 0, 1)
            img = np.ascontiguousarray(img)
            img = torch.from_numpy(img)
            img_seq.append(normalize(img))

        return torch.stack(img_seq)
        #     fine_img = cv2.resize(img, (self.img_size[1], self.img_size[0]))
        #     fine_img = fine_img.astype('float32')/255.
        #     fine_img_seq.append(fine_img.transpose(2, 0, 1))
        # return  fine_img_seq

    def write_img(self, path, img):
        img = img[1:-1, 1:-1]
        img = (img/img.max()*255)
        img = np.array(img.detach().cpu().numpy(), dtype=np.uint8)
        cv2.imwrite(path, img)

    def inference(self,):
        for folder in self.folders:
            dataset = folder.split('_')[0]
            weight = self.root+folder+'/best_weight_%s.pt'%dataset
            self.net.load_state_dict(torch.load(weight))
            print('load weight file', weight)
            root_path = self.root+folder+'/inference/'
            if not os.path.exists(root_path):
                os.makedirs(root_path)
            self.predict_example(root_path)

    def predict_example(self, root_path):
        img_folders = os.listdir(self.img_path)
        for img_folder in img_folders:
            imgs = glob.glob(self.img_path+img_folder+'/*/images/*.png')
            output_path = os.path.join(root_path, img_folder, 
                                        imgs[0].split(os.sep)[1])
            if not os.path.exists(output_path):
                os.makedirs(output_path)
            fine_img_seq = self.read_img_seq(imgs) # [c, h, w]*num
            
            self.net.eval()
            for idx, fine_img in enumerate(fine_img_seq):
                # fine_img = torch.from_numpy(fine_img)
                fine_img = torch.unsqueeze(fine_img, dim=0) # [1, c, num, h, w]

                logits, _ = self.net(fine_img.cuda())
                img_path = output_path + '/%s.png'%(str(idx+1).zfill(4))
                self.write_img(img_path, torch.squeeze(logits))

            if self.write_video:
                self.generate_video(output_path, imgs)

    def infer_example(self, net, root):
        root_path = root +'/inference/'
        if not os.path.exists(root_path):
            os.makedirs(root_path)
        img_folders = os.listdir(self.img_path)
        for img_folder in img_folders:
            imgs = glob.glob(self.img_path+img_folder+'/*/images/*.png')
            output_path = os.path.join(root_path, img_folder, 
                                        imgs[0].split(os.sep)[1])
            if not os.path.exists(output_path):
                os.makedirs(output_path)
            fine_img_seq = self.read_img_seq(imgs) # [c, h, w]*num
            
            net.eval()
            for idx, fine_img in enumerate(fine_img_seq):
                # fine_img = torch.from_numpy(fine_img)
                fine_img = torch.unsqueeze(fine_img, dim=0) # [1, c, num, h, w]

                logits, _ = net(fine_img.cuda())
                img_path = output_path + '/%s.png'%(str(idx+1).zfill(4))
                self.write_img(img_path, torch.squeeze(logits))

            if self.write_video:
                self.generate_video(output_path, imgs)


    def generate_video(self, video_path, images):

        freq_dict = {'DADA2000':30, 'BDDA':30, 'EyeTrack':24, 'DReye':25}
        fourcc= cv2.VideoWriter_fourcc(*'XVID')
        source = video_path.split(os.sep)[-2].split('/')[-1]
        Freq = freq_dict[source]

        video_name = video_path+'_heatmap.avi'
        if os.path.exists(video_name):
            print('pass: ', video_name)
        else:
            if len(images)>0:
                img_size = cv2.imread(images[0]).shape[:2]
                out = cv2.VideoWriter (video_name, \
                    fourcc, Freq, (img_size[1],img_size[0]))
                for img in images[self.num_frames-1:]:
                    gaze_heatmap = self.heat_map(img, img_size, video_path)
                    out.write(gaze_heatmap)
                
                out.release()
                print('write the video:', video_name)
            else:
                pass

    def heat_map(self, img, img_size, video_path):

        gaze_map_path = os.path.join(video_path, img.split(os.sep)[-1])

        image = cv2.imread(img)
        gaze_map = cv2.imread(gaze_map_path, 0)
        gaze_map = cv2.resize(gaze_map, (img_size[1], img_size[0]))
        heatmap = cv2.applyColorMap(gaze_map, cv2.COLORMAP_JET)
        gaze_heatmap =cv2.addWeighted(image,0.5,heatmap,0.5,0)
        
        return gaze_heatmap



if __name__=="__main__":
    Predictor = Example_prediction()
    Predictor.inference()


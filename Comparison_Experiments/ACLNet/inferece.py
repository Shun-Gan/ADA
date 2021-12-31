import os, glob, torch, cv2
import pandas as pd
import numpy as np
from model import ACLNet
from PIL import Image 
import torchvision.transforms as transforms

def transform(snippet):
    ''' stack & noralization '''
    snippet = np.concatenate(snippet, axis=-1)
    snippet = torch.from_numpy(snippet).permute(2, 0, 1).contiguous().float()
    snippet = snippet.mul_(2.).sub_(255).div(255)
    snippet = snippet.view(-1,3,snippet.size(1),snippet.size(2)).permute(1,0,2,3)
    return snippet

# unisal
processing = transforms.Compose([
    transforms.ToPILImage(), # string path= > image data
    transforms.Resize((256, 320),
                interpolation=Image.LANCZOS),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225])
    ])

class Example_prediction():
    def __init__(self, ):
        self.root = './runs/'
        self.img_path = '../inference/'
        # self.img_size = [256, 360]
        self.num_frames = 8
        self.net =  ACLNet()
        self.write_video=True
        self.net = self.net.cuda()
        self.folders = sorted(os.listdir(self.root))

    def read_img_seq(self, imgs):
        img_seq=[]
        for img_path in imgs:
            data = cv2.imread(img_path)
            data = np.ascontiguousarray(data[:, :, ::-1])
            img = processing(data)
            img_seq.append(img)
        return torch.stack(img_seq)

    def write_img(self, path, img):
        img = (img/img.max()*255)
        img = np.array(img.detach().cpu().numpy(), dtype=np.uint8)
        cv2.imwrite(path, img[0,...])

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
            img_seq = self.read_img_seq(imgs) # [num, c, h, w]
            
            self.net.eval()
            for start in range(0, len(imgs)-self.num_frames+1):
                inputs = img_seq[start:start+self.num_frames,...]
                inputs = torch.unsqueeze(inputs, dim=0) # [1, num, c, h, w]
                logits = self.net(inputs.cuda())
                img_path = output_path + '/%s.png'%(str(start+self.num_frames).zfill(4))
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


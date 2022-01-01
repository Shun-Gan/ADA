from torch.utils.data import Dataset, DataLoader
import cv2, glob, torch, random
import numpy as np
from PIL import Image 
import PIL
import torchvision.transforms as transforms
from config import *
import scipy.io as sio

normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
# unisal
processing = transforms.Compose([
    transforms.ToPILImage(), # string path= > image data
    transforms.Resize((normal_shape_r, normal_shape_c),
                interpolation=Image.LANCZOS),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225])
    ])

def worker_init(worker_init):
    seed = 2017
    np.random.seed(int(seed)+worker_init)

def data_generator(dataset, phase, Data_lister, bz, num_workers):
    db_ = Data_Dataset(dataset, phase, Data_lister)
    if phase=='train':
        shuffle=True
    else:
        shuffle=False
    data_loader = DataLoader(db_, batch_size=bz,  worker_init_fn=worker_init,
                shuffle=shuffle, num_workers=num_workers, pin_memory=True)
    return data_loader


def video_generator(dataset, phase, Video_lister, bz, num_workers):
    db_ = Video_Dataset(dataset, phase, Video_lister)
    if phase=='train':
        shuffle=True
    else:
        shuffle=False
    data_loader = DataLoader(db_, batch_size=bz,  worker_init_fn=worker_init,
                shuffle=shuffle, num_workers=num_workers, pin_memory=True)
    return data_loader


class Video_load():
    def __init__(self):
        # [96, 118, 153] 分别对应 valid开始， test开始，以及总的视频文件数+1（因为range最后一位不要）
        self.data_dict={"EyeTrack":[[97, 119, 154],[224, 384],'EyeTrack'],
                        "DReye":[[379, 406, 780],[224, 384],'New_DReye'],
                        # "DADA2000":[[698, 798, 1014],[224, 384],'DADA'], # 避免报错统一
                        "DADA2000":[[0, 1, 220],[224, 384],'DADA_test'],
                        "BDDA":[[927, 1127, 1429],[224, 384],'BDDA']}
        self.root='C:/0_Code/unisal-master/data/'

    def load_list(self, dataset, phase, clip_=False):
        self.path=self.root
        self.dataset=dataset
        self.data_split=self.data_dict[dataset][0]
        self.image_size=self.data_dict[dataset][1]
        # self.image_size=[360, 640]
        self.data_path = self.root+self.data_dict[dataset][2]

        if phase == 'train':
            videos = range(1, self.data_split[0])
        elif phase == 'valid':
            videos = range(self.data_split[0],
                        self.data_split[1])
            if dataset=="EyeTrack":  # aurgement
                videos = list(videos)*2
        elif phase == 'test':
            videos = range(self.data_split[1],
                        self.data_split[2])
        video_list=[]
        img_num = []
        for video in videos:
            video_path = self.data_path+'/%s/'%(str(video).zfill(4))
            img_num.append(len(glob.glob(video_path+'images/*.png')))
            video_list.append(video_path)
        
        return  video_list, img_num, self.image_size


class Video_Dataset(Dataset):
    def __init__(self, dataset, phase, Video_lister):
        self.Data_lister=Video_lister
        self.dataset = dataset
        self.phase = phase 
        # self.map_size = (22, 40)
        self.video_list, self.img_num, self.img_size = \
                Video_lister.load_list(dataset, phase)
    
    def preprocess_img(self,):

        tf = transforms.Compose([
            lambda x:Image.open(x).convert('RGB'), # string path= > image data
            transforms.Resize((shape_r, shape_c)),
            # transforms.RandomRotation(15),
            # transforms.CenterCrop(self.resize),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
            ])
        return tf

    def load_image_data(self, video_path, img_id):
        img_path = video_path+'images/%s.png'%(str(img_id).zfill(4))
        data = cv2.imread(img_path)
        data = np.ascontiguousarray(data[:, :, ::-1])
        img = processing(data)
        return img


    def load_input_data(self, video_path, img_id):
        img_path = video_path+'images/%s.png'%(str(img_id).zfill(4))
        img = cv2.imread(img_path)[:, :, ::-1]  # BGR->RGB 
        img = cv2.resize(img, (shape_c, shape_r),  # c2.resize(W, H)
                interpolation=cv2.INTER_CUBIC)
        img = img.astype('float32')/255.
        img = img.transpose(2, 0, 1) # HWC->CHW
        img = np.ascontiguousarray(img)
        return normalize(torch.from_numpy(img))



    def load_target_data(self, video_path, img_id):
        if self.dataset=='DReye' or self.dataset=='BDDA':
            sal_path = video_path+'new_maps//%s.png'%(str(img_id).zfill(4))
        else:
            sal_path = video_path+'maps//%s.png'%(str(img_id).zfill(4))
        sal = cv2.imread(sal_path, cv2.IMREAD_GRAYSCALE)
        sal = cv2.resize(sal, (shape_c_out, shape_r_out))
        sal = sal.astype('float32')/255.
        sal = sal[None,...]
        sal = np.ascontiguousarray(sal)
        return torch.from_numpy(sal)

    def load_fixation(self, video_path, ii):

        fix_file = video_path+'fixdata.mat'
        data = sio.loadmat(fix_file)
        try:
            fix_x = data['fixdata'][ii - 1][0][:, 3]
            fix_y = data['fixdata'][ii - 1][0][:, 2]
        except:
            try:
                fix_x = data['fixdata'][ii - 2][0][:, 3]
                fix_y = data['fixdata'][ii - 2][0][:, 2]
            except:
                fix_x = data['fixdata'][ii - 3][0][:, 3]
                fix_y = data['fixdata'][ii - 3][0][:, 2]
        mask = np.zeros((720, 1280), dtype='float32')
        for i in range(len(fix_x)):
            mask[fix_x[i], fix_y[i]] = 1
        return torch.from_numpy(mask)

    def load_fixmap(self, data_file, frame_id):
        fix_path = data_file+'/fixations/%s.png'%str(frame_id).zfill(4)
        fixmap = cv2.imread(fix_path, 0)
        return torch.from_numpy(fixmap.astype('float32'))

    def get_data(self, video_path, start):
        id_list = range(start, start+num_frames )
        # img_seq = [self.load_input_data(video_path, ii) for ii in id_list]
        img_seq = [self.load_image_data(video_path, ii) for ii in id_list]
        img_tensor = torch.stack(img_seq)
        sal_seq = [self.load_target_data(video_path, ii) for ii in id_list]
        sal_tensor = torch.stack(sal_seq)
        fix_seq = [self.load_fixmap(video_path, ii) for ii in id_list]
        fix_seq = torch.stack(fix_seq)
        return img_tensor, sal_tensor, fix_seq

    def __len__(self):
        return len(self.video_list)
    
    def __getitem__(self, id):
        video_path = self.video_list[id]
        start = random.randint(1, 
                max(self.img_num[id] - num_frames, 0))
        data = self.get_data(video_path, start)
        return data



class Data_Dataset(Dataset):
    def __init__(self, dataset, phase, Data_lister):
        self.Data_lister=Data_lister
        self.dataset = dataset
        self.phase = phase 
        self.map_size = (22, 40)
        self.frame_list, self.map_list, self.img_size = \
                Data_lister.load_list(dataset, phase)


    def load_input_data(self, data_file):

        fine_data = cv2.imread(str(data_file)) # [360, 640, 3]
        fine_data = fine_data[:, :, ::-1] # BGR->RGB
        coarse_data = cv2.resize(fine_data, tuple((int(self.img_size[1]), int(self.img_size[0]))), 
                interpolation=cv2.INTER_CUBIC) # [224, 384, 3]

        fine_data = fine_data.astype('float32')/255.
        coarse_data = coarse_data.astype('float32')/255.
        fine_data = fine_data.transpose(2, 0, 1)
        coarse_data = coarse_data.transpose(2, 0, 1)
        fine_data = np.ascontiguousarray(fine_data)
        coarse_data = np.ascontiguousarray(coarse_data)
        fine_data = torch.from_numpy(fine_data)
        coarse_data = torch.from_numpy(coarse_data)
        loaders = (normalize(fine_data), normalize(coarse_data))
        return loaders


    def load_target_data(self, data_file):

        data = cv2.imread(str(data_file), cv2.IMREAD_GRAYSCALE)
        data = cv2.resize(data, tuple((int(self.map_size[1]), int(self.map_size[0]))), 
                interpolation=cv2.INTER_CUBIC)
        data = data.astype('float32')/255.
        data = data[None,...]
        data = np.ascontiguousarray(data)
        return torch.from_numpy(data)


    def __len__(self):
        return len(self.frame_list)

    def __getitem__(self, idx):
        # frame_process = self.preprocess_frame()
        # map_process = self.preprocess_map()
        frame = self.load_input_data(self.frame_list[idx])
        map_ = self.load_target_data(self.map_list[idx])
        # frame = frame_process(frame)
        # map_ = map_process(map_)
        return frame, map_

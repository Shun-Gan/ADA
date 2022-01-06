from torch.utils.data import Dataset, DataLoader
import cv2, glob, torch, random
import numpy as np
import torchvision.transforms as transforms
import scipy.io as sio

normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])

def transform(snippet):
    ''' stack & noralization '''
    snippet = np.concatenate(snippet, axis=-1)
    snippet = torch.from_numpy(snippet).permute(2, 0, 1).contiguous().float()
    snippet = snippet.mul_(2.).sub_(255).div(255)
    snippet = snippet.view(-1,3,snippet.size(1),snippet.size(2)).permute(1,0,2,3)
    return snippet

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
    db_ = Video_Dataset(dataset, phase, Video_lister, num_frame=32)
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


class Data_load():
    def __init__(self):
        # [96, 118, 153] 分别对应 valid开始， test开始，以及总的视频文件数+1（因为range最后一位不要）
        self.data_dict={"EyeTrack":[[97, 119, 154],[224, 384],'EyeTrack'],
                        "DReye":[[379, 406, 780],[224, 384],'New_DReye'],
                        # "DADA2000":[[698, 798, 1014],[224, 384],'DADA'], # 避免报错统一
                        "DADA2000":[[0, 1, 220],[224, 384],'DADA_test'], # 避免报错统一
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
            file_list = range(1, self.data_split[0])
        elif phase == 'valid':
            file_list = range(self.data_split[0],
                        self.data_split[1])
        elif phase == 'test':
            file_list = range(self.data_split[1],
                        self.data_split[2])
        frame_list=[]
        map_list=[]
        for file_num in file_list:
            frame_path = self.data_path+'/%s/images/'%(str(file_num).zfill(4))
            frame_list.extend(glob.glob(frame_path+'*.png'))
            if dataset=='DReye' or dataset=='BDDA':
                map_path = self.data_path+'/%s/new_maps/'%(str(file_num).zfill(4))
            else:
                map_path = self.data_path+'/%s/maps/'%(str(file_num).zfill(4))
            map_list.extend(glob.glob(map_path+'*.png'))
        assert len(frame_list)==len(map_list)
        if clip_:
            clip_dict={"EyeTrack":1, "DReye":4, "DADA2000":7, "BDDA":9}
            step = clip_dict[dataset]
            frame_list = frame_list[0:-1:step]
            frame_list = map_list[0:-1:step]

        return frame_list, map_list, self.image_size

class Video_Dataset(Dataset):
    def __init__(self, dataset, phase, Video_lister, num_frame):
        self.Data_lister=Video_lister
        self.dataset = dataset
        self.phase = phase 
        self.num_frame = num_frame
        self.map_size = (22, 40)
        self.video_list, self.img_num, self.img_size = \
                Video_lister.load_list(dataset, phase)
    
    def load_input_data(self, video_path, id_list):
        clip = []
        for img_id in id_list:
            img_path = video_path+'images/%s.png'%(str(img_id).zfill(4))
            img = cv2.imread(img_path)
            img = cv2.resize(img, tuple((int(self.img_size[1]), int(self.img_size[0]))))
            clip.append(img)

        return transform(clip)

    def load_target_data(self, video_path, img_id):
        if self.dataset=='DReye' or self.dataset=='BDDA':
            sal_path = video_path+'new_maps//%s.png'%(str(img_id).zfill(4))
        else:
            sal_path = video_path+'maps//%s.png'%(str(img_id).zfill(4))
        sal = cv2.imread(sal_path, cv2.IMREAD_GRAYSCALE)
        sal = cv2.resize(sal, tuple((int(self.img_size[1]), int(self.img_size[0]))))
        sal = torch.from_numpy(sal.copy()).contiguous().float()
        return sal

    def load_fixation(self, data_file, frame_id):

        fix_file = data_file+'fixdata.mat'
        data = sio.loadmat(fix_file)
        try:
            fix_x = data['fixdata'][frame_id - 1][0][:, 3]
            fix_y = data['fixdata'][frame_id - 1][0][:, 2]
        except:
            try:
                fix_x = data['fixdata'][frame_id - 2][0][:, 3]
                fix_y = data['fixdata'][frame_id - 2][0][:, 2]
            except:
                fix_x = data['fixdata'][frame_id - 3][0][:, 3]
                fix_y = data['fixdata'][frame_id - 3][0][:, 2]
        mask = np.zeros((720, 1280), dtype='float32')
        for i in range(len(fix_x)):
            mask[fix_x[i], fix_y[i]] = 1
        return torch.from_numpy(mask)

    def load_fixmap(self, data_file, frame_id):
        fix_path = data_file+'/fixations/%s.png'%str(frame_id).zfill(4)
        fixmap = cv2.imread(fix_path, 0)
        return fixmap.astype('float32')

    def get_data(self, video_path, start):
        id_list = range(start, start+ self.num_frame)
        img_tensor = self.load_input_data(video_path, id_list)
        sal_tensor = self.load_target_data(video_path, start+ self.num_frame) 
        fix_tensor = self.load_fixmap(video_path, start+ self.num_frame)
        return img_tensor, sal_tensor, fix_tensor

    def __len__(self):
        return len(self.video_list)
    
    def __getitem__(self, id):
        video_path = self.video_list[id]
        start = random.randint(1, 
                max(self.img_num[id] - self.num_frame, 0))
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
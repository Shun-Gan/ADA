from tensorboardX import SummaryWriter
import os, glob, csv, random, time
import datetime, time, json, shutil 
from tqdm import tqdm
import numpy as np
import PIL, cv2
import torch
import torch.nn as nn
import torch.nn.functional as F 
from torchvision import  transforms

from PIL import Image 
import  matplotlib
from    matplotlib import pyplot as plt
from model import Model
import utils

from data import Data_load, data_generator

def torch_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  
    np.random.seed(seed)  # Numpy module.
    random.seed(seed)  # Python random module.
    torch.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True



class Trainer():

    def __init__(self, dataset, epochs=10, lr=0.01, momentum=0.8, bz=20, 
                weight_decay=1e-4, lr_gamma=0.8, best_acc=None, 
                best_weight='', copy_code_flag=True, data_instance=None):
        self.dataset=dataset
        self.bz = bz
        self.lr = lr
        self.lr_gamma = lr_gamma 
        self.epochs = epochs
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.best_weight = best_weight
        self.copy_code_flag = copy_code_flag
        self.net = Model().cuda()
        self.metric_list = ['kld', 'cc','sim', 'auc_j', 'nss']
        self.outpath = './runs/'+'%s_'%dataset+str(datetime.datetime.now())[:-7].replace(' ', '_').replace(':','')
        # self.best_weight = self.outpath+'/best_weight_%s.pt'%dataset
        self.writer = SummaryWriter(self.outpath)

        self.optimizer = self._optimizer()
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer,
                        gamma = self.lr_gamma, step_size=1)
        self.criteon = nn.BCELoss().cuda()        

        self.history_loss = {"train":[],"valid":[], "test":[]}
        self.history_metric={"train":{},"valid":{}, "test":{}}

        self.bestepoch = []
        self.test_predictions = []
        self.test_labels = []

        self.other_map = data_instance.other_map(dataset)

        if self.copy_code_flag:
            self.copy_code()
        for key in self.history_metric:
            for metric_name in self.metric_list:
                self.history_metric[key][metric_name]=[]

        

    def _optimizer(self):
        optimizer =  torch.optim.Adam(self.net.parameters(), lr = self.lr, 
                    weight_decay=self.weight_decay)

        return optimizer

    def fit(self, train_loader, val_loader, clip_=False):
        if clip_: # dowmsampling for large datasets
            clip_dict={"EyeTrack":1, "DReye":4, "DADA2000":7, "BDDA":9}
            clip_num = clip_dict[self.dataset]
            clip_batch = int(len(train_loader)/clip_num)

        for epoch in range(self.epochs):
            self.net.train()
            losses = 0
            metrics = [0]*len(self.metric_list)
            # phar = tqdm(total=len(train_loader.dataset))
            timestamp = time.time()
            print('########## epoch {:0>3d}; \t lr: {:.8f} ##########'.format(epoch, 
                    self.optimizer.param_groups[0]["lr"]))
            for batch_idx, (input, target) in enumerate(train_loader):
                input = input.cuda()
                target = target.cuda()
                output = self.net(input)

                loss = self.criteon(output, target)
                metric = self.metric_compute(output, target)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                losses += loss
                for nn in range(len(self.metric_list)): 
                    metrics[nn] += metric[nn].item()
                if batch_idx % 200 == 0:
                    # phar.update(len(img)*100) 
                    print('\n Train Epoch: {}/{} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                        epoch, self.epochs, batch_idx * len(input), len(train_loader.dataset),
                            100. * batch_idx / len(train_loader), loss.item()),
                            '\t new_kld:%.3f, new_cc:%.3f, new_sim:%.3f'%(metric[0].item(),
                                    metric[1].item(), metric[2].item()),
                            '\t Cost time:{} s'.format(int(time.time()-timestamp)))
                    timestamp = time.time()
                    
                    if clip_ and batch_idx>clip_batch:
                        break

            train_loss = losses/batch_idx
            self.history_loss["train"].append(float(train_loss))
            for ii, metric_name in enumerate(self.metric_list): 
                self.history_metric['train'][metric_name].append(metrics[ii]/batch_idx)
                print('train %s: %.3f \t'%( metric_name, metrics[ii]/batch_idx), end='') 
            print()
            print('Train average loss:', train_loss)
            self.writer.add_scalar('train_loss', train_loss, epoch)
            # self.writer.add_scalar('new_kld', metrics[0]/batch_idx, epoch)
            # self.writer.add_scalar('cc', metrics[1]/batch_idx, epoch)
            # self.writer.add_scalar('new_cc', metrics[2]/batch_idx, epoch)
            # phar.close()
            break_flag = self.evaluate('valid', val_loader, epoch)
            self.scheduler.step()
            if break_flag:
                break


    def evaluate(self, phase, db_loader, epoch=None):

        if phase == 'test' or phase=='infer':
            self.net.load_state_dict(torch.load(self.best_weight))
            print('Load the best weight of trained on %s:'%self.dataset,
                'predict for %s'%dataset)
        
        if phase=='infer': 
            time_list=[]
        self.net.eval()
        with torch.no_grad():
            losses = 0
            metrics = [0]*len(self.metric_list)
            for batch_idx, data_batch in enumerate(db_loader):
                (frame_bh, map_bh, fix_bh) = data_batch
                frame_bh = frame_bh.cuda()
                map_bh = map_bh.cuda()
                t0=time.time()
                logits = self.net(frame_bh)

                if phase=='infer':
                    time_list.append(time.time()-t0)
                    
                losses += self.criteon(logits, map_bh).item()

                if batch_idx>100 and phase=='infer':
                    print('cost time: %.6fms'%(np.mean(time_list)*1000))

                metric = self.metric_compute(logits, map_bh, fix_bh)
                for nn in range(len(self.metric_list)): 
                    metrics[nn] += metric[nn].item()

            for ii, metric_name in enumerate(self.metric_list): 
                if phase=="valid":
                    self.history_metric[phase][metric_name].append(metrics[ii]/(batch_idx+1))
                else:
                    self.history_metric[phase][metric_name]=metrics[ii]/(batch_idx+1)

                print(phase, '%s: %.3f \t'%(metric_name, metrics[ii]/(batch_idx+1)), end='')
            print()   
            losses /= (batch_idx+1)
            print(phase+'_set: Average loss: {:.4f}\n'.format(
                losses))


        if phase == 'valid':
            break_flag=False
            self.writer.add_scalar(phase+'_loss', losses, epoch)
            self.history_loss["valid"].append(float(losses))
            best_loss = min(self.history_loss["valid"])
            if losses == best_loss:
                self.bestepoch = int(epoch)
                # self.best_weight = self.outpath+'/best_weight_%s_epoch%d.pt'%(self.dataset, epoch)
                torch.save(self.net.state_dict(), self.best_weight)
                print('write the best loss {:.2f} weight file in epoch {}'.format(losses, epoch))
            if epoch-self.bestepoch>3:
                print('early stopping')
                break_flag=True
            return break_flag
                

        elif phase == "test":
            self.history_loss["test"] = float(losses)
            metric_file = self.outpath+'/Train_%s_for_%s_metric_history.json'%(self.dataset, dataset)
            if self.dataset==dataset:
                loss_file = self.outpath+'/%s_loss_history.json'%self.dataset
                utils.write_json(self.history_loss, loss_file)
                utils.write_json(self.history_metric, metric_file)
            else:
                # only write metrics in tset phase for predicted datasets
                utils.write_json(self.history_metric['test'], metric_file)
                

    def metric_compute(self, pred, sal, fix):
        # the mean of metrics in every batch 
        num_ =len(self.metric_list)
        fix_np = fix.detach().cpu().numpy()

        pred_r = (pred/pred.max()*255)
        pred_r = np.array(pred_r.detach().cpu().numpy(), dtype=np.uint8)
        pred_np=[]
        for ii in range(pred_r.shape[0]):
            pred_s = cv2.resize(np.squeeze(pred_r[ii]), (1280, 720)) 
            pred_np.append(pred_s.astype('float32')/255.)
        pred_np = np.stack(pred_np)
        pred_resize = torch.from_numpy(pred_np)

        sal_r = (sal/sal.max()*255)
        sal_r = np.array(sal_r.detach().cpu().numpy(), dtype=np.uint8)
        sal_np=[]
        for ii in range(pred_r.shape[0]):
            sal_s = cv2.resize(np.squeeze(sal_r[ii]), (1280, 720)) 
            sal_np.append(sal_s.astype('float32')/255.)
        sal_np = np.stack(sal_np)

        metrics = torch.zeros(num_, dtype=float)
        for ii in range(num_):
            metric_name = self.metric_list[ii]
            if metric_name=='kld':
                metrics[ii] = utils.new_kld(pred, sal).mean()
            elif metric_name=='cc':
                metrics[ii] = utils.new_cc(pred, sal).mean()
            elif metric_name=='sim':
                metrics[ii] = utils.new_sim(pred, sal).mean()
            elif metric_name=='nss':
                metrics[ii] = utils.nss(pred_resize, fix.type(torch.bool)).mean()
            elif metric_name=='auc_s':
                # 偏小 原因未知取值与other_map的采样有关，但是基本不大约30。DADA，DReye都没公开计算代码
                # metrics[ii] = utils.auc_shuff_acl(pred_np, fix_np, self.other_map)
                auc_s=[]
                for jj in range(pred_r.shape[0]):
                    auc = utils.auc_shuff_acl(pred_np[jj], fix_np[jj], self.other_map)
                    auc_s.append(auc)
                metrics[ii] = np.mean(auc_s)
            elif metric_name=='ig':
                metrics[ii] = utils.information_gain(fix_np, pred_np, sal_np)
            elif metric_name=='auc_j':
                auc_j=[]
                for jj in range(pred_r.shape[0]):
                    auc = utils.auc_judd(pred_np[jj], fix_np[jj])
                    auc_j.append(auc)
                metrics[ii] = np.mean(auc_j)
        return metrics

    def copy_code(self):
        code_list = ["run.py","model.py", "utils.py", "data.py",'inference.py']
        code_path = self.outpath+"/copy_code/"
        if not os.path.exists(code_path):
            os.mkdir(code_path)
        for code in code_list:
            shutil.copy2('./'+code, code_path+code)
        print('copy the codes:', code_list, 'into the:', code_path)


if __name__=="__main__":

        dataset_list=["EyeTrack", "DReye", "DADA2000", "BDDA"]
        num_workers= 0
        bz = 8
        torch_seed(2017)
        Data_lister=Data_load()
        # frame_list, map_list = Data_lister.load_list('EyeTrack','train')
        # """  
        for dataset in ["EyeTrack"]:
            train_loader= data_generator(dataset, 'train', Data_lister, bz, num_workers)
            val_loader= data_generator(dataset, 'valid', Data_lister, bz, num_workers)
            # train on specific datasets
            Trainer_ = Trainer(dataset=dataset,
                                epochs=20, 
                                lr=1e-3, 
                                lr_gamma=0.9, 
                                momentum=0.9, 
                                weight_decay=1e-4,
                                bz=bz, 
                                data_instance = Data_lister,
                                best_weight='./runs/EyeTrack_2021-12-07_200847/best_weight_EyeTrack.pt'
                                )
            # Trainer_.fit(train_loader, val_loader, clip_=True)
            # predict in valid sets of all datasets
            for dataset in ["EyeTrack"]:
                test_loader= data_generator(dataset, 'test', Data_lister, bz, num_workers)
                Trainer_.evaluate('test', test_loader)
        # """     
        # num_workers= 0
        # bz = 1
        # # test inference time
        # for dataset in dataset_list:
        #     Trainer_ = Trainer(dataset=dataset,
        #                         best_weight='./runs/EyeTrack_2021-11-15_171929/best_weight_EyeTrack.pt',
        #                         bz=bz, 
        #                         copy_code_flag=False
        #                         )
        #     # predict in valid sets of all datasets
        #     for dataset in dataset_list:
        #         test_loader= data_generator(dataset, 'test', Data_lister, bz, num_workers)
        #         Trainer_.evaluate('infer', test_loader)
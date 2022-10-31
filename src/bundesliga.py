import time
import os
import numpy as np
import pandas as pd
import cv2

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.sampler import SubsetRandomSampler, RandomSampler, SequentialSampler
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, CosineAnnealingLR, ReduceLROnPlateau

import albumentations as A
import albumentations
import matplotlib.pyplot as plt
from tqdm.notebook import tqdm

import torchvision
import glob
import timm
from sklearn.metrics import confusion_matrix
import copy

#--------------------------------------------------------------------------------------------------------
class bundesAnalysis(object):
    def __init__(self):
        self.df = self.preparingData()

    def preparingData(self):
        err_tol = {
            'challenge': [ 0.30, 0.40, 0.50, 0.60, 0.70 ],
            'play': [ 0.15, 0.20, 0.25, 0.30, 0.35 ],
            'throwin': [ 0.15, 0.20, 0.25, 0.30, 0.35 ]
        }

        video_id_split = {
            'val':[
                '3c993bd2_0',
                '3c993bd2_1',
                '35bd9041_0',
                '35bd9041_1',
            ],
            'train':[
                '1606b0e6_0',
                '1606b0e6_1',
                '407c5a9e_1',
                '4ffd5986_0',
                '9a97dae4_1',
                'cfbe2e94_0',
                'cfbe2e94_1',
                'ecf251d4_0',
            ]
        }
        event_names = ['challenge', 'throwin', 'play']

        df = pd.read_csv("../data/train.csv")
        additional_events = []
        for arr in df.sort_values(['video_id','time','event','event_attributes']).values:
            if arr[2] in err_tol:
                tol = err_tol[arr[2]][ERR_TOL]/2
                additional_events.append([arr[0], arr[1]-tol, 'start_'+arr[2], arr[3]])
                additional_events.append([arr[0], arr[1]+tol, 'end_'+arr[2], arr[3]])
                
        for arr in df.sort_values(['video_id','time','event','event_attributes']).values:
            if arr[2] in err_tol:
                tol = err_tol[arr[2]][ERR_TOL]/2
                additional_events.append([arr[0], arr[1]-tol*2, 'pre_'+arr[2], arr[3]])
                additional_events.append([arr[0], arr[1]+tol*2, 'post_'+arr[2], arr[3]])
                
        df = pd.concat([df, pd.DataFrame(additional_events, columns=df.columns)])
        df = df[~df['event'].isin(event_names)]
        df = df.sort_values(['video_id', 'time'])

        cap = cv2.VideoCapture("../data/videos/train/ecf251d4_0.mp4")
        fps = cap.get(cv2.CAP_PROP_FPS)
        print("fps:", fps)
        df["frame"] = df["time"]*fps

        return df

    def extract_images(self, video_path, out_dir):
        video_name = os.path.basename(video_path).split('.')[0]
        cam = cv2.VideoCapture(video_path)
        print(video_path)
        frame_count = 1
        while True:
            successed, img = cam.read()
            if not successed:
                break
            outfile = f'{out_dir}/{video_name}-{frame_count:06}.jpg'
            img = cv2.resize(img, dsize=IMG_SIZE, interpolation=cv2.INTER_AREA)
            cv2.imwrite(outfile, img)
            frame_count += 1

        OUT_DIR = "./work/img/"
        IN_VIDEOS = glob.glob("../data/videos/train/*") # video files

        if not DEBUG:
            for video_path in IN_VIDEOS:
                extract_images(video_path, OUT_DIR)
        else:
            print("skipping video gen")
    
    def get_df(self, video_id, VAL=False):

        df_video = df[df.video_id == video_id]

        print(video_id, df_video.shape)

        #crr_statu => background, play, challenge, throwin
        arr = df_video[['frame','event']].values
        
        start = None
        data = []
        for a in arr:
            if "pre_" in a[1]:
                start = a[0]
                cls = a[1]
            if "start_" in a[1]:
                data.append({"start":start, "end":a[0], "cls":cls})
                start = a[0]
                cls = a[1].split("_")[-1]
            if "end_" in a[1]:
                end = a[0]
                data.append({"start":start, "end":end, "cls":cls})
            if "post_" in a[1]:
                data.append({"start":end, "end":a[0], "cls":a[1]})
        # make events
        out = []
        for d in data:
            start = int(d["start"])
                
            if os.path.isfile(os.path.join("work", IMG_SOURCE, video_id+"-"+str(start).zfill(6)+".jpg")):
                out.append({"frame":start, "cls":d["cls"], "video":video_id})
            start += 1
            while start <= d["end"]:
                if os.path.isfile(os.path.join("work", IMG_SOURCE, video_id+"-"+str(start).zfill(6)+".jpg")):
                    out.append({"frame":start, "cls":d["cls"], "video":video_id})
                start += 1
        
        df2 = pd.DataFrame(out)
        if not VAL:
            for i in range(10,df2.frame.max(), BACK_INTERVAL):
                if np.sum(df2.frame.isin([i]))==0:
                    if os.path.isfile(os.path.join("work", IMG_SOURCE, video_id+"-"+str(i).zfill(6)+".jpg")):
                        out.append({"frame":i, "cls":"background", "video":video_id})
                    else:
                        print("pass:", i)
        else:
            for i in range(10,df2.frame.max(), BACK_INTERVAL_VAL):
                if np.sum(df2.frame.isin([i]))==0:
                    if os.path.isfile(os.path.join("work", IMG_SOURCE, video_id+"-"+str(i).zfill(6)+".jpg")):
                        out.append({"frame":i, "cls":"background", "video":video_id})
                    else:
                        print("pass:", i)
        df2 = pd.DataFrame(out)

        if not DEBUG:
            for i,video_id in enumerate(video_id_split["train"]):
                df2 = get_df(video_id)
                if i > 0:
                    df_train = pd.concat([df_train, df2])
                else:
                    df_train = df2

            for i,video_id in enumerate(video_id_split["val"]):
                df2 = get_df(video_id, True)
                if i > 0:
                    df_val = pd.concat([df_val, df2])
                else:
                    df_val = df2
        return df2
#--------------------------------------------------------------------------------------------------------

class net(nn.Module):
    def __init__(self, modelname, out_dim=10, freeze_bn=True):
        super(net, self).__init__()
        self.model = timm.create_model(modelname, pretrained=True)
        self.model.reset_classifier(out_dim)

    def forward(self, x):
        x = self.model(x)
        return x

    def get_label(label):
        if label == "background":
            label = 0
            label2 = 0
        elif label == "challenge":
            label = 1
            label2 = 1
        elif label == "play":
            label = 2
            label2 = 1
        elif label == "throwin":
            label = 3
            label2 = 1
        elif label == "pre_challenge":
            label = 4
            label2 = 1
        elif label == "pre_play":
            label = 5
            label2 = 1
        elif label == "pre_throwin":
            label = 6
            label2 = 1
        elif label == "post_challenge":
            label = 7
            label2 = 1
        elif label == "post_play":
            label = 8
            label2 = 1
        elif label == "post_throwin":
            label = 9
            label2 = 1
        label = np.array(label)
        return label
    
#--------------------------------------------------------------------------------------------------------

class dflDataset(Dataset):
        def __init__(self,
                    df,
                    test=False,
                    transform=None,
                    ):

            self.df = df.reset_index(drop=True)
            self.test = test
            self.transform = transform

        def __len__(self):
            return self.df.shape[0]

        def __getitem__(self, index):
            row = self.df.iloc[index]
            images = []
            for i in [-1,0,1]:
                tiff_file = "work/{}/".format(IMG_SOURCE)+row.video+"-"+str(row.frame+i).zfill(6)+".jpg"
                im = cv2.imread(tiff_file, cv2.IMREAD_GRAYSCALE)
                images.append(im)
            images = np.array(images).transpose(1, 2, 0)

            # aug
            if self.transform is not None:
                images = self.transform(image=images)['image']
            file = copy.copy(tiff_file)
            images = images.astype(np.float32)
            images /= 255
            images = images.transpose(2, 0, 1)
            
            # Load labels
            label = get_label(row.cls)
            
            # Mixup part
            rd = np.random.rand()
            label2 = label
            gamma = np.array(np.ones(1)).astype(np.float32)[0]
            if mixup and rd < 0.9 and not self.test:
                mix_idx = np.random.randint(0, len(self.df)-1)
                row2 = self.df.iloc[mix_idx]
                images2 = []
                for i in [-1,0,1]:
                    tiff_file = "work/{}/".format(IMG_SOURCE)+row2.video+"-"+str(row2.frame+i).zfill(6)+".jpg"
                    im = cv2.imread(tiff_file, cv2.IMREAD_GRAYSCALE)
                    images2.append(im)
                images2 = np.array(images2).transpose(1, 2, 0)
                
                if self.transform is not None:
                    images2 = self.transform(image=images2)['image']
                    
                images2 = images2.astype(np.float32)
                images2 /= 255
                images2 = images2.transpose(2, 0, 1)
                # blend image
                gamma = np.array(np.random.beta(1,1)).astype(np.float32)
                images = ((images*gamma + images2*(1-gamma)))
                # blend labels
                label2 = get_label(row2.cls)
                
            transforms_train = albumentations.Compose([
                albumentations.ShiftScaleRotate(scale_limit=0.3, rotate_limit=45,p=0.5),
                A.OneOf([
                A.RandomBrightnessContrast(brightness_limit=0.2, 
                                            contrast_limit=0.2, p=0.5),
                ],p=0.9),
                A.Cutout(num_holes=24, max_h_size=64, max_w_size=64, fill_value=0, p=0.5),
                albumentations.Rotate(p=0.5),
                albumentations.Transpose(p=0.5),
                albumentations.VerticalFlip(p=0.5),
                albumentations.HorizontalFlip(p=0.5),   
                albumentations.Resize(IMSIZE[0], IMSIZE[1], p=1.0), 
            ])
            transforms_val = albumentations.Compose([albumentations.Resize(IMSIZE[0], IMSIZE[1], p=1.0),])
            return torch.tensor(images), torch.tensor(label), torch.tensor(label2), torch.tensor(gamma), file
    
    
    

#--------------------------------------------------------------------------------------------------------
STEPS = 25

class cnn(nn.Module):
    def __init__(self):
        super(cnn, self).__init__()
        hdn = 32
        self.fc = nn.Conv1d(10, hdn, 13, bias=False, stride=2)
        self.bn = nn.BatchNorm1d(hdn)
        self.do = nn.Dropout(0.2)
        self.fc2 = nn.Conv1d(hdn, hdn*2, 7, bias=False, stride=2)
        self.bn2 = nn.BatchNorm1d(hdn*2)
        self.do2 = nn.Dropout(0.25)
        self.fc3 = nn.Conv1d(hdn*2, hdn*2, 5, bias=False, stride=1)
        self.bn3 = nn.BatchNorm1d(hdn*2)
        self.do3 = nn.Dropout(0.25)
        self.fc4 = nn.Conv1d(hdn*2, hdn*2, 3, bias=False, stride=1)
        self.bn4 = nn.BatchNorm1d(hdn*2)
        self.do4 = nn.Dropout(0.25)
        self.fc5 = nn.Linear(hdn*2,10)
            
    def extract(self, x):
        return self.basemodel(x)

    def forward(self, x):
        x = self.do(F.relu(self.bn(self.fc(x))))
        x = self.do2(F.relu(self.bn2(self.fc2(x))))
        x = self.do3(F.relu(self.bn3(self.fc3(x))))
        x = self.do4(F.relu(self.bn4(self.fc4(x))))
        #x = self.do4(F.relu(self.bn4(self.fc4(x))))
        #x = self.fc4(x)
        return self.fc5(F.adaptive_avg_pool1d(x, 1).squeeze(-1))


# npz = np.load("../input/dflcv077/prob_train_psi.npy.npz")
# PREDS= npz["arr_0"]
# FILES= npz["arr_1"]
# TARGETS= npz["arr_2"]

# FILES = np.array(FILES)

# # TRAIN-VAL
# trainval = []
# for f in FILES:
#     video = f.split("/")[-1].split("-")[0]
#     if "35bd9041" in video:
#         trainval.append(0)
#     else:
#         trainval.append(1)
# trainval = np.array(trainval)
#--------------------------------------------------------------------------------------------------------


class linDataset(Dataset):
    def __init__(self,
                 df,
                 target,
                 files
                ):

        self.df = df
        self.target = target
        self.files = files

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        if i < STEPS+2:
            i += STEPS
        if i > len(self.df)-STEPS-2:
            i -= STEPS
        data = self.df[i-STEPS:i+STEPS+1]
        
        # Load labels
        label = self.target[i]
        
        # dataset_train = linDataset(PREDS[trainval==0], TARGETS[trainval==0], FILES[trainval==0])
        # dataset_valid = linDataset(PREDS[trainval==1], TARGETS[trainval==1], FILES[trainval==1])

        # # Setup dataloader
        # train_loader = torch.utils.data.DataLoader(dataset_train, batch_size=256, sampler=RandomSampler(dataset_train), num_workers=num_workers)
        # valid_loader = torch.utils.data.DataLoader(dataset_valid, batch_size=512, sampler=SequentialSampler(dataset_valid), num_workers=num_workers)

        # scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

        return torch.tensor(data).transpose(0,1), torch.tensor(label), self.files[i]

    def train_epoch(loader, optimizer):
        model.train().float()
        train_loss = []
        bar = tqdm(loader)
        for (data, target, _) in bar:
            data, target = data.to(device), target.to(device).long()
            loss_func = criterion
            optimizer.zero_grad()
            with torch.cuda.amp.autocast(enabled=use_amp):
                logits = model(data).squeeze(1)            
                loss = loss_func(logits, target)
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

            loss_np = loss.detach().cpu().numpy()
            train_loss.append(loss_np)
            smooth_loss = sum(train_loss[-100:]) / min(len(train_loss), 100)
            bar.set_description('loss: %.5f, smth: %.5f' % (loss_np, smooth_loss))
        return np.mean(train_loss)

    def val_epoch(loader, get_output=False):
        model.eval().half()
        val_loss = []
        LOGITS = []
        PREDS = []
        TARGETS = []
        FILES = []

        with torch.no_grad():
            for (data, target, files) in tqdm(loader):
                data, target = data.to(device), target.to(device).long()
                #data = data.reshape(-1, 15*10)
                logits = model(data.half())
                logits = logits.squeeze(1)        
                loss_func = criterion
                loss = loss_func(logits, target)

                # color
                pred = logits.sigmoid().detach()
                LOGITS.append(logits)
                PREDS.append(pred)
                TARGETS.append(target)
                FILES.extend(files)

                val_loss.append(loss.detach().cpu().numpy())
            val_loss = np.mean(val_loss)

        LOGITS = torch.cat(LOGITS).cpu().numpy()
        PREDS = torch.cat(PREDS).cpu().numpy()
        TARGETS = torch.cat(TARGETS).cpu().numpy()
        acc = np.sum(PREDS.argmax(1) == TARGETS)/len(PREDS.argmax(1))*100
        
        val_df = make_sub(PREDS[:, :4], FILES)
        score, ap_table = event_detection_ap(solution[solution['video_id'].isin(val_df['video_id'].unique())], val_df[['video_id', 'time', 'event', 'score']], tolerances)
        print("*"*50)
        print(score)
        print("*"*50)
        
        print(ap_table)
        
        
        if get_output:
            return LOGITS
        else:
            return val_loss, acc, score

    def make_sub(prob, filenames):
        frame_rate = 25
        window_size = WINDOW
        ignore_width = IGNORE
        df = pd.DataFrame(prob,columns=event_names_with_background)
        df['video_name'] = filenames
        df['video_id'] = df['video_name'].str.split('-').str[0].str.split("/").str[-1]
        df['frame_id'] = df['video_name'].str.split('-').str[1].str.split('.').str[0].astype(int)

        train_df = []
        for video_id,gdf in df.groupby('video_id'):
            for i, event in enumerate(event_names):
                #print(video_id, event)
                prob_arr = gdf[event].rolling(window=window_size, center=True).mean().fillna(-100).values
                sort_arr = np.argsort(-prob_arr)
                rank_arr = np.empty_like(sort_arr)
                rank_arr[sort_arr] = np.arange(len(sort_arr))
                idx_list = []
                for i in range(len(prob_arr)):
                    this_idx = sort_arr[i]
                    if this_idx >= 0:
                        idx_list.append(this_idx)
                        for parity in (-1,1):
                            for j in range(1, ignore_width+1):
                                ex_idx = this_idx + j * parity
                                if ex_idx >= 0 and ex_idx < len(prob_arr):
                                    sort_arr[rank_arr[ex_idx]] = -1
                this_df = gdf.reset_index(drop=True).iloc[idx_list].reset_index(drop=True)
                this_df["score"] = prob_arr[idx_list]
                this_df['event'] = event
                train_df.append(this_df)
        train_df = pd.concat(train_df)
        train_df['time'] = train_df['frame_id']/frame_rate
        
        return train_df.reset_index(drop=True)


    model = cnn()
    model.cuda()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

 
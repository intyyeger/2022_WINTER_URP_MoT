import os
import cv2
import numpy as np
import pandas as pd
import glob
from sklearn.cluster import KMeans
from tqdm import tqdm

os.environ["OMP_NUM_THREADS"] = "1" 

def createDirectory(directory):
    try:
        if not os.path.exists(directory):
            os.makedirs(directory)
    except OSError:
        print("Error: Failed to create the directory.")

def ccwh_to_xywh(x,y,w,h):
    x_new = (x*1920) - (w*1920+1) / 2
    y_new = (y*1080) - (h*1080+1) / 2
    return [x_new, y_new, w*1920+1, h*1080+1]

def tragectory_converter(path):

    #input_path = os.getcwd() + args.project+'/'+args.name + "/label/*.txt"
    #createDirectory("./"+args.stopover)
    input_path = path + '/*.txt'
    file_list = [f for f in glob.glob(input_path)]
    
    track_list = [] #ball location
    miss_index = []
    duplicate_index = []
    dup_idx = []
    for frame, file_name in enumerate(file_list): # for each frame
        with open(file_name) as f:
            label = [line.rstrip('\n') for line in f]

        num_ball = 0
        for i in label: #for each object
            obj = list(map(float,i.split(" ")))
            if obj[0] == 0: # if current frame catch the ball
                num_ball+=1
                track_list.append([frame+1, obj[1], obj[2],obj[3],obj[4], obj[5]])
            else:
                pass
        if num_ball == 0:
            miss_index.append(frame+1)
        elif num_ball > 1:
            duplicate_index.append(frame+1)
            dup_idx.append(len(track_list)-1)
    

    #Eliminate duplicate
    idx = 1
    dup_comp = []
    buf = []
    for track in track_list:
        if track[0] == idx:
            buf.append(track)
        else:
            if len(buf) != 1: #duplicate index in buffer
                if len(dup_comp) == 0: #duplicate occurred on first frame
                    pass
                else:
                    x = dup_comp[-1][1]
                    y = dup_comp[-1][2]
                    tmp = np.array(buf)[:,1:3] #duplicated matrix
                    select = np.argmin(((tmp - np.array([x , y]))**2).sum(axis = 1))
                    dup_comp.append(buf[select])      
            else:
                dup_comp.append(buf[0])
            buf = []#flush the buffer
            buf.append(track)
            idx = track[0]

    #last buffer clear
    if len(buf)!=1:
        x = dup_comp[-1][1]
        y = dup_comp[-1][2]
        tmp = np.array(buf)[:,1:] #duplicated matrix
        select = np.argmin(((tmp - np.array([x , y]))**2).sum(axis = 1))
        dup_comp.append(buf[select])
    else:
        dup_comp.append(buf[0])

    #recovery miss frame
    idx = 1
    miss_comp = []
    for dup in dup_comp:
        if dup[0] == idx:
            miss_comp.append(dup)
            idx += 1
        else: #index doesn't match
            # idx 11
            real_dt = miss_comp[-1]
            dt = dup[0] - idx + 1

            dx = (dup[1] - miss_comp[-1][1]) / dt
            dy = (dup[2] - miss_comp[-1][2]) / dt

            for it in range(idx, dup[0]):
                miss_comp.append([it, miss_comp[-1][1] + dx, miss_comp[-1][2] + dy, 0.01, 0.01, 0.25])

            miss_comp.append(dup)
            idx = dup[0]+1

    return miss_comp

def first_frame_color(file_path, image_path):
    input_path = file_path + '/*.txt'
    file_list = [f for f in glob.glob(input_path)]
    input_path = image_path + '/*.jpg'
    image_list = [f for f in glob.glob(input_path)]
    
    with open(file_list[0], "r") as f:
        label = [line.rstrip('\n') for line in f]
    src = cv2.imread(image_list[0], 1)
    
    data = []
    for i in label:
        label_list = list(map(float, i.split()))
        if label_list[0] == 1:
        
            x = int(label_list[1]*1920)+1
            y = int(label_list[2]*1080)+1
            w = int(label_list[3]*1920 / 2)+1
            h = int(label_list[4]*1080 / 2)+1
            crop_image = src[int(y) : int(y+h/2), int(x-w/2):int(x+w/2),:]
            crop_image = crop_image.mean(axis = 0).mean(axis = 0)
            data.append(crop_image)

        else:
            pass
    train_data = pd.DataFrame(np.array(data))
    model = KMeans(n_clusters=2)
    model.fit(train_data)
    centers = model.cluster_centers_
    pred = model.predict(train_data)
    
    data = np.array(data)
    team1_color = data[np.where(pred==1, True , False)].mean(axis = 0)
    team2_color = data[np.where(pred==0, True , False)].mean(axis = 0)
    
    return team1_color, team2_color #1-dimension numpy array [B,G,R]

def convert_yolov7_to_coco(file_path, image_path):
    team1_color, team2_color = first_frame_color(file_path, image_path)
    ball = tragectory_converter(file_path)
    input_path = file_path + '/*.txt'
    file_list = [f for f in glob.glob(input_path)]
    input_path = image_path + '/*.jpg'
    image_list = [f for f in glob.glob(input_path)]

    gt = []
    for idx, i in tqdm(enumerate(file_list)): #for each frame
        with open(i, "r") as f:
            label = [line.rstrip('\n') for line in f]
        src = cv2.imread(image_list[idx], 1)
        gt.append([ball[idx][0], -1] + ccwh_to_xywh(ball[idx][1],ball[idx][2],ball[idx][3],ball[idx][4]) + [ball[idx][-1], 0,-1])#append(ball) class = 0
        for l in label: #for each object
            buf = list(map(float, l.split()))
            if buf[0] == 1:
                x = buf[1]*1920+1
                y = buf[2]*1080+1
                w = buf[3]*1920+1
                h = buf[4]*1080+1
                crop_image = src[int(y) : int(y+h/2), int(x-w/2):int(x+w/2),:]
                crop_image = crop_image.mean(axis = 0).mean(axis = 0)
                team1 = (((crop_image - team1_color))**2).sum()
                team2 = (((crop_image - team2_color))**2).sum()
                if team1 < team2: #team1
                    gt.append([idx+1, -1, x - w/2, y - h/2, w, h, buf[5], 1, -1])
                else:
                    gt.append([idx+1, -1, x - w/2, y - h/2, w, h, buf[5], 2, -1])

            elif buf[0] == 2:
                gt.append([idx+1, -1]+ccwh_to_xywh(buf[1],buf[2],buf[3],buf[4])+[buf[5], 3, -1])
            elif buf[0] == 3:
                gt.append([idx+1, -1]+ccwh_to_xywh(buf[1],buf[2],buf[3],buf[4])+[buf[5], 4, -1])
    print(pd.DataFrame(gt))
    return gt
if __name__ == '__main__':
    pd.DataFrame(convert_yolov7_to_coco(os.getcwd()+'/labels', os.getcwd() + '/SNMOT-060/img1')).to_csv(os.getcwd()+'/gt.txt',header=None,index=None)
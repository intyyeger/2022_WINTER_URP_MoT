import cv2
import numpy as np
import pandas as pd
import os
from sklearn.cluster import KMeans
import argparse
import glob
from tqdm import tqdm

os.environ["OMP_NUM_THREADS"] = "1" 

def createDirectory(directory):
    try:
        if not os.path.exists(directory):
            os.makedirs(directory)
    except OSError:
        print("Error: Failed to create the directory.")

parser = argparse.ArgumentParser(description='Team Classification Algorithm')
parser.add_argument('--source_jpg', type=str, default='/SNMOT-060/img1', help='img forder in yolov7')
parser.add_argument('--source_label', type=str, default='/runs/detect/exp2/labels', help='img label forder in yolov7')
parser.add_argument('--name', type = str, default='find_team', help='save results to project/name')
parser.add_argument('--display', action='store_true', help='show label on image / video')
args = parser.parse_args()

createDirectory("./"+args.name)

input_path = os.getcwd() + args.source_label +"/*.txt"
file_list = [f for f in glob.glob(input_path)]
input_path = os.getcwd() + args.source_jpg +"/*.jpg"
image_list = [f for f in glob.glob(input_path)]

for s in tqdm(range(len(image_list))):
    src = cv2.imread(image_list[s], 1)

    with open(file_list[s]) as f:
        label = [line.rstrip('\n') for line in f]

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

    data = pd.DataFrame(np.array(data))
    model = KMeans(n_clusters=2)
    model.fit(data)
    centers = model.cluster_centers_
    pred = model.predict(data)

    idx = 0
    new_label = []
    for i in label:
        label_list = list(map(float, i.split()))
        if label_list[0] == 1: # player
            if pred[idx] == 0:
                label_list.append(0)
            else:
                label_list.append(1)
            idx+=1
        else:
            label_list.append(-1)
        new_label.append(label_list)

    pd.DataFrame(new_label).to_csv('./'+args.name+'/{0:0>6}.txt'.format(s+1),header=None,index=None)


if args.display:
#catch = 0
    '''
    for catch, i in enumerate(label):
        label_list = list(map(float, i.split()))
        if label_list[0] == 1:
            x = int(label_list[1]*1920)+1
            y = int(label_list[2]*1080)+1
            w = int(label_list[3]*1920)+1
            h = int(label_list[4]*1080)+1
            crop_image = src[int(y-h/2) : int(y+h/2), int(x-w/2):int(x+w/2),:]
            if pred[catch] == 0:
                src = cv2.rectangle(src, (int(x-w/2), int(y-h/2)),(int(x+w/2), int(y+h/2)), (255,0,0),3)
            else:
                src = cv2.rectangle(src, (int(x-w/2), int(y-h/2)),(int(x+w/2), int(y+h/2)), (0,255,0),3)
            #catch = catch + 1

        else:
            pass

    cv2.imshow("Classification", src)
    cv2.waitKey(0)
    '''
    print("Not implemented yet")

print("Completed!")
print("Please check the "+args.name+" folders")

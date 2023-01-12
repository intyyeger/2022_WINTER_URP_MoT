import os
import numpy as np
import pandas as pd
import argparse
import glob

def createDirectory(directory):
    try:
        if not os.path.exists(directory):
            os.makedirs(directory)
    except OSError:
        print("Error: Failed to create the directory.")

def tragectory_converter(args):

    input_path = os.getcwd() + args.project+'/'+args.name + "/label/*.txt"
    createDirectory("./"+args.stopover)
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
                track_list.append([frame+1, obj[1], obj[2]])
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
                    tmp = np.array(buf)[:,1:] #duplicated matrix
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
                miss_comp.append([it, miss_comp[-1][1] + dx, miss_comp[-1][2] + dy])

            miss_comp.append(dup)
            idx = dup[0]+1

    pd.DataFrame(miss_comp).to_csv('./'+args.stopover+'/ball_tragectory.txt',header=None,index=None)

    print("ball tragectory convering Completed")
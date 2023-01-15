import cv2
import numpy as np
import glob
from tqdm import tqdm

from marker import mark_object, commit_offside
from find_offside_line import find_offside_line



def draw(folder_path, offside):

    img_path = folder_path + '/*'
    img_list = [img for img in glob.glob(f'{img_path}')]

    print('Drawing...')

    cnt = 0

    if offside == True:
        vanishing_x, vanishingn_y = find_offside_line()

    for i in tqdm(range(len(img_list))):

        img_i = img_list[i]
        img = cv2.imread(img_i)
        print(img_i)

        if offside == True:

            # 선 긋고 판단하자
            # commit_offside(img) 오프사이드면 위험 표시            
            pass

        mark_object(img)


        

        cnt+=1
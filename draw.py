import cv2
import numpy as np
import glob
from tqdm import tqdm

from marker import mark_object, warn_offside
from find_offside_line import find_threshold, find_offside_line


def draw(folder_path, offside, gt):

    img_path = folder_path + '/*' + '.png'
    img_list = [img for img in glob.glob(f'{img_path}')]

    print('Drawing...')

    prev_x, prev_y = 0, 0

    idx = 0
    for i in tqdm(range(len(img_list))):

        img_i = img_list[i]
        img = cv2.imread(img_i)
        
        # frame = i + 1
        # while gt[idx][0] == frame:

        #     mark_object(img, int(gt[idx][2] + 0.5*gt[idx][4]), int(gt[idx][3]), int(gt[idx][7]))
        #     print(int(gt[idx][2] + 0.5*gt[idx][4]), int(gt[idx][3]), int(gt[idx][7]))
        #     idx += 1

        # cv2.imshow('img', img)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

        if offside == True:

            if i == 0:
                r_x, r_y, r_w, r_h = cv2.selectROI("ROI", img, False)
                img_thr = find_threshold(img, r_x, r_y, r_w, r_h)

            vanishing_x, vanishing_y = find_offside_line(img_i, r_x, r_y, r_w, r_h, img_thr)
            
        #     if vanishing_x == 0 and vanishing_y == 0:

        #         vanishing_x, vanishing_y = prev_x, prev_y
        #         # print(i, vanishing_x, vanishing_y)
        #         # prev_x, prev_y에서 선 긋기
        #     else:
        #         prev_x, prev_y = vanishing_x, vanishing_y
                
        #     print(vanishing_x, vanishing_y)

        #     # vanishing에서 선 긋기
        #     prev_x, prev_y = vanishing_x, vanishing_y
        #     # print(i, vanishing_x, vanishing_y)

        #     # 공격수가 선 넘었는지 판단해서 표시해주기
        #     # commit_offside(img) 오프사이드면 위험 표시            
        #     pass

# draw('C:/Users/y/Desktop/URP/test', True)
# draw('C:/Users/y/Desktop/URP/detect_line', True)
draw('C:/Users/y/Desktop/URP/test', True, [1])
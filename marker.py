import cv2
import numpy as np


def mark_object(img, x, y, type):

    start_x = x
    start_y = y - 10

    pts = np.array([[start_x,start_y], [start_x+10,start_y-17], [start_x-10,start_y-17]], dtype=np.int32)
    if type == 0:
        cv2.fillPoly(img, [pts], (236, 95,52)) # ball
    elif type == 1:
        cv2.fillPoly(img, [pts], (133,66,249)) # team 1
    elif type == 2:
        cv2.fillPoly(img, [pts], (208,45,153)) # team 2
    elif type == 3:
        cv2.fillPoly(img, [pts], (30,30,30)) # referee
    elif type == 4:
        cv2.fillPoly(img, [pts], (100,100,100)) # goalkeeper


def warn_offside(img):

    w, h = img.shape[:2]

    pts = np.array([[0,0], [0,w], [h, w], [h,0]], dtype=np.int32)
    cv2.polylines(img, [pts], True, (0, 0, 255), 10)


# mark_object('C:/Users/ys102/Desktop/URP/1.jpg', 500, 500, 0)
# commit_offside('C:/Users/ys102/Desktop/URP/1.jpg')
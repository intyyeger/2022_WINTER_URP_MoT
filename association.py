# from ByteTrack.yolox.tracker.byte_tracker import BYTETracker
from tqdm import tqdm
import cv2

class BYTETrackerArgs:
    track_thresh: float = 0.25
    track_buffer: int = 30
    match_thresh: float = 0.8
    aspect_ratio_thresh: float = 3.0
    min_box_area: float = 1.0
    mot20: bool = False


def yolo2byte():

    src = cv2.imread('image_path')

    r_x, r_y, r_w, r_h = cv2.selectROI("ROI", src, False)
    team1_color = src[r_y:r_y+r_h, r_x:r_x+r_w]

    team1_color = cv2.mean(team1_color)

    r_x, r_y, r_w, r_h = cv2.selectROI("ROI", src, False)
    team2_color = src[r_y:r_y+r_h, r_x:r_x+r_w]

    team2_color = cv2.mean(team2_color)

    print(team1_color)
    print(team2_color)



# def tracking():

#     for frame in tqdm()

#     tracker = BYTETracker(BYTETrackerArgs())

#     for image in images:
#         dets = detector(image)
#         online_targets = tracker.update(dets, info_imgs, img_size)

yolo2byte()
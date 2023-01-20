from ByteTrack.yolox.tracker.byte_tracker import BYTETracker
from tqdm import tqdm
import cv2

class BYTETrackerArgs:
    track_thresh: float = 0.25
    track_buffer: int = 30
    match_thresh: float = 0.8
    aspect_ratio_thresh: float = 3.0
    min_box_area: float = 1.0
    mot20: bool = False


def transform_det(det):
    
    team1_det = []
    team2_det = []
    ret_det = []

    for i in range(len(det)):
        if det[i][7] == 1 or det[i][7] == 2:
            ret_det.append(det[i])

        if det[i][7] == 1:
            team1_det.append(det[i])
        elif det[i][7] == 2:
            team2_det.append(det[i])

    return team1_det, team2_det

    

def class_detection(gt):

    return gt

def tracking(det):


    tracker = BYTETracker(BYTETrackerArgs())

    det = transform_det(det)
    for frame in range(5):

        team1_detections, team2_detections = transform_det(det)

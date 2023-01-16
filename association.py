from ByteTrack.yolox.tracker.byte_tracker import BYTETracker
from tqdm import tqdm


class BYTETrackerArgs:
    track_thresh: float = 0.25
    track_buffer: int = 30
    match_thresh: float = 0.8
    aspect_ratio_thresh: float = 3.0
    min_box_area: float = 1.0
    mot20: bool = False


def yolo2byte():
    pass



def tracking():

    for frame in tqdm()

    tracker = BYTETracker(BYTETrackerArgs())

    for image in images:
        dets = detector(image)
        online_targets = tracker.update(dets, info_imgs, img_size)
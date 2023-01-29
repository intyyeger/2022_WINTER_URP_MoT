import cv2
import torch
import numpy as np
import glob
from tqdm import tqdm

from marker import mark_object, warn_offside
from ByteTrack.yolox.tracker.byte_tracker import BYTETracker, STrack
from find_offside_line import find_threshold, find_offside_line, return_line


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

    for i in range(len(det)):

        if det[i][7] == 1:
            team1_det.append([det[i][2], det[i][3], det[i][2] + det[i][4], det[i][3] +det[i][5], det[i][6], det[i][7]])
        elif det[i][7] == 2:
            team2_det.append([det[i][2], det[i][3], det[i][2] + det[i][4], det[i][3] +det[i][5], det[i][6], det[i][7]])

    return team1_det, team2_det


def tracking(folder_path, det, offside):

    img_path = folder_path + '/*' + '.jpg'
    img_list = [img for img in glob.glob(f'{img_path}')]

    tracker = BYTETracker(BYTETrackerArgs())

    total_det = 0
    total_track = 0

    det_idx = 0
    print('Tracking...')
    for i in range(len(img_list)):

        img_i = img_list[i]
        img = cv2.imread(img_i)

        last_defender = []
        front_forward = []
        frame_det = []
        
        frame = i + 1
        while det_idx < len(det) and det[det_idx][0] == frame:

            mark_object(img, int(det[det_idx][2] + 0.5*det[det_idx][4]), int(det[det_idx][3]), int(det[det_idx][7]))
            if det[det_idx][7] == 1:
                last_defender.append([det[det_idx][2], det[det_idx][3] + det[det_idx][5]])
            elif det[det_idx][7] == 2:
                front_forward.append([det[det_idx][2], det[det_idx][3] + det[det_idx][5]])

            if det[det_idx][7] == 1 or det[det_idx][7] == 2:
                frame_det.append(det[det_idx])
            # if det[det_idx][6] == 1 or det[det_idx][6] == 2:
            #     frame_det.append(det[det_idx])
                
            det_idx += 1

        # draw offside line
        if offside == True:

            if i == 0:
                r_x, r_y, r_w, r_h = cv2.selectROI("ROI", img, False)
                img_thr = find_threshold(img, r_x, r_y, r_w, r_h)

            vanishing_x, vanishing_y = find_offside_line(img_i, r_x, r_y, r_w, r_h, img_thr)
            if vanishing_x == 0 and vanishing_y == 0:
                vanishing_x, vanishing_y = prev_x, prev_y
            prev_x, prev_y = vanishing_x, vanishing_y

            ret_y_int = 999999
            for def_idx in range(len(last_defender)):
                [m, y_int] = return_line(vanishing_x, vanishing_y, last_defender[def_idx][0], last_defender[def_idx][1])

                if y_int > 0 and ret_y_int > y_int: ret_y_int = y_int

            ret_y_int_f = 999999
            for for_idx in range(len(front_forward)):
                [m_f, y_int_f] = return_line(vanishing_x, vanishing_y, front_forward[for_idx][0], front_forward[for_idx][1])

                if y_int_f > 0 and ret_y_int_f > y_int_f: ret_y_int_f = y_int_f

            if ret_y_int_f < ret_y_int: warn_offside(img)
            
            cv2.line(img,(vanishing_x, vanishing_y),(0, int(ret_y_int)),(0,0,255),2)
            cv2.line(img,(vanishing_x, vanishing_y),(0, int(ret_y_int_f)),(0,255,255),2)

            # cv2.imshow('img', img)
            # cv2.waitKey(0)
            # cv2.destroyAllWindows()
        # draw offside line
            

        team1_detections, team2_detections = transform_det(frame_det)
        tracked_detections = team1_detections + team2_detections
        tracked_detections = torch.tensor(tracked_detections)
        # print(tracked_detections)

        tracking = tracker.update(output_results=tracked_detections,
                                img_info=img.shape, img_size=img.shape)

        for t in tracking:

            # t_x = int((t.tlbr[0] + t.tlbr[2])/2)
            # t_y = int((t.tlbr[1] + t.tlbr[3])/2)
            t_x = int((t.tlbr[0] + t.tlbr[2])/2)
            t_y = int(t.tlbr[3])
            t_id = str(t.track_id)

            cv2.putText(img, t_id, (t_x, t_y), cv2.FONT_HERSHEY_PLAIN, 2, (255, 204, 255), 2, cv2.LINE_AA)

        
        if i >= 70:
            cv2.imshow('img', img)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        # cv2.imshow('img', img)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

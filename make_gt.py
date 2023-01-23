import cv2
import torch
import numpy as np
import glob
from tqdm import tqdm
from convert_yolov7_to_coco import *

#from marker import mark_object, warn_offside
from yolox.tracker.byte_tracker import BYTETracker, STrack
#from find_offside_line import find_threshold, find_offside_line, return_line

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

    

def class_detection(gt):

    return gt

# def match_detections_with_tracks(
#     detections: List[Detection], 
#     tracks: List[STrack]
# ) -> List[Detection]:
#     detection_boxes = detections2boxes(detections=detections, with_confidence=False)
#     tracks_boxes = tracks2boxes(tracks=tracks)
#     iou = box_iou_batch(tracks_boxes, detection_boxes)
#     track2detection = np.argmax(iou, axis=1)
    
#     for tracker_index, detection_index in enumerate(track2detection):
#         if iou[tracker_index, detection_index] != 0:
#             detections[detection_index].tracker_id = tracks[tracker_index].track_id
#     return detections

def track2det():

    detections = []

    return detections

def tracking(folder_path, det, offside):
    detection_final = []
    tracking_final = []
    id_final = []
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
        frame_det = []
        
        frame = i + 1
        while det_idx < len(det) and det[det_idx][0] == frame:

            mark_object(img, int(det[det_idx][2] + 0.5*det[det_idx][4]), int(det[det_idx][3]), int(det[det_idx][7]))
            last_defender.append([det[det_idx][2], det[det_idx][3] + det[det_idx][5]])

            if det[det_idx][7] == 1 or det[det_idx][7] == 2:
                frame_det.append(det[det_idx])
            # if det[det_idx][6] == 1 or det[det_idx][6] == 2:
            #     frame_det.append(det[det_idx])
                
            det_idx += 1

        # draw offside line
        # if offside == True:

        #     if i == 0:
        #         r_x, r_y, r_w, r_h = cv2.selectROI("ROI", img, False)
        #         img_thr = find_threshold(img, r_x, r_y, r_w, r_h)

        #     vanishing_x, vanishing_y = find_offside_line(img_i, r_x, r_y, r_w, r_h, img_thr)
        #     if vanishing_x == 0 and vanishing_y == 0:
        #         vanishing_x, vanishing_y = prev_x, prev_y
        #     prev_x, prev_y = vanishing_x, vanishing_y

        #     ret_y_int = 2000
        #     for def_idx in range(len(last_defender)):
        #         [m, y_int] = return_line(vanishing_x, vanishing_y, last_defender[def_idx][0], last_defender[def_idx][1])

        #         if y_int > 0 and ret_y_int > y_int: ret_y_int = y_int
            
        #     cv2.line(img,(vanishing_x, vanishing_y),(0, int(ret_y_int)),(0,0,255),2)

        #     cv2.imshow('img', img)
        #     cv2.waitKey(0)
        #     cv2.destroyAllWindows()
        # draw offside line
            

        team1_detections, team2_detections = transform_det(frame_det)
        tracked_detections = team1_detections + team2_detections
        tracked_detections = torch.tensor(tracked_detections)
        # print(tracked_detections)

        tracking = tracker.update(output_results=tracked_detections,
                                img_info=img.shape, img_size=img.shape)

        total_det += len(tracked_detections)
        total_track += len(tracking)

        # if len(tracked_detections) != len(tracking):
        #     print(frame)
        #     print(tracked_detections)
        #     for t in tracking:
        #         print(t.tlbr, t.track_id)

        #print(tracked_detections)
        imm_tmp_id = []
        imm_tmp = []
        for t in tracking:
            #print(t.tlbr, t.track_id)
            imm_tmp.append(t.tlbr)
            imm_tmp_id.append(t.track_id)

        tracking_final.append(imm_tmp)
        id_final.append(imm_tmp_id)
        detection_final.append(tracked_detections)
        #tracking_final.append(tracking)
    print('det', total_det)
    print('track', total_track)
    return id_final, tracking_final, detection_final

def make_gt(tmp_id, tmp,result_file):#tmp_id, tmp, tmp_detect

    idx = 0
    result = []
    for frame in range(len(tmp)):#len(tmp)):
        arr = [] #각 프레임의 result file 저장

        while True:
            if result_file[idx][0]!= frame+1:
                break
            arr.append(result_file[idx])
            idx+=1
            if idx>=len(result_file):
                break
        
        for t_idx, t in enumerate(tmp[frame]):
            select = np.argmin(np.sum((np.array(arr)[:,2:4] - t[:2])**2, axis = 1))
            arr[select][1] = tmp_id[frame][t_idx]
        result.append(arr)
    gt = []    
    for r_idx, r in enumerate(result):
        if r_idx ==0:
            gt = r
        else:
            gt+=r
    #print(pd.DataFrame(gt))
    return gt

def make_gt_compare(tmp_id, tmp,result_file):#tmp_id, tmp, tmp_detect

    idx = 0
    result = []
    for frame in range(len(tmp)):#len(tmp)):
        arr = [] #각 프레임의 result file 저장

        while True:
            if result_file[idx][0]!= frame+1:
                break
            arr.append(result_file[idx])
            idx+=1
            if idx>=len(result_file):
                break
        
        for t_idx, t in enumerate(tmp[frame]):
            select = np.argmin(np.sum((np.array(arr)[:,2:4] - t[:2])**2, axis = 1))
            arr[select][1] = tmp_id[frame][t_idx]
        result.append(arr)
    gt = []    
    for r_idx, r in enumerate(result):
        if r_idx ==0:
            gt = r
        else:
            gt+=r

    rrr = np.array(gt)

    rrr[:,-2] = -1
    add = np.array([-1]*len(rrr)).reshape(-1,1)
    rrr = np.concatenate((rrr, add), axis = 1)
    #pd.DataFrame(rrr)

    return rrr

if __name__=="__main__":
    result_file = convert_yolov7_to_coco(os.getcwd()+'/data/labels', os.getcwd() + '/data/img1')
    tmp_id, tmp, tmp_detect = tracking( os.getcwd() + '/data/img1',  result_file, True)


    rrr = make_gt_compare(tmp_id, tmp,result_file)
    rrr = pd.DataFrame(rrr)
    print(rrr)
    rrr.to_csv("gt_new.txt",header=None,index=None)
    #!python ./tools/interpolation.py -> data_root에 data있는 파일 root, txt_path에 {os.getcwd() + "/gt_new.txt"} 이런식

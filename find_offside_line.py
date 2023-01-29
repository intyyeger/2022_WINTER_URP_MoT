import cv2
import numpy as np
import matplotlib.pyplot as plt
import glob
from tqdm import tqdm


def return_green_area(src_img): # image green 제외하고 black으로 mask

    hsv_img = cv2.cvtColor(src_img, cv2.COLOR_BGR2HSV)
    hsv_mask = cv2.inRange(hsv_img, (40, 80, 80), (80, 255, 255))    # bound 수정 가능
    ret_img = cv2.bitwise_and(src_img, src_img, mask=hsv_mask)

    ret_img = cv2.cvtColor(ret_img, cv2.COLOR_HSV2BGR)

    return ret_img


def return_k_means(src_img, k): # k개의 색으로 그림화

    t_img = src_img.reshape((-1, 3)).astype(np.float32)
    criteria = (cv2.TERM_CRITERIA_MAX_ITER + cv2.TERM_CRITERIA_EPS, 10, 0.1)
    
    ret,label,center = cv2.kmeans(t_img, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

    center = np.uint8(center)
    res = center[label.flatten()]
    res = res.reshape((src_img.shape))

    return res


def return_line(x1, y1, x2, y2): # 기울기, y절편 리턴
    
    if x1 == x2:
        return (0, 0)

    m = (y2 - y1) / (x2 - x1)
    y_int = y1 - m*x1

    return [m, y_int]


def return_intersection(lines, points):

    num_lines = len(lines)
    if num_lines < 2: return (0, 0)

    total_x, total_y = 0, 0
    cnt = 0

    for i in range(num_lines):
        
        for j in range(i + 1, num_lines):

            if lines[i][0] == lines[j][0]: continue # 평행

            ret_x = int((lines[j][1] - lines[i][1]) / (lines[i][0] - lines[j][0]))
            ret_y = int(lines[i][0] * (lines[j][1] - lines[i][1]) / (lines[i][0] - lines[j][0]) + lines[i][1])

            if not( -2500 < ret_y < -2000): continue # 여기 숫자로 정확도 조정 이미지 크기마다 다름

            total_x += ret_x
            total_y += ret_y
            cnt += 1

    if cnt == 0 : return (0, 0)

    total_x = int(total_x / cnt)
    total_y = int(total_y / cnt)

    return (total_x, total_y)


def find_threshold(img, r_x, r_y, r_w, r_h):

    crop_img = img[r_y:r_y+r_h, r_x:r_x+r_w]
    crop_img = return_green_area(crop_img)
    k_means_crop_img = return_k_means(crop_img, 4)

    # img_gray = cv2.cvtColor(k_means_crop_img, cv2.COLOR_BGR2GRAY)  # grayscale로 변환
    # img_gray_norm = cv2.normalize(img_gray, None, 0, 255, cv2.NORM_MINMAX) # normalize
    # img_gray_norm = 2*img_gray_norm

    (h,s,v) = cv2.split(k_means_crop_img)
    img_gray = v
    img_gray_norm = cv2.normalize(img_gray, None, 0, 255, cv2.NORM_MINMAX) # normalize
    img_gray_norm = 2*img_gray_norm
    
    r_x2, r_y2, r_w2, r_h2 = cv2.selectROI("ROI", img_gray_norm, False)  # 가장 진한것과 두번째로 진한것 같이 잡는게 잘됨
    check_img = img_gray_norm[r_y2:r_y2+r_h2, r_x2:r_x2+r_w2]
    avg_check_img = cv2.mean(check_img)[0]

    return avg_check_img


def find_offside_line(path, r_x, r_y, r_w, r_h, avg_check_img):

    img = cv2.imread(path) # 이미지 경로

    # r_x, r_y, r_w, r_h = cv2.selectROI("ROI", img, False) # 관심 영역 추출, space or enter 눌러야 창 종료, c로 취소
    crop_img = img[r_y:r_y+r_h, r_x:r_x+r_w] # ROI 영역으로 자르기
    crop_img = return_green_area(crop_img) # 필드 제외 마스킹
    k_means_crop_img = return_k_means(crop_img, 4) # 그림화

    # img_gray = cv2.cvtColor(k_means_crop_img, cv2.COLOR_BGR2GRAY)  # grayscale로 변환
    (h,s,v) = cv2.split(k_means_crop_img)
    img_gray = v
    img_gray_norm = cv2.normalize(img_gray, None, 0, 255, cv2.NORM_MINMAX) # normalize
    img_gray_norm = 2*img_gray_norm

    ret, thr = cv2.threshold(img_gray_norm, avg_check_img, 255, cv2.THRESH_BINARY)

    canny = cv2.Canny(thr, 50, 255, apertureSize=3)

    lines = cv2.HoughLines(canny,1,np.pi/180,180)

    line_points = []
    offside_lines = []
    point_cache = []

    cnt = 0

    if lines is None:
        return (0, 0)

    for i in range(len(lines)):
        for rho, theta in lines[i]:

            a = np.cos(theta)
            b = np.sin(theta)

            if -0.9 < a < 0.9: continue    # 여기서 각도 조절

            x0 = a*rho
            y0 = b*rho
            
            x1 = int(x0 + 1000*(-b))
            y1 = int(y0+1000*(a))
            x2 = int(x0 - 1000*(-b))
            y2 = int(y0 -1000*(a))

            [ret_m, ret_y_int] = return_line(r_x+x1,r_y+y1,r_x+x2,r_y+y2)
            if ret_m == 0 : continue
            
            point_cache.append([x0, y0])
            line_points.append([r_x+x1,r_y+y1,r_x+x2,r_y+y2])
            offside_lines.append([ret_m, ret_y_int])

            # cv2.line(img,(r_x+x1,r_y+y1),(r_x+x2,r_y+y2),(255,0,0),2)

            break

    v_x, v_y = return_intersection(offside_lines, line_points)

    # cv2.line(img,(v_x, v_y),(1000, 1080),(0,0,255),2)
    # cv2.imshow('img', img)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    return (v_x, v_y)
    # 0, 0이면 검출x
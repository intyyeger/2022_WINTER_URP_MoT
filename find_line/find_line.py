import cv2
import numpy as np
# from shaply.geometry import Point  이거 왜 import 안되노
import matplotlib.pyplot as plt


def return_green_area(src_img): # image green 제외하고 black으로 mask

    hsv_img = cv2.cvtColor(src_img, cv2.COLOR_BGR2HSV)
    hsv_mask = cv2.inRange(hsv_img, (40, 80, 80), (80, 255, 255))    # bound 수정 가능
    ret_img = cv2.bitwise_and(src_img, src_img, mask=hsv_mask)

    return ret_img

def return_k_means(src_img, k): # k개의 색으로 그림화

    t_img = src_img.reshape((-1, 3)).astype(np.float32)
    criteria = (cv2.TERM_CRITERIA_MAX_ITER + cv2.TERM_CRITERIA_EPS, 10, 0.1)
    
    ret,label,center = cv2.kmeans(t_img, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

    center = np.uint8(center)
    res = center[label.flatten()]
    res = res.reshape((src_img.shape))

    return res

def black_to_green(src_img):

    hsv_img = cv2.cvtColor(src_img, cv2.COLOR_BGR2HSV)
    # 미완




img = cv2.imread('your/image/path') # 이미지 경로

# return_green_area_img = return_green_area(img)
# k_means_img = return_k_means(img, 5)

# green으로 변환된 이미지 보기
# plt.imshow(return_green_area_img)
# plt.show()

# k-means로 변환된 이미지 보기
# plt.imshow(k_means_img)
# plt.show()

r_x, r_y, r_w, r_h = cv2.selectROI("ROI", img, False) # 관심 영역 추출, space or enter 눌러야 창 종료, c로 취소
crop_img = img[r_y:r_y+r_h, r_x:r_x+r_w]
crop_img = return_green_area(crop_img)
k_means_crop_img = return_k_means(crop_img, 4)
# plt.imshow(k_means_crop_img)
# plt.show()
# 관심역역 이미지 보기
# plt.imshow(crop_img)
# plt.show()

img_gray = cv2.cvtColor(k_means_crop_img, cv2.COLOR_BGR2GRAY)
img_gray_norm = cv2.normalize(img_gray, None, 0, 255, cv2.NORM_MINMAX)
plt.imshow(img_gray_norm)
plt.show()

r_x2, r_y2, r_w2, r_h2 = cv2.selectROI("ROI", img_gray_norm, False)
check_img = img_gray_norm[r_y2:r_y2+r_h2, r_x2:r_x2+r_w2]

avg_check_img = cv2.mean(check_img)[0]
print(avg_check_img)

ret, thr = cv2.threshold(img_gray_norm, avg_check_img, 255, cv2.THRESH_BINARY)

# 이진화 이미지 보기
# plt.imshow(thr)
# plt.show()

canny = cv2.Canny(thr, 50, 200, apertureSize=3)

# canny 보기
plt.imshow(canny)
plt.show()

lines = cv2.HoughLines(canny,1,np.pi/180,150)

for i in range(len(lines)):
    for rho, theta in lines[i]:
        a = np.cos(theta)
        b = np.sin(theta)

        if -0.5 < a < 0.5: continue    # 여기서 각도 조절

        x0 = a*rho
        y0 = b*rho
        print("x: ", x0, "y: ", y0)
        x1 = int(x0 + 1000*(-b))
        y1 = int(y0+1000*(a))
        x2 = int(x0 - 1000*(-b))
        y2 = int(y0 -1000*(a))

        cv2.line(img,(r_x+x1,r_y+y1),(r_x+x2,r_y+y2),(255,0,0),2)

plt.imshow(img)
plt.show()
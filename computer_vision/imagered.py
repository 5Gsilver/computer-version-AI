import cv2
import numpy as np

# 이미지 불러오기
img = cv2.imread('sample.jpg')

# 이미지 복사하기
img_clone = img.copy()

# BGR을 HSV로 변환
hsv_img = cv2.cvtColor(img_clone, cv2.COLOR_BGR2HSV)

# 빨간색 범위 정의 (HSV에서)
# 빨간색은 HSV에서 0도와 180도 근처에 있음
lower_red1 = np.array([0, 50, 50])      # 첫 번째 빨간색 범위 (0-10도)
upper_red1 = np.array([10, 255, 255])
lower_red2 = np.array([170, 50, 50])    # 두 번째 빨간색 범위 (170-180도)
upper_red2 = np.array([180, 255, 255])

# 빨간색 마스크 생성
mask_red1 = cv2.inRange(hsv_img, lower_red1, upper_red1)
mask_red2 = cv2.inRange(hsv_img, lower_red2, upper_red2)
mask_red = cv2.bitwise_or(mask_red1, mask_red2)  # 두 범위 합치기

# 빨간색 마스크
cv2.imshow('mask_red', mask_red)

# 빨간색 부분만 추출
red_only = cv2.bitwise_and(img_clone, img_clone, mask=mask_red)

# 이미지 출력
cv2.imshow('red_only', red_only)

# q를 누르면 종료하기
while True:
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break

# 윈도우 삭제
cv2.destroyAllWindows()
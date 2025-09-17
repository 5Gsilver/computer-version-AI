import cv2

# 이미지 파일 읽기
image = cv2.imread('sample.jpg')

# 이미지 복사
image_clone = image.copy()

cv2.imshow("image",image)
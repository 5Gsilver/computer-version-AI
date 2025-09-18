import requests
import cv2
import numpy as np

# 이미지 URL
image_url = "https://images.unsplash.com/photo-1628526498666-add5eddf65df?w=640"

# 저장할 파일 경로 및 이름
file_name = 'sample.jpg'

# 다운로드 요청
response = requests.get(image_url)

# 다운로드 요청이 성공했는지 확인
if response.status_code == 200:
    with open(file_name, 'wb') as f:
        f.write(response.content)
    print('성공적으로 저장되었습니다.')
else:
    print('다운로드하는 데 문제가 발생했습니다. 상태 코드:', response.status_code)

# 이미지 불러오기
img = cv2.imread('sample.jpg')

# 이미지 복사하기
img_clone = img.copy()

# BGR 채널 분리
b_channel, g_channel, r_channel = cv2.split(img_clone)

# Red 채널만 보기
red_img = cv2.merge([np.zeros_like(r_channel), np.zeros_like(r_channel), r_channel])

# 이미지 출력
cv2.imshow('image', red_img)

# q를 누르면 종료하기
while True:
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break

# 윈도우 삭제
cv2.destroyAllWindows()
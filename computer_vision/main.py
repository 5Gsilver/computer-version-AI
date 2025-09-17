import requests
import cv2

# 이미지 URL
image_url = "https://images.unsplash.com/photo-1542037104857-ffbb0b9155fb?w=640"

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
    
# # 이미지 파일 읽기
image = cv2.imread(file_name)

cv2.imshow('Downloaded Image', image)
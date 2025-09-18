import requests

# URL
image_url = 'https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_frontalface_default.xml'

# 저장할 파일 경로 및 이름
file_name = 'haarcascade_frontalface_default.xml'

# 다운로드 요청
response = requests.get(image_url)

# 다운로드 요청이 성공했는지 확인
if response.status_code == 200:
    with open(file_name, 'wb') as f:
        f.write(response.content)
    print('성공적으로 저장되었습니다.')
else:
    print('다운로드하는 데 문제가 발생했습니다. 상태 코드:', response.status_code)
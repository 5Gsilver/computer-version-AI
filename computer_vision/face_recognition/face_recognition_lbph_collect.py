import cv2
import os
import re
import time


# 정규 표현식 패턴: 소문자 알파벳만 포함하고 최소 3자 이상
pattern = re.compile("^[a-z]{3,}$")

while True:
    # 사용자로부터 이름을 입력받음
    user_name = input("소문자 영어 이름을 입력하세요 (최소 3자 이상): ")

    # 입력된 이름이 패턴과 일치하는지 확인
    if pattern.match(user_name):
        break
    else:
        print("유효하지 않은 입력입니다. 소문자 영어 알파벳으로 최소 3자 이상 입력해야 합니다.")

# 새로운 디렉토리 경로 설정
new_directory = os.path.join("./faces", user_name)

# 디렉토리가 존재하는지 확인
if os.path.exists(new_directory):
    # 디렉토리 안의 파일 목록을 가져옴
    files_in_directory = os.listdir(new_directory)
    
    if files_in_directory:
        print(f"{new_directory} 디렉토리가 이미 존재하며, 파일이 있습니다.")
        # 프로그램 종료
        exit()
    else:
        print(f"{new_directory} 디렉토리가 이미 존재하지만, 비어 있습니다. 계속 진행합니다.")
else:
    # 디렉토리가 존재하지 않으면 생성
    os.makedirs(new_directory)
    print(f"{new_directory} 디렉토리가 생성되었습니다.")

# 카메라 선언
cap = cv2.VideoCapture(0)

# 크기 설정
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# 카메라 연결 확인
if not cap.isOpened():
    print("카메라를 연결할 수 없습니다.")
    exit()

# 분류기 
detector = cv2.CascadeClassifier("haarcascade_frontalface_default.xml") 

# 순번: 이름_순번.png 파일로 만들 때 사용
count = 1

# 현재 시간(시작 시간)
start_time = time.time()

# 사진을 찍기위한 딜레이(몇초에 한번씩 찍을지 결정할 때 사용)
delay_time = 1

while True:
    # 응답, 프레임
    ret, frame = cap.read()
    
    # 흑백처리(BGR -> GRAY)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) 
 
    # 얼굴 찾기 
    faces = detector.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)) 
    
    # 가장 큰 얼굴 하나만 골라내기
    max_face = None
    max_face_area = 0
    for (x, y, w, h) in faces:
        face_area = w * h
        if face_area > max_face_area:
            max_face = (x, y, w, h)
            max_face_area = face_area
        
    # delay_time 마다 사진을 찍음
    current_time = time.time()
    if current_time - start_time >= delay_time:
        start_time = current_time
        if max_face:
            x, y, w, h = max_face
            face_img = frame[y:y+h, x:x+w]            
            cv2.imwrite(f"faces/{user_name}/{count}.png", face_img)
            print(f"이미지 저장 => faces/{user_name}/{count}.png")
            count += 1
            
            # 찾은 얼굴에 사각형 그리기
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 1)     
            
    # 프레임 보여주기
    cv2.imshow("camera", frame)
    
    # q를 누르면 종료하기
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# 카메라 해제    
cap.release()

# 윈도우 삭제
cv2.destroyAllWindows()

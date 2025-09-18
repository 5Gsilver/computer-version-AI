import cv2
import json
import os

# 이름


names_path = os.path.join(os.path.dirname(__file__), 'names.json')
with open(names_path, 'r') as f:
    train_names_list = json.load(f)
    train_names = {i: name for i, name in enumerate(train_names_list)}
    
# 학습 모델
model = cv2.face.LBPHFaceRecognizer_create()
model.read('train.yml')

# 카메라 선언
cap = cv2.VideoCapture(0)

# 카메라 연결 확인
if not cap.isOpened():
    print('can not open camera')
    exit()

# 분류기 
detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml') 

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
            
    if max_face:
        # 얼굴 좌표
        x, y, w, h = max_face
        face_image = frame[y:y+h, x:x+w]
        
        # 찾은 얼굴에 사각형 그리기
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 1) 
                
        # 흑백처리(BGR -> GRAY)
        gray = cv2.cvtColor(face_image, cv2.COLOR_BGR2GRAY) 
        
        # 예측하기
        result = model.predict(gray) 
        label, confidence = result
        
        if confidence < 50:
            name = train_names.get(label)
            print(f'Label: {label}, Confidence: {confidence}, {name}')    
            # 흰색 배경
            cv2.rectangle(frame, (x, y), (x + 100, y - 30), (255, 255, 255), -1)
            # 검정 글씨
            cv2.putText(frame, name, (x + 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)            
    
    # 프레임 보여주기
    cv2.imshow('camera', frame)

    # q를 누르면 종료하기
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 카메라 해제    
cap.release()

# 윈도우 삭제
cv2.destroyAllWindows()

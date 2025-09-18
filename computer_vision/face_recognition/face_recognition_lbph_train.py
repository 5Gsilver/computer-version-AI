import os
import cv2
import numpy as np
import json

# 기본 경로 설정
base_dir = './faces'

# base_image_dir 내 디렉터리 목록 가져오기
sub_dirs = [d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))]

# 결과 출력
print("하위 디렉터리 목록:", sub_dirs)

image_files = []

# 하위 디렉터리에 있는 파일 목록
for sub_dir in sub_dirs:
    sub_image_files = [os.path.join(base_dir, sub_dir, f) for f in os.listdir(os.path.join(base_dir, sub_dir)) if f.endswith('.png')]
    image_files.extend(sub_image_files)

print(image_files)

# 학습 데이터, 라벨, 이름
train_data = []
train_labels = []
train_names = {}
seq_num = 0 # 라벨

# 파일 읽어서 학습하기
for file in image_files:
    # 이미지 파일 읽어오기
    image = cv2.imread(file)

    # 흑백처리(BGR -> GRAY)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # 이미지 파일을 numpy 배열로 변환
    gray_arr = np.asarray(gray, dtype=np.uint8)

    # 파일 경로 나누기
    parts = file.split(os.sep)

    # 파일명만 가져오기
    file_name_only = parts[-1]

    # 이름
    name = parts[-2]

    # 학습 데이터, 라벨, 이름 저장
    train_data.append(gray_arr)
    train_labels.append(seq_num)
    train_names[seq_num] = name
    seq_num += 1

# 32비트 정수로 변환
train_labels = np.asarray(train_labels, dtype=np.int32)

# 학습 모델 생성
model = cv2.face.LBPHFaceRecognizer_create()

# 학습 시작
model.train(train_data, np.array(train_labels))

# 학습 모델 저장
model.write('train.yml')

# 이름 저장
with open('names.json', 'w') as f:
    json.dump(train_names, f, indent=4)

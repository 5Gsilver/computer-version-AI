# 컴퓨터 비전 & AI 학습 레포지토리

파이썬으로 컴퓨터 비전과 AI를 학습하기 위한 스터디용 레포지토리입니다.
이미지 처리, 객체 탐지, 신경망 등 컴퓨터 비전의 핵심 개념과 알고리즘을 직접 구현하며 실습 경험을 쌓는 것을 목표로 합니다.

## 📚 학습 목표

- **기초 이론 이해**: 컴퓨터 비전의 기본 개념과 원리 학습
- **실습 중심 학습**: 이론을 코드로 직접 구현하며 체득
- **단계별 학습**: 기초부터 고급까지 체계적인 학습 경로 제공
- **프로젝트 경험**: 실제 문제 해결을 통한 실무 능력 향상

## 🗂️ 레포지토리 구조

```
computer-version-AI/
├── 01_basics/              # 기초 개념 및 환경 설정
├── 02_image_processing/    # 이미지 처리 기초
├── 03_feature_detection/   # 특징 검출 및 매칭
├── 04_object_detection/    # 객체 탐지
├── 05_neural_networks/     # 신경망 기초
├── 06_deep_learning/       # 딥러닝 응용
├── 07_projects/            # 실습 프로젝트
├── datasets/               # 샘플 데이터셋
├── utils/                  # 유틸리티 함수
└── requirements.txt        # 필요한 라이브러리
```

## 🔧 환경 설정

### 1. 저장소 클론
```bash
git clone https://github.com/5Gsilver/computer-version-AI.git
cd computer-version-AI
```

### 2. 가상환경 생성 (권장)
```bash
# venv 사용
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# conda 사용
conda create -n cv-ai python=3.9
conda activate cv-ai
```

### 3. 패키지 설치
```bash
pip install -r requirements.txt
```

## 📖 학습 경로

### 🔰 1단계: 기초 (01_basics)
- Python 기초 복습
- NumPy, OpenCV 기본 사용법
- 이미지 읽기/쓰기/표시

### 🖼️ 2단계: 이미지 처리 (02_image_processing)
- 이미지 필터링 (블러, 샤프닝)
- 형태학적 연산
- 히스토그램 균등화
- 색상 공간 변환

### 🎯 3단계: 특징 검출 (03_feature_detection)
- 에지 검출 (Canny, Sobel)
- 코너 검출 (Harris, FAST)
- 특징 기술자 (SIFT, ORB)
- 특징 매칭

### 🔍 4단계: 객체 탐지 (04_object_detection)
- 템플릿 매칭
- HOG + SVM
- Haar Cascade
- YOLO 기초

### 🧠 5단계: 신경망 (05_neural_networks)
- 퍼셉트론 구현
- 다층 신경망
- 역전파 알고리즘
- 활성화 함수

### 🚀 6단계: 딥러닝 (06_deep_learning)
- CNN 구조 이해
- 전이 학습
- 객체 검출 모델
- 이미지 분할

### 🛠️ 7단계: 프로젝트 (07_projects)
- 얼굴 인식 시스템
- 객체 추적
- 이미지 분류기
- 실시간 비디오 처리

## 🔗 주요 라이브러리

| 라이브러리 | 용도 | 버전 |
|-----------|------|------|
| OpenCV | 컴퓨터 비전 | >=4.8.0 |
| NumPy | 수치 연산 | >=1.21.0 |
| Matplotlib | 시각화 | >=3.5.0 |
| scikit-learn | 머신러닝 | >=1.0.0 |
| TensorFlow | 딥러닝 | >=2.10.0 |
| PyTorch | 딥러닝 | >=1.12.0 |

## 📋 학습 가이드

### 권장 학습 순서
1. **01_basics**: 기본 환경 세팅 및 라이브러리 숙지
2. **02_image_processing**: 이미지 처리 기법 실습
3. **03_feature_detection**: 특징 검출 알고리즘 구현
4. **04_object_detection**: 전통적인 객체 탐지 방법
5. **05_neural_networks**: 신경망 기초 이론 및 구현
6. **06_deep_learning**: 현대적인 딥러닝 기법
7. **07_projects**: 종합 프로젝트 실습

### 학습 팁
- 각 폴더의 README.md를 먼저 읽어보세요
- 코드를 직접 실행하고 결과를 확인하세요
- 매개변수를 변경해가며 실험해보세요
- 궁금한 점은 Issues에 등록해주세요

## 🤝 기여하기

이 레포지토리는 학습 목적으로 만들어졌습니다. 다음과 같은 기여를 환영합니다:

- 버그 수정
- 새로운 예제 추가
- 문서 개선
- 한국어 번역 개선

## 📄 라이센스

이 프로젝트는 MIT 라이센스 하에 있습니다. 자세한 내용은 [LICENSE](LICENSE) 파일을 참조하세요.

## 📞 문의

- GitHub Issues: 기술적 질문이나 버그 리포트
- 학습 관련 질문: Discussions 탭 활용

---

**즐거운 컴퓨터 비전 학습되세요! 🎉**
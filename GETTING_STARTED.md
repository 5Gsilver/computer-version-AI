# 시작하기 가이드

## 🚀 빠른 시작

### 1. 환경 설정
```bash
# 저장소 클론
git clone https://github.com/5Gsilver/computer-version-AI.git
cd computer-version-AI

# 가상환경 생성 (권장)
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 패키지 설치
pip install -r requirements.txt
```

### 2. 첫 번째 실습
```bash
# Python 기초 복습
python 01_basics/01_python_basics.py

# NumPy 기초 학습
python 01_basics/02_numpy_fundamentals.py

# OpenCV 시작하기
python 01_basics/03_opencv_introduction.py
```

### 3. 이미지 처리 실습
```bash
# 이미지 필터링
python 02_image_processing/01_image_filtering.py

# 임계값 처리
python 02_image_processing/05_thresholding.py
```

### 4. 실습 프로젝트
```bash
# 이미지 개선기
python 07_projects/01_image_enhancer/main.py
```

### 5. Jupyter 노트북
```bash
# Jupyter 실행
jupyter notebook computer_vision_basics.ipynb
```

## 📖 학습 순서

1. **기초 다지기** (01_basics)
   - Python 복습 → NumPy → OpenCV

2. **이미지 처리** (02_image_processing)  
   - 필터링 → 임계값 처리

3. **특징 검출** (03_feature_detection)
   - 에지 검출 → 코너 검출

4. **프로젝트** (07_projects)
   - 실무 응용 경험

## 💡 학습 팁

- 각 모듈의 README.md를 먼저 읽어보세요
- 코드를 직접 실행하고 결과를 확인하세요
- 매개변수를 바꿔가며 실험해보세요
- 이해가 안 되는 부분은 Issues에 질문하세요

## 🛠️ 문제 해결

### 패키지 설치 오류
```bash
# pip 업그레이드
pip install --upgrade pip

# 개별 설치
pip install numpy opencv-python matplotlib
```

### OpenCV 오류
```bash
# OpenCV 재설치
pip uninstall opencv-python
pip install opencv-python
```

### Jupyter 노트북 실행 안됨
```bash
pip install jupyter ipywidgets
```

## 🤔 자주 묻는 질문

**Q: Python 버전은 어떻게 되나요?**
A: Python 3.7 이상을 권장합니다.

**Q: GPU가 필요한가요?**
A: 기초 과정에서는 CPU만으로도 충분합니다.

**Q: 수학 지식이 필요한가요?**
A: 기본적인 선형대수와 미적분학 지식이 도움됩니다.

**Q: 완주하는데 얼마나 걸리나요?**
A: 개인차가 있지만, 주 5시간씩 공부하면 2-3개월 정도 소요됩니다.

즐거운 학습되세요! 🎉
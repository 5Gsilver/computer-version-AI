"""
02. NumPy 기초
이미지는 본질적으로 숫자 배열입니다. NumPy는 이런 배열을 효율적으로 처리할 수 있게 해줍니다.
"""

import numpy as np

def array_creation_demo():
    """배열 생성 방법들"""
    print("=== NumPy 배열 생성 ===")
    
    # 1차원 배열 (그레이스케일 이미지 한 줄)
    gray_line = np.array([0, 64, 128, 192, 255])
    print(f"그레이스케일 라인: {gray_line}")
    
    # 2차원 배열 (작은 이미지)
    small_image = np.array([
        [0, 128, 255],
        [64, 192, 128],
        [255, 0, 64]
    ])
    print(f"작은 이미지 (3x3):\n{small_image}")
    
    # 3차원 배열 (컬러 이미지 시뮬레이션)
    color_pixel = np.array([[[255, 0, 0]]])  # 빨간색 픽셀
    print(f"컬러 픽셀 (R,G,B): {color_pixel}")
    
    # 배열 생성 함수들
    zeros_img = np.zeros((3, 3), dtype=np.uint8)  # 검은 이미지
    ones_img = np.ones((2, 2), dtype=np.uint8) * 255  # 흰 이미지
    random_img = np.random.randint(0, 256, (2, 2), dtype=np.uint8)  # 노이즈
    
    print(f"\n검은 이미지:\n{zeros_img}")
    print(f"흰 이미지:\n{ones_img}")
    print(f"랜덤 이미지:\n{random_img}")

def array_properties_demo():
    """배열 속성 확인"""
    print("\n=== 배열 속성 ===")
    
    # 실제 이미지 크기와 유사한 배열 생성
    image = np.random.randint(0, 256, (480, 640, 3), dtype=np.uint8)
    
    print(f"이미지 형태(shape): {image.shape}")  # (높이, 너비, 채널)
    print(f"데이터 타입: {image.dtype}")
    print(f"총 요소 수: {image.size}")
    print(f"차원 수: {image.ndim}")
    print(f"메모리 크기: {image.nbytes} bytes")

def array_indexing_demo():
    """배열 인덱싱과 슬라이싱 (ROI 추출)"""
    print("\n=== 배열 인덱싱 (ROI 추출) ===")
    
    # 8x8 체스보드 패턴 생성
    image = np.zeros((8, 8), dtype=np.uint8)
    image[1::2, ::2] = 255  # 홀수 행, 짝수 열
    image[::2, 1::2] = 255  # 짝수 행, 홀수 열
    
    print("원본 체스보드:")
    print(image)
    
    # ROI (Region of Interest) 추출
    roi = image[2:6, 2:6]  # 중앙 4x4 영역
    print(f"\nROI (2:6, 2:6):\n{roi}")
    
    # 특정 픽셀 접근
    pixel_value = image[3, 3]
    print(f"픽셀 (3,3) 값: {pixel_value}")
    
    # 조건부 인덱싱 (임계값 처리)
    bright_pixels = image > 128
    print(f"밝은 픽셀 위치:\n{bright_pixels}")

def array_operations_demo():
    """배열 연산 (이미지 처리 기본)"""
    print("\n=== 배열 연산 ===")
    
    # 두 이미지 생성
    img1 = np.array([[100, 150], [200, 250]], dtype=np.uint8)
    img2 = np.array([[50, 100], [150, 200]], dtype=np.uint8)
    
    print(f"이미지 1:\n{img1}")
    print(f"이미지 2:\n{img2}")
    
    # 기본 연산
    addition = cv2.add(img1, img2)  # OpenCV의 포화 연산
    addition_np = np.clip(img1.astype(np.int16) + img2, 0, 255).astype(np.uint8)
    
    subtraction = cv2.subtract(img1, img2)
    multiplication = img1 * 0.5  # 밝기 조절
    
    print(f"\n덧셈 (OpenCV):\n{addition}")
    print(f"덧셈 (NumPy):\n{addition_np}")
    print(f"뺄셈:\n{subtraction}")
    print(f"곱셈 (밝기 50%):\n{multiplication.astype(np.uint8)}")

def statistical_operations_demo():
    """통계 연산 (이미지 분석)"""
    print("\n=== 통계 연산 ===")
    
    # 랜덤 이미지 생성
    image = np.random.randint(0, 256, (5, 5), dtype=np.uint8)
    print(f"샘플 이미지:\n{image}")
    
    # 통계값 계산
    mean_val = np.mean(image)
    std_val = np.std(image)
    min_val = np.min(image)
    max_val = np.max(image)
    
    print(f"\n평균값: {mean_val:.2f}")
    print(f"표준편차: {std_val:.2f}")
    print(f"최솟값: {min_val}")
    print(f"최댓값: {max_val}")
    
    # 축별 통계 (각 행/열의 평균)
    row_means = np.mean(image, axis=1)  # 각 행의 평균
    col_means = np.mean(image, axis=0)  # 각 열의 평균
    
    print(f"행별 평균: {row_means}")
    print(f"열별 평균: {col_means}")

def reshaping_demo():
    """배열 형태 변경 (이미지 변환)"""
    print("\n=== 배열 형태 변경 ===")
    
    # 1차원 배열을 2차원 이미지로
    flat_array = np.arange(12)
    image_2d = flat_array.reshape(3, 4)
    
    print(f"1차원 배열: {flat_array}")
    print(f"2차원 이미지:\n{image_2d}")
    
    # 2차원을 다시 1차원으로 (flatten)
    flattened = image_2d.flatten()
    print(f"다시 1차원: {flattened}")
    
    # 전치 (transpose) - 이미지 회전에 사용
    transposed = image_2d.T
    print(f"전치 이미지:\n{transposed}")

def broadcasting_demo():
    """브로드캐스팅 (효율적인 연산)"""
    print("\n=== 브로드캐스팅 ===")
    
    # 이미지와 스칼라 연산
    image = np.array([[100, 150], [200, 250]], dtype=np.uint8)
    brightened = np.clip(image + 50, 0, 255)  # 밝기 증가
    
    print(f"원본:\n{image}")
    print(f"밝기 +50:\n{brightened}")
    
    # 이미지와 1차원 배열 연산 (각 채널에 다른 값 적용)
    color_image = np.random.randint(0, 256, (2, 2, 3), dtype=np.uint8)
    channel_weights = np.array([0.8, 1.0, 1.2])  # R, G, B 가중치
    
    weighted_image = np.clip(color_image * channel_weights, 0, 255).astype(np.uint8)
    
    print(f"\n컬러 이미지:\n{color_image}")
    print(f"채널별 가중치 적용:\n{weighted_image}")

# OpenCV import (실제 사용시)
try:
    import cv2
    opencv_available = True
except ImportError:
    opencv_available = False
    # OpenCV 없이도 실행 가능하도록 더미 함수 정의
    class cv2:
        @staticmethod
        def add(a, b):
            return np.clip(a.astype(np.int16) + b, 0, 255).astype(np.uint8)
        
        @staticmethod
        def subtract(a, b):
            return np.clip(a.astype(np.int16) - b, 0, 255).astype(np.uint8)

if __name__ == "__main__":
    print("NumPy 기초 - 이미지는 숫자 배열입니다!")
    print("=" * 50)
    
    if not opencv_available:
        print("주의: OpenCV가 설치되지 않았습니다. 일부 기능은 시뮬레이션됩니다.")
        print("설치: pip install opencv-python")
        print()
    
    array_creation_demo()
    array_properties_demo()
    array_indexing_demo()
    array_operations_demo()
    statistical_operations_demo()
    reshaping_demo()
    broadcasting_demo()
    
    print("\n" + "=" * 50)
    print("NumPy 기초 완료! 다음은 OpenCV를 학습해보세요.")
    print("팁: 이미지는 [높이, 너비, 채널] 순서의 NumPy 배열입니다.")
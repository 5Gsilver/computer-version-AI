"""
05. 임계값 처리 (Thresholding)
이미지를 이진화하여 객체와 배경을 분리하는 기법들을 학습합니다.
"""

import numpy as np

try:
    import cv2
    opencv_available = True
except ImportError:
    opencv_available = False
    print("OpenCV가 설치되지 않았습니다.")

def create_test_image():
    """테스트용 이미지 생성 (다양한 밝기 영역 포함)"""
    img = np.zeros((200, 200), dtype=np.uint8)
    
    if opencv_available:
        # 여러 밝기의 사각형들
        cv2.rectangle(img, (20, 20), (80, 80), 60, -1)    # 어두운 회색
        cv2.rectangle(img, (100, 20), (160, 80), 120, -1)  # 중간 회색
        cv2.rectangle(img, (20, 100), (80, 160), 180, -1)  # 밝은 회색
        cv2.rectangle(img, (100, 100), (160, 160), 240, -1) # 매우 밝은 회색
        
        # 원형 객체들
        cv2.circle(img, (50, 180), 15, 200, -1)
        cv2.circle(img, (130, 180), 15, 100, -1)
    
    return img

def simple_thresholding_demo():
    """간단한 임계값 처리"""
    print("=== 간단한 임계값 처리 ===")
    
    if not opencv_available:
        print("OpenCV 필요")
        return
    
    img = create_test_image()
    
    # 다양한 임계값으로 테스트
    thresholds = [64, 128, 192]
    
    for thresh in thresholds:
        _, binary = cv2.threshold(img, thresh, 255, cv2.THRESH_BINARY)
        
        # 전경 픽셀 비율 계산
        foreground_ratio = np.sum(binary == 255) / binary.size * 100
        
        print(f"임계값 {thresh}: 전경 비율 {foreground_ratio:.1f}%")

def adaptive_thresholding_demo():
    """적응적 임계값 처리"""
    print("\n=== 적응적 임계값 처리 ===")
    
    if not opencv_available:
        print("OpenCV 필요")
        return
    
    # 조명이 불균등한 이미지 생성
    img = np.zeros((200, 200), dtype=np.uint8)
    
    # 그라디언트 배경 (조명 불균등 시뮬레이션)
    for i in range(200):
        for j in range(200):
            img[i, j] = int(50 + 100 * i / 200)
    
    # 객체 추가
    cv2.rectangle(img, (50, 50), (100, 100), 255, -1)
    cv2.rectangle(img, (120, 120), (170, 170), 255, -1)
    
    # 일반 임계값 vs 적응적 임계값
    _, simple_thresh = cv2.threshold(img, 128, 255, cv2.THRESH_BINARY)
    
    adaptive_mean = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_MEAN_C, 
                                         cv2.THRESH_BINARY, 11, 2)
    adaptive_gaussian = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                            cv2.THRESH_BINARY, 11, 2)
    
    # 결과 비교
    methods = [
        ("일반 임계값", simple_thresh),
        ("적응적 (평균)", adaptive_mean),
        ("적응적 (가우시안)", adaptive_gaussian)
    ]
    
    for name, result in methods:
        foreground_pixels = np.sum(result == 255)
        print(f"{name}: 전경 픽셀 {foreground_pixels}개")

def otsu_thresholding_demo():
    """Otsu 자동 임계값 처리"""
    print("\n=== Otsu 자동 임계값 ===")
    
    if not opencv_available:
        print("OpenCV 필요")
        return
    
    img = create_test_image()
    
    # Otsu 방법으로 자동 임계값 결정
    otsu_thresh, otsu_binary = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    print(f"Otsu 임계값: {otsu_thresh:.0f}")
    
    # 수동 임계값과 비교
    manual_thresholds = [64, 128, 192]
    
    def calculate_variance_ratio(image, binary):
        """클래스 간 분산 비율 계산 (Otsu 기준)"""
        # 전경과 배경 분리
        background = image[binary == 0]
        foreground = image[binary == 255]
        
        if len(background) == 0 or len(foreground) == 0:
            return 0
        
        # 각 클래스의 분산
        bg_var = np.var(background)
        fg_var = np.var(foreground)
        
        # 가중 평균 분산
        total_pixels = len(background) + len(foreground)
        weighted_var = (len(background) * bg_var + len(foreground) * fg_var) / total_pixels
        
        return weighted_var
    
    otsu_variance = calculate_variance_ratio(img, otsu_binary)
    
    for thresh in manual_thresholds:
        _, manual_binary = cv2.threshold(img, thresh, 255, cv2.THRESH_BINARY)
        manual_variance = calculate_variance_ratio(img, manual_binary)
        
        print(f"임계값 {thresh}: 분산 {manual_variance:.2f}")
    
    print(f"Otsu 방법: 분산 {otsu_variance:.2f} (최적)")

def multi_level_thresholding():
    """다단계 임계값 처리"""
    print("\n=== 다단계 임계값 처리 ===")
    
    img = create_test_image()
    
    # 3단계 임계값
    thresholds = [85, 170]
    result = np.zeros_like(img)
    
    # 각 레벨에 다른 값 할당
    result[img <= thresholds[0]] = 64        # 어두운 영역
    result[(img > thresholds[0]) & (img <= thresholds[1])] = 128  # 중간 영역
    result[img > thresholds[1]] = 192        # 밝은 영역
    
    # 각 레벨의 픽셀 수 계산
    levels = [64, 128, 192]
    for i, level in enumerate(levels):
        pixel_count = np.sum(result == level)
        percentage = pixel_count / result.size * 100
        print(f"레벨 {i+1} (값 {level}): {pixel_count}픽셀 ({percentage:.1f}%)")

def triangle_thresholding():
    """Triangle 임계값 방법 (수동 구현)"""
    print("\n=== Triangle 임계값 방법 ===")
    
    img = create_test_image()
    
    # 히스토그램 계산
    hist = cv2.calcHist([img], [0], None, [256], [0, 256]).flatten()
    
    # Triangle 방법으로 임계값 찾기
    # 1. 히스토그램의 최댓값 찾기
    max_idx = np.argmax(hist)
    
    # 2. 가장 먼 점 찾기 (0 또는 255)
    if max_idx < 128:
        far_idx = 255
    else:
        far_idx = 0
    
    # 3. 최대 거리 점 찾기
    max_distance = 0
    triangle_thresh = 0
    
    for i in range(256):
        if hist[i] > 0:
            # 직선에서 점까지의 거리 계산
            distance = abs((far_idx - max_idx) * hist[i] - (i - max_idx) * (hist[far_idx] - hist[max_idx]))
            distance = distance / np.sqrt((far_idx - max_idx)**2 + (hist[far_idx] - hist[max_idx])**2)
            
            if distance > max_distance:
                max_distance = distance
                triangle_thresh = i
    
    print(f"Triangle 임계값: {triangle_thresh}")
    
    # 결과 적용
    if opencv_available:
        _, triangle_binary = cv2.threshold(img, triangle_thresh, 255, cv2.THRESH_BINARY)
        foreground_ratio = np.sum(triangle_binary == 255) / triangle_binary.size * 100
        print(f"전경 비율: {foreground_ratio:.1f}%")

def morphological_cleaning():
    """형태학적 연산을 이용한 이진화 결과 정리"""
    print("\n=== 형태학적 정리 ===")
    
    if not opencv_available:
        print("OpenCV 필요")
        return
    
    img = create_test_image()
    
    # 노이즈 추가
    noise = np.random.randint(0, 50, img.shape, dtype=np.uint8)
    noisy_img = cv2.add(img, noise)
    
    # 임계값 처리
    _, binary = cv2.threshold(noisy_img, 128, 255, cv2.THRESH_BINARY)
    
    # 형태학적 연산으로 정리
    kernel = np.ones((3, 3), np.uint8)
    
    # Opening (침식 + 팽창): 작은 노이즈 제거
    opening = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
    
    # Closing (팽창 + 침식): 작은 구멍 메우기
    closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel)
    
    # 결과 비교
    def count_objects(binary_img):
        """객체 개수 세기 (연결 요소 분석)"""
        num_labels, _ = cv2.connectedComponents(binary_img)
        return num_labels - 1  # 배경 제외
    
    original_objects = count_objects(binary)
    cleaned_objects = count_objects(closing)
    
    print(f"원본 객체 수: {original_objects}")
    print(f"정리 후 객체 수: {cleaned_objects}")

if __name__ == "__main__":
    print("임계값 처리 - 이미지 이진화 기법")
    print("=" * 50)
    
    if not opencv_available:
        print("이 예제를 실행하려면 OpenCV가 필요합니다.")
        print("설치: pip install opencv-python")
        print()
    else:
        simple_thresholding_demo()
        adaptive_thresholding_demo()
        otsu_thresholding_demo()
        morphological_cleaning()
    
    # OpenCV 없이도 실행 가능한 부분
    multi_level_thresholding()
    triangle_thresholding()
    
    print("\n" + "=" * 50)
    print("임계값 처리 학습 완료!")
    print("핵심 개념:")
    print("- 고정 임계값: 간단하지만 조명 변화에 민감")
    print("- 적응적 임계값: 지역적 조건을 고려")
    print("- Otsu 방법: 자동으로 최적 임계값 찾기")
    print("- 형태학적 정리: 이진화 결과 개선")
    print("- 다단계 임계값: 여러 레벨로 분할")
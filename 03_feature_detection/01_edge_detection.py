"""
01. 에지 검출 (Edge Detection)
에지는 이미지에서 밝기가 급격하게 변하는 부분으로, 객체의 경계를 나타냅니다.
"""

import numpy as np

try:
    import cv2
    opencv_available = True
except ImportError:
    opencv_available = False
    print("OpenCV가 설치되지 않았습니다.")

def create_test_image():
    """에지 검출 테스트용 이미지 생성"""
    img = np.zeros((200, 200), dtype=np.uint8)
    
    if opencv_available:
        # 다양한 형태의 객체들
        cv2.rectangle(img, (50, 50), (100, 100), 255, -1)  # 사각형
        cv2.circle(img, (150, 75), 25, 128, -1)             # 원
        cv2.line(img, (20, 150), (180, 150), 200, 3)        # 선
        
        # 삼각형
        pts = np.array([[100, 120], [80, 160], [120, 160]], np.int32)
        cv2.fillPoly(img, [pts], 180)
    
    return img

def sobel_edge_demo():
    """Sobel 에지 검출"""
    print("=== Sobel 에지 검출 ===")
    
    if not opencv_available:
        print("OpenCV 필요")
        return
    
    img = create_test_image()
    
    # Sobel 연산자 적용
    sobel_x = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)  # 수직 에지
    sobel_y = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)  # 수평 에지
    
    # 절댓값 계산
    sobel_x = np.abs(sobel_x)
    sobel_y = np.abs(sobel_y)
    
    # 그래디언트 크기와 방향 계산
    gradient_magnitude = np.sqrt(sobel_x**2 + sobel_y**2)
    gradient_direction = np.arctan2(sobel_y, sobel_x)
    
    # 결과 분석
    edge_pixels = np.sum(gradient_magnitude > 50)
    total_pixels = img.size
    edge_percentage = edge_pixels / total_pixels * 100
    
    print(f"에지 픽셀 수: {edge_pixels}")
    print(f"전체 대비 에지 비율: {edge_percentage:.2f}%")
    print(f"평균 그래디언트 크기: {np.mean(gradient_magnitude):.2f}")
    
    return sobel_x, sobel_y, gradient_magnitude

def laplacian_edge_demo():
    """Laplacian 에지 검출"""
    print("\n=== Laplacian 에지 검출 ===")
    
    if not opencv_available:
        print("OpenCV 필요")
        return
    
    img = create_test_image()
    
    # 가우시안 블러 적용 (노이즈 감소)
    blurred = cv2.GaussianBlur(img, (3, 3), 0)
    
    # Laplacian 연산자 적용
    laplacian = cv2.Laplacian(blurred, cv2.CV_64F)
    laplacian = np.abs(laplacian)
    
    # 임계값 적용
    _, binary_edges = cv2.threshold(laplacian.astype(np.uint8), 20, 255, cv2.THRESH_BINARY)
    
    # 에지 강도 분석
    edge_strength = np.var(laplacian)
    edge_pixels = np.sum(binary_edges > 0)
    
    print(f"Laplacian 분산 (에지 강도): {edge_strength:.2f}")
    print(f"이진 에지 픽셀 수: {edge_pixels}")
    
    return laplacian, binary_edges

def canny_edge_demo():
    """Canny 에지 검출 (가장 널리 사용되는 방법)"""
    print("\n=== Canny 에지 검출 ===")
    
    if not opencv_available:
        print("OpenCV 필요")
        return
    
    img = create_test_image()
    
    # 다양한 임계값으로 Canny 적용
    thresholds = [(50, 150), (30, 100), (100, 200)]
    
    results = []
    for low, high in thresholds:
        edges = cv2.Canny(img, low, high)
        edge_pixels = np.sum(edges > 0)
        results.append((edges, edge_pixels))
        
        print(f"임계값 ({low}, {high}): 에지 픽셀 {edge_pixels}개")
    
    # 최적 임계값 찾기 (Otsu 기반)
    blurred = cv2.GaussianBlur(img, (5, 5), 0)
    otsu_thresh, _ = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # Otsu 임계값을 기반으로 Canny 임계값 설정
    lower = 0.5 * otsu_thresh
    upper = otsu_thresh
    
    auto_canny = cv2.Canny(img, lower, upper)
    auto_edge_pixels = np.sum(auto_canny > 0)
    
    print(f"자동 임계값 ({lower:.0f}, {upper:.0f}): 에지 픽셀 {auto_edge_pixels}개")
    
    return results, auto_canny

def compare_edge_detectors():
    """다양한 에지 검출 방법 비교"""
    print("\n=== 에지 검출 방법 비교 ===")
    
    if not opencv_available:
        print("OpenCV 필요")
        return
    
    img = create_test_image()
    
    # 각 방법 적용
    sobel_x = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)
    sobel_combined = np.sqrt(sobel_x**2 + sobel_y**2)
    
    laplacian = cv2.Laplacian(img, cv2.CV_64F)
    canny = cv2.Canny(img, 50, 150)
    
    # 성능 지표 계산
    methods = [
        ("Sobel", sobel_combined),
        ("Laplacian", np.abs(laplacian)),
        ("Canny", canny.astype(np.float64))
    ]
    
    print("방법별 성능 비교:")
    for name, edges in methods:
        # 에지 강도
        edge_strength = np.mean(edges)
        # 에지 연속성 (인접 픽셀과의 연결성)
        edge_pixels = np.sum(edges > 50)  # 임계값 50 이상
        
        print(f"{name:12}: 평균 강도 {edge_strength:6.2f}, 에지 픽셀 {edge_pixels:4d}개")

def gradient_direction_analysis():
    """그래디언트 방향 분석"""
    print("\n=== 그래디언트 방향 분석 ===")
    
    if not opencv_available:
        print("OpenCV 필요")
        return
    
    img = create_test_image()
    
    # 그래디언트 계산
    grad_x = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)
    
    # 방향 계산 (라디안)
    direction = np.arctan2(grad_y, grad_x)
    
    # 방향을 8개 구간으로 나누기 (0°, 45°, 90°, 135°, 180°, 225°, 270°, 315°)
    direction_degrees = np.degrees(direction) % 180  # 0-180도로 정규화
    
    # 방향별 히스토그램
    bins = [0, 22.5, 67.5, 112.5, 157.5, 180]
    direction_labels = ["수평", "대각선1", "수직", "대각선2"]
    
    for i in range(len(bins)-1):
        mask = (direction_degrees >= bins[i]) & (direction_degrees < bins[i+1])
        count = np.sum(mask)
        if i < len(direction_labels):
            print(f"{direction_labels[i]} 방향 에지: {count}개")

def non_maximum_suppression_demo():
    """비최대 억제 데모 (Canny의 핵심 단계)"""
    print("\n=== 비최대 억제 ===")
    
    img = create_test_image()
    
    # 간단한 비최대 억제 구현
    def simple_non_max_suppression(magnitude, direction):
        """간단한 비최대 억제 구현"""
        M, N = magnitude.shape
        suppressed = np.zeros((M, N), dtype=np.float64)
        
        # 방향을 4개 구간으로 단순화
        angle = np.degrees(direction) % 180
        
        for i in range(1, M-1):
            for j in range(1, N-1):
                try:
                    # 방향에 따른 인접 픽셀 선택
                    if (0 <= angle[i,j] < 22.5) or (157.5 <= angle[i,j] <= 180):
                        # 수평 방향
                        neighbors = [magnitude[i, j-1], magnitude[i, j+1]]
                    elif (22.5 <= angle[i,j] < 67.5):
                        # 대각선 방향 (/)
                        neighbors = [magnitude[i-1, j+1], magnitude[i+1, j-1]]
                    elif (67.5 <= angle[i,j] < 112.5):
                        # 수직 방향
                        neighbors = [magnitude[i-1, j], magnitude[i+1, j]]
                    else:
                        # 대각선 방향 (\)
                        neighbors = [magnitude[i-1, j-1], magnitude[i+1, j+1]]
                    
                    # 현재 픽셀이 인접 픽셀들보다 크면 유지
                    if magnitude[i,j] >= max(neighbors):
                        suppressed[i,j] = magnitude[i,j]
                except:
                    pass
        
        return suppressed
    
    if opencv_available:
        # 그래디언트 계산
        grad_x = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)
        
        magnitude = np.sqrt(grad_x**2 + grad_y**2)
        direction = np.arctan2(grad_y, grad_x)
        
        # 비최대 억제 적용
        suppressed = simple_non_max_suppression(magnitude, direction)
        
        # 결과 비교
        original_edges = np.sum(magnitude > 50)
        suppressed_edges = np.sum(suppressed > 50)
        
        print(f"억제 전 에지 픽셀: {original_edges}개")
        print(f"억제 후 에지 픽셀: {suppressed_edges}개")
        print(f"억제율: {(1 - suppressed_edges/original_edges)*100:.1f}%")

if __name__ == "__main__":
    print("에지 검출 - 객체 경계 찾기")
    print("=" * 50)
    
    if not opencv_available:
        print("이 예제를 실행하려면 OpenCV가 필요합니다.")
        print("설치: pip install opencv-python")
        print()
    else:
        sobel_edge_demo()
        laplacian_edge_demo()
        canny_edge_demo()
        compare_edge_detectors()
        gradient_direction_analysis()
        non_maximum_suppression_demo()
    
    print("\n" + "=" * 50)
    print("에지 검출 학습 완료!")
    print("핵심 개념:")
    print("- Sobel: 방향성 에지 검출, 노이즈에 강함")
    print("- Laplacian: 2차 미분, 에지의 정확한 위치 검출")
    print("- Canny: 다단계 처리로 최적의 에지 검출")
    print("- 비최대 억제: 에지를 얇고 연속적으로 만들기")
    print("- 그래디언트 방향: 에지의 방향성 정보 활용")
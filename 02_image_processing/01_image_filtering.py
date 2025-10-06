"""
01. 이미지 필터링
이미지 필터링은 노이즈 제거, 에지 강조, 블러 효과 등을 위해 사용됩니다.
"""

import numpy as np

try:
    import cv2
    opencv_available = True
except ImportError:
    opencv_available = False
    print("OpenCV가 설치되지 않았습니다. pip install opencv-python")

def create_test_image():
    """테스트용 이미지 생성 (노이즈 포함)"""
    # 기본 이미지 생성
    img = np.zeros((200, 200), dtype=np.uint8)
    
    # 사각형들 추가
    cv2.rectangle(img, (50, 50), (100, 100), 255, -1)
    cv2.rectangle(img, (120, 120), (170, 170), 128, -1)
    cv2.circle(img, (150, 75), 25, 200, -1)
    
    # 노이즈 추가
    noise = np.random.normal(0, 25, img.shape).astype(np.int16)
    noisy_img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    
    return img, noisy_img

def gaussian_blur_demo():
    """가우시안 블러 필터링"""
    print("=== 가우시안 블러 ===")
    
    if not opencv_available:
        print("OpenCV 필요")
        return
    
    original, noisy = create_test_image()
    
    # 다양한 커널 크기로 블러 적용
    blur_sizes = [5, 15, 31]
    
    for size in blur_sizes:
        blurred = cv2.GaussianBlur(noisy, (size, size), 0)
        
        # 노이즈 감소 효과 측정
        original_std = np.std(original.astype(np.float32))
        noisy_std = np.std(noisy.astype(np.float32))
        blurred_std = np.std(blurred.astype(np.float32))
        
        print(f"커널 크기 {size}x{size}:")
        print(f"  원본 표준편차: {original_std:.2f}")
        print(f"  노이즈 표준편차: {noisy_std:.2f}")
        print(f"  블러 후 표준편차: {blurred_std:.2f}")
        print(f"  노이즈 감소율: {((noisy_std - blurred_std) / noisy_std * 100):.1f}%")

def median_filter_demo():
    """미디안 필터 (잡점 노이즈 제거에 효과적)"""
    print("\n=== 미디안 필터 ===")
    
    if not opencv_available:
        print("OpenCV 필요")
        return
    
    # 잡점 노이즈가 있는 이미지 생성
    img = np.full((100, 100), 128, dtype=np.uint8)
    
    # 잡점 노이즈 추가 (salt and pepper noise)
    noise_coords = np.random.randint(0, 100, (200, 2))
    for coord in noise_coords:
        if np.random.random() > 0.5:
            img[coord[0], coord[1]] = 255  # 흰 점
        else:
            img[coord[0], coord[1]] = 0    # 검은 점
    
    # 미디안 필터 적용
    median_filtered = cv2.medianBlur(img, 5)
    
    # 가우시안 블러와 비교
    gaussian_filtered = cv2.GaussianBlur(img, (5, 5), 0)
    
    # 효과 비교
    original_noise_count = np.sum((img == 0) | (img == 255))
    median_noise_count = np.sum((median_filtered == 0) | (median_filtered == 255))
    gaussian_noise_count = np.sum((gaussian_filtered == 0) | (gaussian_filtered == 255))
    
    print(f"원본 잡점 수: {original_noise_count}")
    print(f"미디안 필터 후: {median_noise_count}")
    print(f"가우시안 필터 후: {gaussian_noise_count}")
    print(f"미디안 필터 제거율: {((original_noise_count - median_noise_count) / original_noise_count * 100):.1f}%")

def bilateral_filter_demo():
    """양방향 필터 (에지 보존하며 노이즈 제거)"""
    print("\n=== 양방향 필터 ===")
    
    if not opencv_available:
        print("OpenCV 필요")
        return
    
    original, noisy = create_test_image()
    
    # 양방향 필터 적용
    bilateral = cv2.bilateralFilter(noisy, 9, 75, 75)
    
    # 가우시안 블러와 비교
    gaussian = cv2.GaussianBlur(noisy, (9, 9), 0)
    
    # 에지 보존 효과 측정 (Laplacian 분산)
    def edge_strength(image):
        laplacian = cv2.Laplacian(image, cv2.CV_64F)
        return np.var(laplacian)
    
    original_edges = edge_strength(original)
    bilateral_edges = edge_strength(bilateral)
    gaussian_edges = edge_strength(gaussian)
    
    print(f"원본 에지 강도: {original_edges:.2f}")
    print(f"양방향 필터 에지 강도: {bilateral_edges:.2f}")
    print(f"가우시안 필터 에지 강도: {gaussian_edges:.2f}")
    print(f"양방향 필터 에지 보존율: {(bilateral_edges / original_edges * 100):.1f}%")
    print(f"가우시안 필터 에지 보존율: {(gaussian_edges / original_edges * 100):.1f}%")

def sharpening_demo():
    """이미지 샤프닝 (선명하게 만들기)"""
    print("\n=== 이미지 샤프닝 ===")
    
    if not opencv_available:
        print("OpenCV 필요")
        return
    
    # 블러된 이미지 생성
    original, _ = create_test_image()
    blurred = cv2.GaussianBlur(original, (15, 15), 0)
    
    # 언샵 마스킹을 이용한 샤프닝
    # Sharpened = Original + k * (Original - Blurred)
    k = 1.5  # 샤프닝 강도
    sharpened = cv2.addWeighted(original, 1 + k, blurred, -k, 0)
    
    # 라플라시안 커널을 이용한 샤프닝
    laplacian_kernel = np.array([[0, -1, 0],
                                [-1, 5, -1],
                                [0, -1, 0]])
    sharpened_conv = cv2.filter2D(blurred, -1, laplacian_kernel)
    
    # 선명도 측정 (그래디언트 크기)
    def sharpness_measure(image):
        grad_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)
        gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)
        return np.mean(gradient_magnitude)
    
    original_sharpness = sharpness_measure(original)
    blurred_sharpness = sharpness_measure(blurred)
    unsharp_sharpness = sharpness_measure(sharpened)
    conv_sharpness = sharpness_measure(sharpened_conv)
    
    print(f"원본 선명도: {original_sharpness:.2f}")
    print(f"블러 후 선명도: {blurred_sharpness:.2f}")
    print(f"언샵 마스킹 후: {unsharp_sharpness:.2f}")
    print(f"컨볼루션 후: {conv_sharpness:.2f}")

def custom_filter_demo():
    """사용자 정의 필터 (컨볼루션)"""
    print("\n=== 사용자 정의 필터 ===")
    
    if not opencv_available:
        print("OpenCV 필요")
        return
    
    original, _ = create_test_image()
    
    # 다양한 커널 정의
    kernels = {
        "Identity": np.array([[0, 0, 0],
                             [0, 1, 0],
                             [0, 0, 0]]),
        
        "Edge Detection": np.array([[-1, -1, -1],
                                   [-1,  8, -1],
                                   [-1, -1, -1]]),
        
        "Emboss": np.array([[-2, -1, 0],
                           [-1,  1, 1],
                           [ 0,  1, 2]]),
        
        "Sharpen": np.array([[ 0, -1,  0],
                            [-1,  5, -1],
                            [ 0, -1,  0]])
    }
    
    print("필터 효과:")
    for name, kernel in kernels.items():
        filtered = cv2.filter2D(original, -1, kernel)
        
        # 필터 효과 측정
        diff = np.mean(np.abs(filtered.astype(np.float32) - original.astype(np.float32)))
        print(f"{name}: 평균 변화량 {diff:.2f}")

def frequency_domain_filtering():
    """주파수 도메인 필터링 (푸리에 변환 이용)"""
    print("\n=== 주파수 도메인 필터링 ===")
    
    original, noisy = create_test_image()
    
    # 푸리에 변환
    f_transform = np.fft.fft2(noisy)
    f_shifted = np.fft.fftshift(f_transform)
    
    # 로우패스 필터 생성 (중심에서 거리에 따른 마스크)
    rows, cols = noisy.shape
    crow, ccol = rows // 2, cols // 2
    
    # 원형 로우패스 필터
    mask = np.zeros((rows, cols), dtype=np.uint8)
    radius = 30
    y, x = np.ogrid[:rows, :cols]
    mask_area = (x - ccol)**2 + (y - crow)**2 <= radius**2
    mask[mask_area] = 1
    
    # 필터 적용
    f_filtered = f_shifted * mask
    
    # 역변환
    f_ishifted = np.fft.ifftshift(f_filtered)
    filtered_image = np.fft.ifft2(f_ishifted)
    filtered_image = np.abs(filtered_image).astype(np.uint8)
    
    # 효과 비교
    original_std = np.std(original.astype(np.float32))
    noisy_std = np.std(noisy.astype(np.float32))
    filtered_std = np.std(filtered_image.astype(np.float32))
    
    print(f"원본 표준편차: {original_std:.2f}")
    print(f"노이즈 표준편차: {noisy_std:.2f}")
    print(f"주파수 필터 후: {filtered_std:.2f}")
    print(f"주파수 도메인 노이즈 감소율: {((noisy_std - filtered_std) / noisy_std * 100):.1f}%")

if __name__ == "__main__":
    print("이미지 필터링 - 노이즈 제거와 이미지 개선")
    print("=" * 50)
    
    if not opencv_available:
        print("이 예제를 실행하려면 OpenCV가 필요합니다.")
        print("설치: pip install opencv-python")
        print()
    else:
        gaussian_blur_demo()
        median_filter_demo()
        bilateral_filter_demo()
        sharpening_demo()
        custom_filter_demo()
    
    # NumPy만으로도 실행 가능한 부분
    frequency_domain_filtering()
    
    print("\n" + "=" * 50)
    print("필터링 학습 완료!")
    print("핵심 개념:")
    print("- 가우시안 블러: 전체적인 노이즈 감소")
    print("- 미디안 필터: 잡점 노이즈 제거")
    print("- 양방향 필터: 에지 보존하며 노이즈 제거")
    print("- 샤프닝: 이미지 선명도 향상")
    print("- 주파수 도메인: 고급 필터링 기법")
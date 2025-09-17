"""
유틸리티 함수들
컴퓨터 비전 프로젝트에서 자주 사용되는 함수들을 모아놓았습니다.
"""

import numpy as np

try:
    import cv2
    opencv_available = True
except ImportError:
    opencv_available = False

def create_sample_images():
    """다양한 샘플 이미지 생성"""
    
    # 1. 그라디언트 이미지
    def create_gradient(width=300, height=200, direction='horizontal'):
        if direction == 'horizontal':
            gradient = np.linspace(0, 255, width, dtype=np.uint8)
            gradient = np.tile(gradient, (height, 1))
        else:  # vertical
            gradient = np.linspace(0, 255, height, dtype=np.uint8)
            gradient = np.tile(gradient.reshape(-1, 1), (1, width))
        return gradient
    
    # 2. 체스보드 패턴
    def create_checkerboard(size=200, squares=8):
        square_size = size // squares
        board = np.zeros((size, size), dtype=np.uint8)
        
        for i in range(squares):
            for j in range(squares):
                if (i + j) % 2 == 0:
                    y1, y2 = i * square_size, (i + 1) * square_size
                    x1, x2 = j * square_size, (j + 1) * square_size
                    board[y1:y2, x1:x2] = 255
        return board
    
    # 3. 원형 패턴
    def create_circles(width=300, height=200):
        img = np.zeros((height, width), dtype=np.uint8)
        if opencv_available:
            cv2.circle(img, (width//4, height//2), 30, 255, -1)
            cv2.circle(img, (width//2, height//2), 40, 128, -1)
            cv2.circle(img, (3*width//4, height//2), 35, 200, -1)
        return img
    
    # 4. 텍스처 패턴
    def create_texture(width=300, height=200):
        # 사인파 기반 텍스처
        x = np.arange(width)
        y = np.arange(height)
        X, Y = np.meshgrid(x, y)
        
        texture = 128 + 50 * np.sin(X * 0.1) * np.cos(Y * 0.1)
        return np.clip(texture, 0, 255).astype(np.uint8)
    
    return {
        'gradient_h': create_gradient(direction='horizontal'),
        'gradient_v': create_gradient(direction='vertical'),
        'checkerboard': create_checkerboard(),
        'circles': create_circles(),
        'texture': create_texture()
    }

def add_noise(image, noise_type='gaussian', intensity=25):
    """이미지에 노이즈 추가"""
    
    if noise_type == 'gaussian':
        # 가우시안 노이즈
        noise = np.random.normal(0, intensity, image.shape)
        noisy = image.astype(np.float32) + noise
        
    elif noise_type == 'salt_pepper':
        # 잡점 노이즈
        noisy = image.copy().astype(np.float32)
        num_salt = int(intensity * image.size / 1000)
        
        # Salt noise (흰 점)
        coords = [np.random.randint(0, i-1, num_salt) for i in image.shape]
        noisy[tuple(coords)] = 255
        
        # Pepper noise (검은 점)
        coords = [np.random.randint(0, i-1, num_salt) for i in image.shape]
        noisy[tuple(coords)] = 0
        
    elif noise_type == 'uniform':
        # 균등 노이즈
        noise = np.random.uniform(-intensity, intensity, image.shape)
        noisy = image.astype(np.float32) + noise
    
    return np.clip(noisy, 0, 255).astype(np.uint8)

def calculate_metrics(original, processed):
    """이미지 품질 메트릭 계산"""
    
    # MSE (Mean Squared Error)
    mse = np.mean((original.astype(np.float32) - processed.astype(np.float32)) ** 2)
    
    # PSNR (Peak Signal-to-Noise Ratio)
    if mse == 0:
        psnr = float('inf')
    else:
        psnr = 20 * np.log10(255.0 / np.sqrt(mse))
    
    # SSIM 간단 버전 (구조적 유사성)
    def ssim_simple(img1, img2):
        mu1 = np.mean(img1)
        mu2 = np.mean(img2)
        sigma1 = np.var(img1)
        sigma2 = np.var(img2)
        sigma12 = np.mean((img1 - mu1) * (img2 - mu2))
        
        c1 = (0.01 * 255) ** 2
        c2 = (0.03 * 255) ** 2
        
        ssim = ((2 * mu1 * mu2 + c1) * (2 * sigma12 + c2)) / \
               ((mu1**2 + mu2**2 + c1) * (sigma1 + sigma2 + c2))
        return ssim
    
    ssim = ssim_simple(original.astype(np.float32), processed.astype(np.float32))
    
    return {
        'MSE': mse,
        'PSNR': psnr,
        'SSIM': ssim
    }

def resize_with_aspect_ratio(image, target_width=None, target_height=None):
    """종횡비를 유지하며 이미지 크기 조정"""
    
    if not opencv_available:
        print("OpenCV가 필요합니다")
        return image
    
    height, width = image.shape[:2]
    
    if target_width is None and target_height is None:
        return image
    
    if target_width is None:
        # 높이 기준으로 조정
        ratio = target_height / height
        target_width = int(width * ratio)
    elif target_height is None:
        # 너비 기준으로 조정
        ratio = target_width / width
        target_height = int(height * ratio)
    else:
        # 둘 다 지정된 경우, 더 작은 비율 사용 (이미지가 잘리지 않도록)
        ratio = min(target_width / width, target_height / height)
        target_width = int(width * ratio)
        target_height = int(height * ratio)
    
    return cv2.resize(image, (target_width, target_height), interpolation=cv2.INTER_AREA)

def create_region_mask(image_shape, regions):
    """특정 영역들의 마스크 생성"""
    
    mask = np.zeros(image_shape[:2], dtype=np.uint8)
    
    for region in regions:
        if region['type'] == 'rectangle':
            x1, y1, x2, y2 = region['coords']
            mask[y1:y2, x1:x2] = 255
            
        elif region['type'] == 'circle' and opencv_available:
            center = region['center']
            radius = region['radius']
            cv2.circle(mask, center, radius, 255, -1)
            
        elif region['type'] == 'polygon' and opencv_available:
            points = np.array(region['points'], dtype=np.int32)
            cv2.fillPoly(mask, [points], 255)
    
    return mask

def histogram_stats(image):
    """히스토그램 통계 계산"""
    
    if len(image.shape) == 3:
        # 컬러 이미지인 경우 그레이스케일로 변환
        if opencv_available:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            # 간단한 그레이스케일 변환
            gray = np.dot(image[...,:3], [0.299, 0.587, 0.114]).astype(np.uint8)
    else:
        gray = image
    
    # 히스토그램 계산
    hist = np.histogram(gray, bins=256, range=(0, 256))[0]
    
    # 통계 계산
    total_pixels = gray.size
    mean_intensity = np.mean(gray)
    std_intensity = np.std(gray)
    
    # 엔트로피 (정보량)
    hist_norm = hist / total_pixels
    entropy = -np.sum(hist_norm * np.log2(hist_norm + 1e-7))
    
    return {
        'histogram': hist,
        'mean': mean_intensity,
        'std': std_intensity,
        'entropy': entropy,
        'min': np.min(gray),
        'max': np.max(gray)
    }

def apply_gamma_correction(image, gamma=1.0):
    """감마 보정 적용"""
    
    # 룩업 테이블 생성
    inv_gamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in range(256)]).astype(np.uint8)
    
    # 룩업 테이블 적용
    if opencv_available:
        return cv2.LUT(image, table)
    else:
        # OpenCV 없이 구현
        return table[image]

def auto_contrast(image, clip_percent=1):
    """자동 대비 조정"""
    
    if len(image.shape) == 3:
        # 컬러 이미지인 경우 각 채널별로 처리
        result = np.zeros_like(image)
        for i in range(image.shape[2]):
            result[:,:,i] = auto_contrast(image[:,:,i], clip_percent)
        return result
    
    # 히스토그램 계산
    hist = np.histogram(image, bins=256, range=(0, 256))[0]
    
    # 클리핑할 픽셀 수 계산
    total_pixels = image.size
    clip_pixels = int(total_pixels * clip_percent / 100)
    
    # 하위/상위 클리핑 지점 찾기
    cumsum = np.cumsum(hist)
    low_val = np.where(cumsum > clip_pixels)[0][0]
    high_val = np.where(cumsum < total_pixels - clip_pixels)[0][-1]
    
    # 선형 변환 적용
    if high_val > low_val:
        result = np.clip((image.astype(np.float32) - low_val) * 255 / (high_val - low_val), 0, 255)
        return result.astype(np.uint8)
    else:
        return image

def create_visualization_grid(images, titles=None, grid_size=None):
    """여러 이미지를 격자로 배열하여 시각화용 이미지 생성"""
    
    if not images:
        return None
    
    num_images = len(images)
    
    if grid_size is None:
        # 자동으로 격자 크기 결정
        cols = int(np.ceil(np.sqrt(num_images)))
        rows = int(np.ceil(num_images / cols))
    else:
        rows, cols = grid_size
    
    # 첫 번째 이미지 크기 기준
    h, w = images[0].shape[:2]
    
    # 결과 이미지 초기화
    if len(images[0].shape) == 3:
        result = np.zeros((rows * h, cols * w, images[0].shape[2]), dtype=np.uint8)
    else:
        result = np.zeros((rows * h, cols * w), dtype=np.uint8)
    
    # 이미지 배치
    for i, img in enumerate(images):
        if i >= rows * cols:
            break
            
        row = i // cols
        col = i % cols
        
        y1, y2 = row * h, (row + 1) * h
        x1, x2 = col * w, (col + 1) * w
        
        # 크기 조정 (필요시)
        if img.shape[:2] != (h, w) and opencv_available:
            img = cv2.resize(img, (w, h))
        
        result[y1:y2, x1:x2] = img
    
    return result

if __name__ == "__main__":
    print("컴퓨터 비전 유틸리티 함수 테스트")
    print("=" * 50)
    
    # 샘플 이미지 생성
    samples = create_sample_images()
    print(f"생성된 샘플 이미지: {list(samples.keys())}")
    
    # 노이즈 추가 테스트
    original = samples['checkerboard']
    noisy = add_noise(original, 'gaussian', 30)
    
    # 메트릭 계산
    metrics = calculate_metrics(original, noisy)
    print(f"\n노이즈 추가 후 품질:")
    for metric, value in metrics.items():
        print(f"  {metric}: {value:.2f}")
    
    # 히스토그램 통계
    stats = histogram_stats(original)
    print(f"\n히스토그램 통계:")
    print(f"  평균: {stats['mean']:.2f}")
    print(f"  표준편차: {stats['std']:.2f}")
    print(f"  엔트로피: {stats['entropy']:.2f}")
    
    print("\n유틸리티 함수 테스트 완료!")
"""
이미지 개선기 (Image Enhancer)
노이즈 제거, 선명도 향상, 대비 조정 등을 통해 이미지 품질을 개선하는 도구
"""

import numpy as np
import sys
import os

# 상위 디렉토리의 utils 모듈 import
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
from utils.image_utils import *

try:
    import cv2
    opencv_available = True
except ImportError:
    opencv_available = False
    print("OpenCV가 설치되지 않았습니다. 일부 기능이 제한됩니다.")

class ImageEnhancer:
    """이미지 개선 클래스"""
    
    def __init__(self):
        self.enhancement_methods = {
            'denoise_gaussian': self.denoise_gaussian,
            'denoise_bilateral': self.denoise_bilateral,
            'denoise_median': self.denoise_median,
            'sharpen': self.sharpen,
            'auto_contrast': self.auto_contrast,
            'gamma_correction': self.gamma_correction,
            'histogram_equalization': self.histogram_equalization
        }
    
    def denoise_gaussian(self, image, kernel_size=5, sigma=1.0):
        """가우시안 블러를 이용한 노이즈 제거"""
        if not opencv_available:
            return image
        
        if kernel_size % 2 == 0:
            kernel_size += 1  # 홀수로 만들기
        
        return cv2.GaussianBlur(image, (kernel_size, kernel_size), sigma)
    
    def denoise_bilateral(self, image, d=9, sigma_color=75, sigma_space=75):
        """양방향 필터를 이용한 노이즈 제거 (에지 보존)"""
        if not opencv_available:
            return image
        
        return cv2.bilateralFilter(image, d, sigma_color, sigma_space)
    
    def denoise_median(self, image, kernel_size=5):
        """미디안 필터를 이용한 잡점 노이즈 제거"""
        if not opencv_available:
            return image
        
        if kernel_size % 2 == 0:
            kernel_size += 1
        
        return cv2.medianBlur(image, kernel_size)
    
    def sharpen(self, image, strength=1.0, method='unsharp'):
        """이미지 선명도 향상"""
        if not opencv_available:
            return image
        
        if method == 'unsharp':
            # 언샵 마스킹
            blurred = cv2.GaussianBlur(image, (0, 0), 1.0)
            sharpened = cv2.addWeighted(image, 1 + strength, blurred, -strength, 0)
            return sharpened
        
        elif method == 'laplacian':
            # 라플라시안 커널 사용
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                laplacian = cv2.Laplacian(gray, cv2.CV_64F)
                laplacian = np.abs(laplacian).astype(np.uint8)
                
                # 각 채널에 적용
                result = image.copy()
                for i in range(3):
                    result[:,:,i] = cv2.add(result[:,:,i], (laplacian * strength).astype(np.uint8))
                return result
            else:
                laplacian = cv2.Laplacian(image, cv2.CV_64F)
                return cv2.add(image, (np.abs(laplacian) * strength).astype(np.uint8))
    
    def auto_contrast(self, image, clip_percent=1):
        """자동 대비 조정"""
        return auto_contrast(image, clip_percent)
    
    def gamma_correction(self, image, gamma=1.0):
        """감마 보정"""
        return apply_gamma_correction(image, gamma)
    
    def histogram_equalization(self, image):
        """히스토그램 균등화"""
        if not opencv_available:
            return image
        
        if len(image.shape) == 3:
            # 컬러 이미지인 경우 LAB 색공간에서 L 채널만 균등화
            lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
            lab[:,:,0] = cv2.equalizeHist(lab[:,:,0])
            return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
        else:
            # 그레이스케일 이미지
            return cv2.equalizeHist(image)
    
    def enhance_image(self, image, enhancement_pipeline):
        """
        이미지 개선 파이프라인 실행
        
        Args:
            image: 입력 이미지
            enhancement_pipeline: 적용할 개선 방법들의 리스트
                예: [
                    ('denoise_bilateral', {'d': 9}),
                    ('sharpen', {'strength': 0.5}),
                    ('auto_contrast', {'clip_percent': 2})
                ]
        """
        result = image.copy()
        
        for method_name, params in enhancement_pipeline:
            if method_name in self.enhancement_methods:
                try:
                    result = self.enhancement_methods[method_name](result, **params)
                    print(f"✓ {method_name} 적용 완료")
                except Exception as e:
                    print(f"✗ {method_name} 적용 실패: {e}")
            else:
                print(f"✗ 알 수 없는 방법: {method_name}")
        
        return result
    
    def compare_enhancements(self, original, noisy):
        """다양한 개선 방법 비교"""
        print("=== 이미지 개선 방법 비교 ===")
        
        methods = [
            ('Original', noisy),
            ('Gaussian Blur', self.denoise_gaussian(noisy, 5)),
            ('Bilateral Filter', self.denoise_bilateral(noisy)),
            ('Median Filter', self.denoise_median(noisy, 5)),
            ('Sharpened', self.sharpen(noisy, 0.5)),
            ('Auto Contrast', self.auto_contrast(noisy)),
            ('Gamma Corrected', self.gamma_correction(noisy, 1.2)),
        ]
        
        if opencv_available:
            methods.append(('Histogram Eq.', self.histogram_equalization(noisy)))
        
        # 각 방법의 성능 평가
        for name, enhanced in methods:
            if name == 'Original':
                continue
            
            metrics = calculate_metrics(original, enhanced)
            print(f"\n{name}:")
            print(f"  PSNR: {metrics['PSNR']:.2f} dB")
            print(f"  SSIM: {metrics['SSIM']:.3f}")
        
        return methods

def demo_basic_enhancement():
    """기본 이미지 개선 데모"""
    print("=== 기본 이미지 개선 데모 ===")
    
    # 샘플 이미지 생성
    samples = create_sample_images()
    original = samples['checkerboard']
    
    # 노이즈 추가
    noisy = add_noise(original, 'gaussian', 25)
    salt_pepper = add_noise(original, 'salt_pepper', 10)
    
    # 개선기 초기화
    enhancer = ImageEnhancer()
    
    # 가우시안 노이즈 개선
    print("\n가우시안 노이즈 개선:")
    enhanced_gaussian = enhancer.denoise_bilateral(noisy)
    metrics = calculate_metrics(original, enhanced_gaussian)
    print(f"PSNR 향상: {metrics['PSNR']:.2f} dB")
    
    # 잡점 노이즈 개선
    print("\n잡점 노이즈 개선:")
    enhanced_salt_pepper = enhancer.denoise_median(salt_pepper, 3)
    metrics = calculate_metrics(original, enhanced_salt_pepper)
    print(f"PSNR 향상: {metrics['PSNR']:.2f} dB")

def demo_enhancement_pipeline():
    """개선 파이프라인 데모"""
    print("\n=== 개선 파이프라인 데모 ===")
    
    # 복합 노이즈가 있는 이미지 생성
    samples = create_sample_images()
    original = samples['circles']
    
    # 여러 종류의 노이즈 추가
    noisy = add_noise(original, 'gaussian', 20)
    noisy = add_noise(noisy, 'salt_pepper', 5)
    
    # 어두운 이미지로 만들기 (대비 문제 시뮬레이션)
    dark_noisy = (noisy * 0.6).astype(np.uint8)
    
    # 개선 파이프라인 정의
    pipeline = [
        ('denoise_median', {'kernel_size': 3}),      # 잡점 노이즈 제거
        ('denoise_bilateral', {'d': 5}),             # 가우시안 노이즈 제거 + 에지 보존
        ('auto_contrast', {'clip_percent': 2}),       # 대비 개선
        ('sharpen', {'strength': 0.3, 'method': 'unsharp'})  # 선명도 향상
    ]
    
    # 개선 실행
    enhancer = ImageEnhancer()
    enhanced = enhancer.enhance_image(dark_noisy, pipeline)
    
    # 결과 비교
    original_metrics = calculate_metrics(original, dark_noisy)
    enhanced_metrics = calculate_metrics(original, enhanced)
    
    print(f"\n개선 전 PSNR: {original_metrics['PSNR']:.2f} dB")
    print(f"개선 후 PSNR: {enhanced_metrics['PSNR']:.2f} dB")
    print(f"PSNR 향상: {enhanced_metrics['PSNR'] - original_metrics['PSNR']:.2f} dB")

def demo_adaptive_enhancement():
    """적응적 이미지 개선 데모"""
    print("\n=== 적응적 이미지 개선 데모 ===")
    
    samples = create_sample_images()
    enhancer = ImageEnhancer()
    
    for name, image in samples.items():
        print(f"\n{name} 이미지 분석:")
        
        # 이미지 특성 분석
        stats = histogram_stats(image)
        
        # 특성에 따른 개선 방법 선택
        pipeline = []
        
        # 낮은 대비 이미지 감지
        if stats['std'] < 50:
            pipeline.append(('auto_contrast', {'clip_percent': 1}))
            print("  - 낮은 대비 감지: 자동 대비 조정 적용")
        
        # 어두운 이미지 감지
        if stats['mean'] < 100:
            pipeline.append(('gamma_correction', {'gamma': 0.8}))
            print("  - 어두운 이미지 감지: 감마 보정 적용")
        
        # 밝은 이미지 감지
        if stats['mean'] > 180:
            pipeline.append(('gamma_correction', {'gamma': 1.2}))
            print("  - 밝은 이미지 감지: 감마 보정 적용")
        
        # 높은 엔트로피 (노이즈가 많을 가능성)
        if stats['entropy'] > 7:
            pipeline.append(('denoise_bilateral', {'d': 5}))
            print("  - 높은 엔트로피 감지: 노이즈 제거 적용")
        
        if pipeline:
            enhanced = enhancer.enhance_image(image, pipeline)
            enhanced_stats = histogram_stats(enhanced)
            print(f"  개선 후 평균 밝기: {stats['mean']:.1f} → {enhanced_stats['mean']:.1f}")
            print(f"  개선 후 대비: {stats['std']:.1f} → {enhanced_stats['std']:.1f}")
        else:
            print("  - 개선이 필요하지 않은 양질의 이미지")

if __name__ == "__main__":
    print("이미지 개선기 프로젝트")
    print("=" * 50)
    
    if not opencv_available:
        print("주의: OpenCV가 설치되지 않아 일부 기능이 제한됩니다.")
        print("전체 기능 사용을 위해 'pip install opencv-python'을 실행하세요.")
        print()
    
    # 기본 개선 데모
    demo_basic_enhancement()
    
    # 파이프라인 데모
    demo_enhancement_pipeline()
    
    # 적응적 개선 데모
    demo_adaptive_enhancement()
    
    print("\n" + "=" * 50)
    print("이미지 개선기 프로젝트 완료!")
    print("\n학습한 내용:")
    print("- 다양한 노이즈 제거 기법")
    print("- 이미지 선명도 향상 방법")
    print("- 자동 대비 및 밝기 조정")
    print("- 개선 파이프라인 구성")
    print("- 이미지 특성에 따른 적응적 처리")
    
    print("\n다음 단계:")
    print("- 실제 사진으로 테스트해보기")
    print("- GUI 인터페이스 추가")
    print("- 더 고급 노이즈 제거 알고리즘 연구")
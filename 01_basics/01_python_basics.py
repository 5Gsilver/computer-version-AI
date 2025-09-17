"""
01. Python 기초 복습
컴퓨터 비전에 필요한 Python 핵심 개념들을 복습합니다.
"""

# 1. 리스트와 배열 조작
def list_operations_demo():
    """리스트 기본 연산 데모"""
    print("=== 리스트 기본 연산 ===")
    
    # 리스트 생성
    numbers = [1, 2, 3, 4, 5]
    print(f"원본 리스트: {numbers}")
    
    # 리스트 슬라이싱 (이미지 ROI 추출과 유사)
    subset = numbers[1:4]
    print(f"슬라이싱 [1:4]: {subset}")
    
    # 리스트 컴프리헨션 (픽셀 처리와 유사)
    squared = [x**2 for x in numbers]
    print(f"제곱값: {squared}")
    
    # 조건부 필터링 (임계값 처리와 유사)
    filtered = [x for x in numbers if x > 3]
    print(f"3보다 큰 값: {filtered}")

# 2. 함수와 람다
def function_demo():
    """함수 정의와 사용법 데모"""
    print("\n=== 함수 사용법 ===")
    
    # 일반 함수
    def apply_threshold(value, threshold=128):
        """임계값 적용 함수 (이진화와 유사)"""
        return 255 if value > threshold else 0
    
    # 람다 함수
    normalize = lambda x: x / 255.0
    
    # 사용 예시
    pixel_value = 200
    binary_value = apply_threshold(pixel_value)
    normalized_value = normalize(pixel_value)
    
    print(f"픽셀값: {pixel_value}")
    print(f"이진화 결과: {binary_value}")
    print(f"정규화 결과: {normalized_value:.3f}")

# 3. 클래스 기초 (이미지 처리 클래스 예시)
class ImageProcessor:
    """간단한 이미지 처리기 클래스"""
    
    def __init__(self, name="기본처리기"):
        self.name = name
        self.processed_count = 0
    
    def apply_filter(self, image_data, filter_type="blur"):
        """필터 적용 (시뮬레이션)"""
        self.processed_count += 1
        print(f"{self.name}: {filter_type} 필터 적용 완료")
        print(f"처리된 이미지 수: {self.processed_count}")
        return f"filtered_{filter_type}_image"
    
    def get_stats(self):
        """처리 통계 반환"""
        return {
            "name": self.name,
            "processed": self.processed_count
        }

# 4. 예외 처리 (파일 처리 시 중요)
def safe_file_operation():
    """안전한 파일 처리 데모"""
    print("\n=== 예외 처리 ===")
    
    try:
        # 존재하지 않는 파일 처리 시뮬레이션
        filename = "nonexistent_image.jpg"
        print(f"파일 로드 시도: {filename}")
        
        # 실제로는 cv2.imread() 등을 사용
        if filename == "nonexistent_image.jpg":
            raise FileNotFoundError(f"파일을 찾을 수 없습니다: {filename}")
            
    except FileNotFoundError as e:
        print(f"오류 발생: {e}")
        print("기본 이미지를 사용합니다.")
        
    except Exception as e:
        print(f"예상치 못한 오류: {e}")
        
    finally:
        print("파일 처리 완료")

# 5. 딕셔너리와 데이터 구조
def data_structures_demo():
    """데이터 구조 활용 데모"""
    print("\n=== 데이터 구조 활용 ===")
    
    # 이미지 메타데이터 예시
    image_info = {
        "filename": "sample.jpg",
        "width": 640,
        "height": 480,
        "channels": 3,
        "format": "RGB"
    }
    
    print("이미지 정보:")
    for key, value in image_info.items():
        print(f"  {key}: {value}")
    
    # 처리 결과 저장
    results = []
    operations = ["resize", "blur", "sharpen"]
    
    for op in operations:
        result = {
            "operation": op,
            "timestamp": "2024-01-01 12:00:00",
            "success": True
        }
        results.append(result)
    
    print(f"\n처리 작업 {len(results)}개 완료")

if __name__ == "__main__":
    print("컴퓨터 비전을 위한 Python 기초")
    print("=" * 40)
    
    # 모든 데모 실행
    list_operations_demo()
    function_demo()
    
    # 클래스 사용 예시
    print("\n=== 클래스 사용 ===")
    processor = ImageProcessor("고급처리기")
    processor.apply_filter("image1.jpg", "blur")
    processor.apply_filter("image2.jpg", "sharpen")
    print(f"처리기 상태: {processor.get_stats()}")
    
    safe_file_operation()
    data_structures_demo()
    
    print("\n" + "=" * 40)
    print("Python 기초 복습 완료! 다음은 NumPy를 학습해보세요.")
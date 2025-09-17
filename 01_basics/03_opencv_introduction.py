"""
03. OpenCV 소개
OpenCV는 컴퓨터 비전을 위한 가장 인기 있는 라이브러리입니다.
"""

import numpy as np

try:
    import cv2
    opencv_available = True
    print("OpenCV 버전:", cv2.__version__)
except ImportError:
    opencv_available = False
    print("OpenCV가 설치되지 않았습니다!")
    print("설치 명령: pip install opencv-python")

def create_sample_images():
    """샘플 이미지 생성 (실제 이미지가 없을 때 사용)"""
    print("=== 샘플 이미지 생성 ===")
    
    # 그레이스케일 그라디언트 이미지
    height, width = 200, 300
    gradient = np.zeros((height, width), dtype=np.uint8)
    
    for i in range(height):
        gradient[i, :] = int(255 * i / height)
    
    print(f"그라디언트 이미지 생성: {gradient.shape}")
    
    # 컬러 체스보드 이미지
    chess = np.zeros((200, 200, 3), dtype=np.uint8)
    square_size = 25
    
    for i in range(0, 200, square_size):
        for j in range(0, 200, square_size):
            if (i // square_size + j // square_size) % 2 == 0:
                chess[i:i+square_size, j:j+square_size] = [255, 0, 0]  # 빨간색
            else:
                chess[i:i+square_size, j:j+square_size] = [0, 255, 0]  # 초록색
    
    print(f"체스보드 이미지 생성: {chess.shape}")
    
    return gradient, chess

def basic_image_operations():
    """기본 이미지 연산"""
    print("\n=== 기본 이미지 연산 ===")
    
    if not opencv_available:
        print("OpenCV가 필요합니다. 샘플 코드만 표시합니다.")
        print("""
        # 이미지 읽기
        img = cv2.imread('image.jpg')
        gray = cv2.imread('image.jpg', cv2.IMREAD_GRAYSCALE)
        
        # 이미지 정보 확인
        print(f"크기: {img.shape}")
        print(f"데이터 타입: {img.dtype}")
        
        # 이미지 저장
        cv2.imwrite('output.jpg', img)
        """)
        return
    
    # 샘플 이미지 생성
    gradient, chess = create_sample_images()
    
    # 이미지 정보 출력
    print(f"그라디언트 - 형태: {gradient.shape}, 타입: {gradient.dtype}")
    print(f"체스보드 - 형태: {chess.shape}, 타입: {chess.dtype}")
    
    # 색상 공간 변환
    chess_gray = cv2.cvtColor(chess, cv2.COLOR_BGR2GRAY)
    print(f"그레이스케일 변환 후: {chess_gray.shape}")
    
    # 이미지 크기 변경
    resized = cv2.resize(chess, (100, 100))
    print(f"크기 변경 후: {resized.shape}")
    
    # 이미지 회전
    center = (chess.shape[1]//2, chess.shape[0]//2)
    rotation_matrix = cv2.getRotationMatrix2D(center, 45, 1.0)
    rotated = cv2.warpAffine(chess, rotation_matrix, (chess.shape[1], chess.shape[0]))
    print(f"회전 후: {rotated.shape}")

def pixel_access_demo():
    """픽셀 접근 방법"""
    print("\n=== 픽셀 접근 ===")
    
    # 작은 샘플 이미지 생성
    img = np.random.randint(0, 256, (5, 5, 3), dtype=np.uint8)
    print("샘플 이미지:")
    print(img[:,:,0])  # R 채널만 표시
    
    # 개별 픽셀 접근
    pixel = img[2, 2]  # (2,2) 위치의 픽셀
    print(f"\n픽셀 (2,2) RGB: {pixel}")
    
    # 픽셀 값 변경
    img[2, 2] = [255, 0, 0]  # 빨간색으로 변경
    print(f"변경 후 픽셀 (2,2): {img[2, 2]}")
    
    # ROI (Region of Interest) 추출
    roi = img[1:4, 1:4]  # 3x3 영역 추출
    print(f"\nROI 크기: {roi.shape}")
    
    # ROI 수정
    roi[:] = [0, 255, 0]  # 초록색으로 변경
    print("ROI를 초록색으로 변경 완료")

def image_properties_demo():
    """이미지 속성 확인"""
    print("\n=== 이미지 속성 ===")
    
    # 다양한 이미지 생성
    gray_img = np.random.randint(0, 256, (100, 100), dtype=np.uint8)
    color_img = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)
    
    images = [
        ("그레이스케일", gray_img),
        ("컬러", color_img)
    ]
    
    for name, img in images:
        print(f"\n{name} 이미지:")
        print(f"  형태: {img.shape}")
        print(f"  차원: {img.ndim}")
        print(f"  크기: {img.size} 픽셀")
        print(f"  데이터 타입: {img.dtype}")
        print(f"  메모리: {img.nbytes} bytes")
        
        if img.ndim == 2:  # 그레이스케일
            print(f"  평균: {np.mean(img):.2f}")
            print(f"  최솟값: {np.min(img)}")
            print(f"  최댓값: {np.max(img)}")
        else:  # 컬러
            for i, channel in enumerate(['B', 'G', 'R']):
                channel_mean = np.mean(img[:,:,i])
                print(f"  {channel} 채널 평균: {channel_mean:.2f}")

def basic_drawing_demo():
    """기본 도형 그리기"""
    print("\n=== 기본 도형 그리기 ===")
    
    if not opencv_available:
        print("OpenCV가 필요합니다. 샘플 코드만 표시합니다.")
        print("""
        # 빈 이미지 생성
        img = np.zeros((300, 300, 3), dtype=np.uint8)
        
        # 선 그리기
        cv2.line(img, (0, 0), (299, 299), (255, 0, 0), 2)
        
        # 사각형 그리기
        cv2.rectangle(img, (50, 50), (150, 150), (0, 255, 0), 2)
        
        # 원 그리기
        cv2.circle(img, (200, 100), 50, (0, 0, 255), -1)
        
        # 텍스트 추가
        cv2.putText(img, 'OpenCV', (10, 250), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        """)
        return
    
    # 빈 캔버스 생성
    canvas = np.zeros((300, 300, 3), dtype=np.uint8)
    
    # 도형 그리기
    cv2.line(canvas, (0, 0), (299, 299), (255, 0, 0), 2)  # 대각선
    cv2.rectangle(canvas, (50, 50), (150, 150), (0, 255, 0), 2)  # 사각형
    cv2.circle(canvas, (200, 100), 30, (0, 0, 255), -1)  # 원 (채움)
    cv2.putText(canvas, 'Hello CV', (10, 280), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    print("도형 그리기 완료")
    print("실제 표시를 원한다면 cv2.imshow()와 cv2.waitKey()를 사용하세요")

def image_arithmetic_demo():
    """이미지 산술 연산"""
    print("\n=== 이미지 산술 연산 ===")
    
    # 두 개의 작은 이미지 생성
    img1 = np.full((3, 3), 100, dtype=np.uint8)
    img2 = np.full((3, 3), 50, dtype=np.uint8)
    
    print(f"이미지 1:\n{img1}")
    print(f"이미지 2:\n{img2}")
    
    if opencv_available:
        # OpenCV 산술 연산 (포화 연산)
        add_result = cv2.add(img1, img2)
        sub_result = cv2.subtract(img1, img2)
        
        print(f"\nOpenCV 덧셈:\n{add_result}")
        print(f"OpenCV 뺄셈:\n{sub_result}")
    
    # NumPy 연산
    np_add = np.clip(img1.astype(np.int16) + img2, 0, 255).astype(np.uint8)
    np_sub = np.clip(img1.astype(np.int16) - img2, 0, 255).astype(np.uint8)
    
    print(f"\nNumPy 덧셈:\n{np_add}")
    print(f"NumPy 뺄셈:\n{np_sub}")
    
    # 가중합 (이미지 블렌딩)
    alpha = 0.7
    beta = 0.3
    blended = np.clip(alpha * img1 + beta * img2, 0, 255).astype(np.uint8)
    print(f"\n가중합 (α={alpha}, β={beta}):\n{blended}")

if __name__ == "__main__":
    print("OpenCV 기초 - 컴퓨터 비전의 시작!")
    print("=" * 50)
    
    if opencv_available:
        basic_image_operations()
    else:
        print("OpenCV 설치 후 모든 기능을 사용할 수 있습니다.")
        print("설치: pip install opencv-python")
    
    pixel_access_demo()
    image_properties_demo()
    basic_drawing_demo()
    image_arithmetic_demo()
    
    print("\n" + "=" * 50)
    print("OpenCV 기초 완료!")
    print("다음 단계: 실제 이미지 파일로 실습해보세요.")
    
    if opencv_available:
        print("\n추가 팁:")
        print("- cv2.imshow()로 이미지 표시")
        print("- cv2.waitKey()로 키 입력 대기")
        print("- cv2.destroyAllWindows()로 창 닫기")
#!/usr/bin/env python3
"""
컴퓨터 비전 AI 학습 환경 설정 스크립트
"""

import subprocess
import sys
import os

def check_python_version():
    """Python 버전 확인"""
    version = sys.version_info
    print(f"Python 버전: {version.major}.{version.minor}.{version.micro}")
    
    if version.major < 3 or (version.major == 3 and version.minor < 7):
        print("❌ Python 3.7 이상이 필요합니다.")
        return False
    
    print("✅ Python 버전이 적합합니다.")
    return True

def install_requirements():
    """필수 패키지 설치"""
    print("\n📦 필수 패키지 설치 중...")
    
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "--upgrade", "pip"])
        print("✅ pip 업그레이드 완료")
        
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("✅ 필수 패키지 설치 완료")
        
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ 패키지 설치 실패: {e}")
        return False

def test_imports():
    """주요 라이브러리 import 테스트"""
    print("\n🔍 라이브러리 테스트 중...")
    
    tests = [
        ("numpy", "NumPy"),
        ("cv2", "OpenCV"),
        ("matplotlib", "Matplotlib")
    ]
    
    success = True
    for module, name in tests:
        try:
            __import__(module)
            print(f"✅ {name} 정상 작동")
        except ImportError:
            print(f"❌ {name} import 실패")
            success = False
    
    return success

def run_basic_test():
    """기본 기능 테스트"""
    print("\n🧪 기본 기능 테스트 중...")
    
    try:
        # 간단한 테스트 실행
        result = subprocess.run([sys.executable, "01_basics/01_python_basics.py"], 
                              capture_output=True, text=True, timeout=30)
        
        if result.returncode == 0:
            print("✅ 기본 테스트 통과")
            return True
        else:
            print(f"❌ 기본 테스트 실패: {result.stderr}")
            return False
    except subprocess.TimeoutExpired:
        print("❌ 테스트 시간 초과")
        return False
    except FileNotFoundError:
        print("❌ 테스트 파일을 찾을 수 없습니다.")
        return False

def print_next_steps():
    """다음 단계 안내"""
    print("\n🎉 설정 완료!")
    print("\n다음 단계:")
    print("1. 기초 학습: python 01_basics/01_python_basics.py")
    print("2. NumPy 학습: python 01_basics/02_numpy_fundamentals.py")
    print("3. OpenCV 학습: python 01_basics/03_opencv_introduction.py")
    print("4. Jupyter 노트북: jupyter notebook computer_vision_basics.ipynb")
    print("\n자세한 가이드: GETTING_STARTED.md 참조")
    print("문제 발생시: GitHub Issues에 질문 등록")

def main():
    """메인 설정 함수"""
    print("컴퓨터 비전 AI 학습 환경 설정")
    print("=" * 50)
    
    # 1. Python 버전 확인
    if not check_python_version():
        return
    
    # 2. 필수 패키지 설치
    if not install_requirements():
        print("\n패키지 설치가 실패했습니다.")
        print("수동 설치를 시도해보세요: pip install -r requirements.txt")
        return
    
    # 3. 라이브러리 테스트
    if not test_imports():
        print("\n일부 라이브러리 import에 실패했습니다.")
        print("패키지를 다시 설치해보세요.")
        return
    
    # 4. 기본 기능 테스트
    if os.path.exists("01_basics/01_python_basics.py"):
        if not run_basic_test():
            print("\n기본 테스트에 실패했지만 계속 진행할 수 있습니다.")
    
    # 5. 완료 안내
    print_next_steps()

if __name__ == "__main__":
    main()
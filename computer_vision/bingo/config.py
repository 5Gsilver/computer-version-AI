# -*- coding: utf-8 -*-
"""
색깔 빙고 게임 설정 파일
"""

# 게임 설정
GAME_TIME = 15 * 60  # 15분 (초)
DETECTION_THRESHOLD = 1000  # 색상 검출 임계값
COOLDOWN_TIME = 3.0  # 중복 검출 방지 시간 (초)

# 점수 설정
COLOR_SCORE = 10  # 색상 발견 점수
BINGO_SCORE = 50  # 빙고 완성 보너스 점수

# 웹캠 설정
CAMERA_WIDTH = 800
CAMERA_HEIGHT = 600

# 색상 정의 (HSV 범위와 BGR 색상)
COLORS = {
    '빨강': {
        'range': [[(0, 50, 50), (10, 255, 255)], [(170, 50, 50), (180, 255, 255)]],
        'bgr': (0, 0, 255),
        'found': False
    },
    '주황': {
        'range': [[(10, 50, 50), (25, 255, 255)]],
        'bgr': (0, 165, 255),
        'found': False
    },
    '노랑': {
        'range': [[(25, 50, 50), (35, 255, 255)]],
        'bgr': (0, 255, 255),
        'found': False
    },
    '연두': {
        'range': [[(35, 50, 50), (50, 255, 255)]],
        'bgr': (0, 255, 128),
        'found': False
    },
    '초록': {
        'range': [[(50, 50, 50), (80, 255, 255)]],
        'bgr': (0, 255, 0),
        'found': False
    },
    '청록': {
        'range': [[(80, 50, 50), (100, 255, 255)]],
        'bgr': (255, 255, 0),
        'found': False
    },
    '파랑': {
        'range': [[(100, 50, 50), (130, 255, 255)]],
        'bgr': (255, 0, 0),
        'found': False
    },
    '보라': {
        'range': [[(130, 50, 50), (160, 255, 255)]],
        'bgr': (255, 0, 128),
        'found': False
    },
    '분홍': {
        'range': [[(160, 50, 50), (170, 255, 255)]],
        'bgr': (203, 192, 255),
        'found': False
    }
}

# UI 설정
UI_COLORS = {
    'overlay_bg': (0, 0, 0),
    'board_bg': (50, 50, 50),
    'timer': (0, 255, 0),
    'score': (255, 255, 255),
    'bingo': (255, 255, 0),
    'detected': (0, 255, 255),
    'game_over': (0, 0, 255),
    'start_msg': (0, 255, 0)
}

# 보드/폰트 설정
CELL_SIZE = 120
BOARD_SCALE = 2  # 빙고판 스케일(배율)
OVERLAY_ALPHA = 0.7

# 폰트 크기 설정
FONT_SIZES = {
    'timer': 32,      # 시간 표시
    'score': 32,      # 점수 표시
    'guide': 32,      # 안내 문구
    'cell': 32        # 칸 텍스트 (BOARD_SCALE 적용 전 기준)
}
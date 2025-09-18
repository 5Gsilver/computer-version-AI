# -*- coding: utf-8 -*-
"""
색깔 빙고 게임 로직
"""

import cv2
import numpy as np
import random
import time
from config import CAMERA_WIDTH, CAMERA_HEIGHT
from game import ColorBingoGame
from ui import render_overlay

# game 모듈의 ColorBingoGame을 사용하므로 이곳 정의는 제거됨


if __name__ == "__main__":

    # 웹캠 초기화
    # cap = cv2.VideoCapture(0, cv2.CAP_AVFOUNDATION)  # 0→1→2… 시도
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)

    game = ColorBingoGame()
    running = False

    print("SPACE: 시작/재시작 | R: 빙고판 리셋 | ESC: 종료")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("⚠️ 웹캠에서 프레임을 읽을 수 없습니다. 카메라 연결을 확인하세요.")
            break

        # 셀피 뷰
        frame = cv2.flip(frame, 1)

        # 진행 중일 때만 검출
        if running:
            remaining = game.get_remaining_time()
            if remaining <= 0:
                running = False
            else:
                detected = game.detect_color_in_frame(frame)
                if detected:
                    # 가장 많은 픽셀을 가진 색상 하나만 처리
                    detected.sort(key=lambda x: x[1], reverse=True)
                    game.process_color_detection(detected[0][0])

        # HUD/보드 오버레이 렌더링 (ui.render_overlay)
        frame = render_overlay(frame, game, running)

        # 창 표시
        cv2.imshow("색깔 빙고", frame)

        # 키 입력
        key = cv2.waitKey(1) & 0xFF
        if key == 27:  # ESC
            break
        elif key in (ord(' '), ord('S'), ord('s')):  # SPACE 또는 S
            game.start_game()
            running = True
        elif key in (ord('R'), ord('r')):
            game.reset()
            running = False

    cap.release()
    cv2.destroyAllWindows()
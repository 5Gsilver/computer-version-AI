# -*- coding: utf-8 -*-
"""
색깔 빙고 게임 로직
"""

import cv2
import numpy as np
import random
import time
from config import COLORS, GAME_TIME, DETECTION_THRESHOLD, COLOR_SCORE, BINGO_SCORE

class ColorBingoGame:
    def __init__(self):
        self.game_time = GAME_TIME
        self.start_time = None
        self.score = 0
        self.bingo_count = 0
        self.colors = {name: info.copy() for name, info in COLORS.items()}
        self.create_bingo_board()

    def create_bingo_board(self):
        """3x3 빙고판을 랜덤하게 생성"""
        color_names = list(self.colors.keys())
        random.shuffle(color_names)

        self.bingo_board = []
        for i in range(3):
            row = []
            for j in range(3):
                color_name = color_names[i*3 + j]
                row.append(color_name)
            self.bingo_board.append(row)

        print("=== 🎯 색깔 빙고 게임 시작! ===")
        self.print_bingo_board()

    def print_bingo_board(self):
        """빙고판을 콘솔에 출력"""
        print("\n🎯 현재 빙고판:")
        for row in self.bingo_board:
            row_str = ""
            for color in row:
                status = "✅" if self.colors[color]['found'] else "⬜"
                row_str += f"{status} {color:^6} "
            print(f"  {row_str}")
        print()

    def detect_color_in_frame(self, frame):
        """프레임에서 색상 검출"""
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        detected_colors = []

        for color_name, color_info in self.colors.items():
            if color_info['found']:
                continue

            combined_mask = self._create_color_mask(hsv, color_info['range'])
            pixel_count = cv2.countNonZero(combined_mask)

            if pixel_count > DETECTION_THRESHOLD:
                detected_colors.append((color_name, pixel_count, combined_mask))

        return detected_colors

    def _create_color_mask(self, hsv, color_ranges):
        """색상 범위로 마스크 생성"""
        combined_mask = np.zeros(hsv.shape[:2], dtype=np.uint8)

        for color_range in color_ranges:
            mask = cv2.inRange(hsv, np.array(color_range[0]), np.array(color_range[1]))
            combined_mask = cv2.bitwise_or(combined_mask, mask)

        # 모폴로지 연산으로 노이즈 제거
        kernel = np.ones((5,5), np.uint8)
        combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_OPEN, kernel)
        combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, kernel)

        return combined_mask

    def process_color_detection(self, color_name):
        """색상 검출 처리"""
        if not self.colors[color_name]['found']:
            self.colors[color_name]['found'] = True
            self.score += COLOR_SCORE

            print(f"🎉 {color_name} 발견! (+{COLOR_SCORE}점)")
            self.print_bingo_board()

            # 빙고 체크
            bingo_lines = self.check_bingo()
            if bingo_lines > self.bingo_count:
                new_bingo = bingo_lines - self.bingo_count
                self.bingo_count = bingo_lines
                bonus_score = BINGO_SCORE * new_bingo
                self.score += bonus_score
                print(f"🎊 빙고! {bingo_lines}줄 완성! (+{bonus_score}점)")

    def check_bingo(self):
        """빙고 완성 체크"""
        bingo_lines = 0

        # 가로 체크
        for row in self.bingo_board:
            if all(self.colors[color]['found'] for color in row):
                bingo_lines += 1

        # 세로 체크
        for j in range(3):
            if all(self.colors[self.bingo_board[i][j]]['found'] for i in range(3)):
                bingo_lines += 1

        # 대각선 체크
        if all(self.colors[self.bingo_board[i][i]]['found'] for i in range(3)):
            bingo_lines += 1

        if all(self.colors[self.bingo_board[i][2-i]]['found'] for i in range(3)):
            bingo_lines += 1

        return bingo_lines

    def get_remaining_time(self):
        """남은 시간 계산"""
        if self.start_time is None:
            return self.game_time

        elapsed = time.time() - self.start_time
        remaining = max(0, self.game_time - elapsed)
        return remaining

    def start_game(self):
        """게임 시작"""
        self.start_time = time.time()
        print("🎮 게임이 시작되었습니다!")
        print("📹 웹캠으로 색깔 물체를 보여주세요!")

    def reset(self):
        """게임 리셋"""
        self.__init__()

    def get_final_results(self):
        """최종 결과 반환"""
        if self.bingo_count >= 3:
            message = "🌟 축하합니다! 빙고 마스터입니다!"
        elif self.bingo_count >= 1:
            message = "👏 잘했습니다! 다음에 더 도전해보세요!"
        else:
            message = "💪 아쉽네요! 다시 도전해보세요!"

        return {
            'score': self.score,
            'bingo_count': self.bingo_count,
            'message': message
        }
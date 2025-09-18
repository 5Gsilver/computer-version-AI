# -*- coding: utf-8 -*-
"""
UI 렌더링 모듈: 한글 폰트 렌더링과 HUD/빙고판 오버레이 표시
"""

import cv2
import numpy as np
from typing import Optional, Tuple
from PIL import Image, ImageDraw, ImageFont
import os
import sys
from config import UI_COLORS, OVERLAY_ALPHA, CELL_SIZE, BOARD_SCALE, FONT_SIZES


def get_font_paths_by_os() -> list:
    env_path = os.environ.get("FON	T_PATH")
    candidates = []
    if env_path and os.path.isfile(env_path):
        candidates.append(env_path)

    platform_key = sys.platform
    if platform_key.startswith("darwin"):
        candidates += [
            "/System/Library/Fonts/AppleSDGothicNeo.ttc",
            "/Library/Fonts/AppleGothic.ttf",
            "/Library/Fonts/NanumGothic.ttf",
            "/System/Library/Fonts/Supplemental/NotoSansCJK-Regular.ttc",
        ]
    elif platform_key.startswith("win"):
        candidates += [
            "C:\\Windows\\Fonts\\malgun.ttf",
            "C:\\Windows\\Fonts\\malgunbd.ttf",
            "C:\\Windows\\Fonts\\gulim.ttc",
            "C:\\Windows\\Fonts\\batang.ttc",
            "C:\\Windows\\Fonts\\NotoSansCJK-Regular.ttc",
        ]
    else:
        candidates += [
            "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",
            "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
            "/usr/share/fonts/truetype/nanum/NanumGothic.ttf",
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        ]
    uniq = []
    for p in candidates:
        if p and p not in uniq and os.path.isfile(p):
            uniq.append(p)
    return uniq


_FONT_CANDIDATES = get_font_paths_by_os()


def _load_font(font_size: int) -> Optional[ImageFont.FreeTypeFont]:
    for path in _FONT_CANDIDATES:
        try:
            return ImageFont.truetype(path, font_size)
        except Exception:
            continue
    return None


def draw_korean_text_bgr(
    frame_bgr: np.ndarray,
    text: str,
    org: Tuple[int, int],
    font_size: int = 20,
    color_bgr: Tuple[int, int, int] = (255, 255, 255),
    anchor: str = "la",
) -> np.ndarray:
    font = _load_font(font_size)
    if font is None:
        cv2.putText(frame_bgr, text, org, cv2.FONT_HERSHEY_SIMPLEX, max(0.5, font_size/32.0), color_bgr, 2, cv2.LINE_AA)
        return frame_bgr
    img_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(img_rgb)
    draw = ImageDraw.Draw(pil_img)
    color_rgb = (color_bgr[2], color_bgr[1], color_bgr[0])
    draw.text(org, text, fill=color_rgb, font=font, anchor=anchor)
    return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)


def draw_korean_text_center_in_rect(
    frame_bgr: np.ndarray,
    text: str,
    rect: Tuple[int, int, int, int],
    font_size: int,
    color_bgr: Tuple[int, int, int],
) -> np.ndarray:
    x1, y1, x2, y2 = rect
    font = _load_font(font_size)
    if font is None:
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, max(0.5, font_size/32.0), 1)
        tx = x1 + (x2 - x1 - tw) // 2
        ty = y1 + (y2 - y1 + th) // 2
        cv2.putText(frame_bgr, text, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, max(0.5, font_size/32.0), color_bgr, 1, cv2.LINE_AA)
        return frame_bgr
    img_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(img_rgb)
    draw = ImageDraw.Draw(pil_img)
    color_rgb = (color_bgr[2], color_bgr[1], color_bgr[0])
    bbox = draw.textbbox((0, 0), text, font=font)
    tw = bbox[2] - bbox[0]
    th = bbox[3] - bbox[1]
    tx = x1 + (x2 - x1 - tw) // 2
    ty = y1 + (y2 - y1 - th) // 2
    draw.text((tx, ty), text, fill=color_rgb, font=font)
    return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)


def render_overlay(frame: np.ndarray, game, running: bool) -> np.ndarray:
    # HUD 패널
    overlay = frame.copy()
    cv2.rectangle(overlay, (10, 10), (320, 110), UI_COLORS['overlay_bg'], -1)
    cv2.addWeighted(overlay, OVERLAY_ALPHA, frame, 1 - OVERLAY_ALPHA, 0, frame)

    # 타이머/점수
    timer_color = UI_COLORS['timer'] if running else UI_COLORS['game_over']
    remaining_text = int(game.get_remaining_time())
    frame = draw_korean_text_bgr(frame, f"시간: {remaining_text:>4}초", (20, 45), FONT_SIZES['timer'], timer_color, "la")
    frame = draw_korean_text_bgr(frame, f"점수: {game.score}", (20, 80), FONT_SIZES['score'], UI_COLORS['score'], "la")

    # 가이드 문구
    if not running and game.start_time is None:
        frame = draw_korean_text_bgr(frame, "SPACE: 게임 시작", (360, 45), FONT_SIZES['guide'], UI_COLORS['start_msg'], "la")
    elif not running and game.get_remaining_time() <= 0:
        results = game.get_final_results()
        msg = results['message']
        frame = draw_korean_text_bgr(frame, msg, (360, 45), FONT_SIZES['guide'], UI_COLORS['bingo'], "la")
        frame = draw_korean_text_bgr(frame, "SPACE: 재시작  |  R: 빙고판 리셋", (360, 80), max(12, FONT_SIZES['guide']-2), UI_COLORS['start_msg'], "la")

    # 빙고판(2배 확대 유지)
    CELL = CELL_SIZE * BOARD_SCALE
    board_w = CELL * 3
    board_h = CELL * 3
    board_x = frame.shape[1] - board_w - 20
    board_y = 20

    board_layer = frame.copy()
    cv2.rectangle(board_layer, (board_x - 10, board_y - 10), (board_x + board_w + 10, board_y + board_h + 10), UI_COLORS['board_bg'], -1)
    cv2.addWeighted(board_layer, OVERLAY_ALPHA, frame, 1 - OVERLAY_ALPHA, 0, frame)

    for i in range(3):
        for j in range(3):
            cell_x1 = board_x + j * CELL
            cell_y1 = board_y + i * CELL
            cell_x2 = cell_x1 + CELL
            cell_y2 = cell_y1 + CELL

            color_name = game.bingo_board[i][j]
            info = game.colors[color_name]
            bgr = info['bgr']

            # 칸 배경(반투명)
            cell_layer = frame.copy()
            cv2.rectangle(cell_layer, (cell_x1, cell_y1), (cell_x2, cell_y2), bgr, -1)
            cv2.addWeighted(cell_layer, 0.35, frame, 0.65, 0, frame)

            # 테두리
            border_color = (0, 255, 0) if info['found'] else (200, 200, 200)
            thickness = 2 if info['found'] else 1
            cv2.rectangle(frame, (cell_x1, cell_y1), (cell_x2, cell_y2), border_color, thickness)

            # 텍스트
            brightness = int(0.299 * bgr[2] + 0.587 * bgr[1] + 0.114 * bgr[0])
            text_color = (0, 0, 0) if brightness > 160 else (255, 255, 255)
            frame = draw_korean_text_center_in_rect(frame, color_name, (cell_x1, cell_y1, cell_x2, cell_y2), int(FONT_SIZES['cell'] * (BOARD_SCALE/2)), text_color)

            # 체크
            if info['found']:
                cx1 = cell_x1 + int(CELL * 0.2)
                cy1 = cell_y1 + int(CELL * 0.55)
                cx2 = cell_x1 + int(CELL * 0.45)
                cy2 = cell_y1 + int(CELL * 0.8)
                cx3 = cell_x1 + int(CELL * 0.85)
                cy3 = cell_y1 + int(CELL * 0.25)
                cv2.line(frame, (cx1, cy1), (cx2, cy2), (0, 255, 0), 4, cv2.LINE_AA)
                cv2.line(frame, (cx2, cy2), (cx3, cy3), (0, 255, 0), 4, cv2.LINE_AA)

    return frame
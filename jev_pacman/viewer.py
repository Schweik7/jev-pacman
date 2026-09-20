"""在终端里实时看对局（ANSI 真彩色，不依赖任何第三方库）。

和 GifRecorder 一样实现 capture(env, label)，所以能直接传给 run_turn_based / run_realtime：

    from jev_pacman.viewer import TerminalViewer
    run_realtime(env, agent, recorder=TerminalViewer())

每格画成两个字符宽，让 28x31 的迷宫在终端里接近正方形。
"""
from __future__ import annotations

import os
import sys
import time

from .maze import HEIGHT, LAYOUT, WIDTH

RESET = "\x1b[0m"
WALL = (33, 33, 222)
DOOR = (255, 184, 222)
PELLET = (255, 200, 170)
PAC = (255, 235, 59)
GHOST_COLORS = {"blinky": (255, 0, 0), "pinky": (255, 140, 220),
                "inky": (0, 230, 255), "clyde": (255, 160, 60)}
FRIGHTENED = (40, 60, 255)
FLASH = (240, 240, 240)
EYES = (150, 160, 200)
PAC_FACE = {"RIGHT": "▶", "LEFT": "◀", "UP": "▲", "DOWN": "▼"}


def fg(rgb):
    return f"\x1b[38;2;{rgb[0]};{rgb[1]};{rgb[2]}m"


def bg(rgb):
    return f"\x1b[48;2;{rgb[0]};{rgb[1]};{rgb[2]}m"


def _cell(env, r, c, ghosts, flashing):
    """返回 (样式转义码, 两个字符的图形)。"""
    cell = (r, c)
    if cell == tuple(env.pac):
        return fg(PAC), " " + PAC_FACE.get(env.pac_dir, "●")
    if cell in ghosts:
        g = ghosts[cell]
        if g.state == "eyes":
            return fg(EYES), '""'
        if g.state == "frightened":
            return bg(FLASH if flashing else FRIGHTENED), "  "
        return bg(GHOST_COLORS.get(g.name, (255, 0, 0))), "  "
    if cell in env.powers:
        return fg(PELLET), " ●"
    if cell in env.pellets:
        return fg(PELLET), " ·"
    if LAYOUT[r][c] == "#":
        return fg(WALL), "██"
    if LAYOUT[r][c] == "-":
        return fg(DOOR), "──"
    return "", "  "


def render(env, label="", extra=""):
    """把当前局面渲染成一段带颜色的多行字符串。相邻同色的格子共用一个转义码。"""
    ghosts = {tuple(g.pos): g for g in env.ghosts}
    flashing = 0 < env.fright_timer < 12 and env.fright_timer % 4 < 2
    lines = []
    for r in range(HEIGHT):
        row, style = [], ""
        for c in range(WIDTH):
            s, glyph = _cell(env, r, c, ghosts, flashing)
            if s != style:
                row.append(RESET + s)  # 先清掉上一格的前景/背景色，再换新样式
                style = s
            row.append(glyph)
        if style:
            row.append(RESET)
        row.append("\x1b[K")  # 擦掉这一行残留的旧画面
        lines.append("".join(row))
    eaten = env.total_food - len(env.pellets) - len(env.powers)
    hud = (f" {label}  分数 {env.score:<6} 命 {'♥' * max(env.lives, 0):<3} "
           f"tick {env.tick:<5} 豆 {eaten}/{env.total_food}")
    if env.fright_timer:
        hud += f"  受惊 {env.fright_timer}"
    if extra:
        hud += f"  {extra}"
    lines.append(hud + "\x1b[K")
    return "\n".join(lines)


class TerminalViewer:
    """每帧把画面重绘在同一块屏幕区域上。min_frame_ms 用来限制刷新率。"""

    def __init__(self, min_frame_ms=60, stream=None, label_agent=True):
        self.min_frame_ms = min_frame_ms
        self.stream = stream or sys.stdout
        self.label_agent = label_agent
        self._last = 0.0
        self._started = False
        if os.name == "nt":
            os.system("")  # 打开 Windows 控制台的 ANSI 转义支持

    def capture(self, env, label=""):
        now = time.perf_counter()
        if self._started and (now - self._last) * 1000 < self.min_frame_ms:
            return
        self._last = now
        if not self._started:
            self.stream.write("\x1b[2J\x1b[?25l")  # 清屏 + 藏光标
            self._started = True
        self.stream.write("\x1b[H" + render(env, label if self.label_agent else "") + "\n")
        self.stream.flush()

    def close(self):
        if self._started:
            self.stream.write("\x1b[?25h\n")  # 恢复光标
            self.stream.flush()
            self._started = False


class Tee:
    """把同一帧同时喂给多个观察者（例如终端 + GIF）。"""

    def __init__(self, *targets):
        self.targets = [t for t in targets if t is not None]

    def capture(self, env, label=""):
        for t in self.targets:
            t.capture(env, label)

    def close(self):
        for t in self.targets:
            close = getattr(t, "close", None)
            if callable(close):
                close()

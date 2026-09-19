"""把对局录成 GIF（依赖 Pillow）。"""
from __future__ import annotations

from PIL import Image, ImageDraw

from .maze import HEIGHT, LAYOUT, WIDTH

GHOST_COLORS = {"blinky": (255, 0, 0), "pinky": (255, 184, 255), "inky": (0, 255, 255), "clyde": (255, 184, 82)}
MOUTH = {"RIGHT": 0, "DOWN": 90, "LEFT": 180, "UP": 270}


class GifRecorder:
    def __init__(self, cell=12, every=1, max_frames=3000):
        self.cell = cell
        self.every = every
        self.max_frames = max_frames
        self.frames = []
        self._n = 0
        self.hud = 18
        self._bg = self._background()

    def _background(self):
        s = self.cell
        img = Image.new("RGB", (WIDTH * s, HEIGHT * s + self.hud), (0, 0, 0))
        d = ImageDraw.Draw(img)
        for r, row in enumerate(LAYOUT):
            for c, ch in enumerate(row):
                if ch == "#":
                    d.rectangle([c * s + 1, r * s + 1, c * s + s - 2, r * s + s - 2], outline=(33, 33, 222))
                elif ch == "-":
                    d.rectangle([c * s, r * s + s // 2 - 1, c * s + s - 1, r * s + s // 2], fill=(255, 184, 222))
        return img

    def capture(self, env, label=""):
        self._n += 1
        if (self._n - 1) % self.every or len(self.frames) >= self.max_frames:
            return
        s = self.cell
        img = self._bg.copy()
        d = ImageDraw.Draw(img)
        for r, c in env.pellets:
            cx, cy = c * s + s // 2, r * s + s // 2
            d.rectangle([cx - 1, cy - 1, cx, cy], fill=(255, 184, 174))
        for r, c in env.powers:
            d.ellipse([c * s + 2, r * s + 2, c * s + s - 3, r * s + s - 3], fill=(255, 184, 174))
        for g in env.ghosts:
            r, c = g.pos
            box = [c * s, r * s, c * s + s - 1, r * s + s - 1]
            if g.state == "eyes":
                d.ellipse([c * s + 2, r * s + 3, c * s + 5, r * s + 6], fill=(255, 255, 255))
                d.ellipse([c * s + s - 6, r * s + 3, c * s + s - 3, r * s + 6], fill=(255, 255, 255))
                continue
            if g.state == "frightened":
                flash = env.fright_timer < 12 and env.fright_timer % 4 < 2
                color = (255, 255, 255) if flash else (33, 33, 255)
            else:
                color = GHOST_COLORS.get(g.name, (255, 0, 0))
            d.pieslice(box, 180, 360, fill=color)
            d.rectangle([box[0], r * s + s // 2, box[2], box[3]], fill=color)
        r, c = env.pac
        a = MOUTH[env.pac_dir]
        opening = 35 if env.tick % 2 == 0 else 5
        d.pieslice([c * s, r * s, c * s + s - 1, r * s + s - 1], a + opening, a - opening + 360, fill=(255, 255, 0))
        text = f"{label}  score {env.score}  lives {env.lives}  tick {env.tick}"
        d.text((4, HEIGHT * s + 3), text.strip(), fill=(255, 255, 255))
        self.frames.append(img)

    def save(self, path, frame_ms=100):
        if not self.frames:
            return
        # 结尾停留 1.5 秒方便看结果
        durations = [frame_ms] * (len(self.frames) - 1) + [1500]
        self.frames[0].save(path, save_all=True, append_images=self.frames[1:],
                            duration=durations, loop=0, optimize=True)

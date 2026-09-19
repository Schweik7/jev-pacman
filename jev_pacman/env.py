"""吃豆人环境：网格化、按 tick 推进，接口仿照 Gymnasium（reset / step 返回五元组）。"""
from __future__ import annotations

import random
from collections import deque
from dataclasses import dataclass, field

import numpy as np

from .maze import GHOST_EXIT, GHOST_SPECS, HEIGHT, LAYOUT, PAC_START, TUNNEL_ROW, WIDTH

ACTIONS = ("NOOP", "UP", "DOWN", "LEFT", "RIGHT")
DIRS = {"UP": (-1, 0), "DOWN": (1, 0), "LEFT": (0, -1), "RIGHT": (0, 1)}
OPPOSITE = {"UP": "DOWN", "DOWN": "UP", "LEFT": "RIGHT", "RIGHT": "LEFT"}
GHOST_PRIORITY = ("UP", "LEFT", "DOWN", "RIGHT")  # 原版平局时的方向优先级

# grid 观测里的编码
EMPTY, WALL, PELLET, POWER, DOOR, PACMAN, GHOST, SCARED, EYES = range(9)
PLANE_NAMES = ("wall", "pellet", "power", "pacman", "hunting_ghost", "frightened_ghost", "eyes")


@dataclass
class PacmanConfig:
    lives: int = 3
    max_ticks: int = 5000
    num_ghosts: int = 4
    ghost_speed: float = 0.9          # 相对吃豆人（每 tick 1 格）的速度
    frightened_speed: float = 0.5
    tunnel_speed: float = 0.5
    eyes_speed: float = 2.0
    frightened_ticks: int = 45
    respawn_ticks: int = 15           # 被吃掉的幽灵回屋后等待多久再出来
    # 分散 / 追击 交替时间表，最后一段持续到游戏结束
    schedule: tuple = (("scatter", 50), ("chase", 150), ("scatter", 50), ("chase", 150),
                       ("scatter", 40), ("chase", 150), ("scatter", 40), ("chase", None))
    death_penalty: float = -200.0
    win_bonus: float = 500.0
    obs_type: str = "state"           # state | grid | planes | text


@dataclass
class Ghost:
    name: str
    start: tuple
    scatter_target: tuple
    release_tick: int
    pos: tuple = (0, 0)
    dir: str = "LEFT"
    state: str = "house"              # house | hunting | frightened | eyes
    house_timer: int = 0
    acc: float = 0.0


class PacmanEnv:
    """单关吃豆人。吃光所有豆子即胜利，命数用完即失败。

    每个 tick：吃豆人移动 1 格 → 幽灵按各自速度移动 → 判定碰撞。
    动作可以是 ACTIONS 中的下标或名字；NOOP 表示保持之前的意图方向。
    转向意图会被缓存：想拐的方向暂时是墙时，会继续直走，到路口再拐（与原版一致）。
    """

    def __init__(self, config: PacmanConfig | None = None, seed: int | None = None):
        self.cfg = config or PacmanConfig()
        self.rng = random.Random(seed)
        self._exit_dist = self._bfs_map(GHOST_EXIT, self.ghost_walkable)
        self.reset(seed=seed)

    # ------------------------------------------------------------------ 基础几何
    def neighbor(self, pos, d):
        dr, dc = DIRS[d]
        return pos[0] + dr, (pos[1] + dc) % WIDTH

    def _tile(self, pos):
        r, c = pos
        if not 0 <= r < HEIGHT:
            return "#"
        return LAYOUT[r][c]

    def pac_walkable(self, pos):
        return self._tile(pos) not in "#-"

    ghost_walkable = pac_walkable

    def legal_moves(self, pos=None):
        pos = pos or self.pac
        return [d for d in DIRS if self.pac_walkable(self.neighbor(pos, d))]

    def _bfs_map(self, start, walkable):
        dist = {start: 0}
        q = deque([start])
        while q:
            cur = q.popleft()
            for d in DIRS:
                n = self.neighbor(cur, d)
                if n not in dist and walkable(n):
                    dist[n] = dist[cur] + 1
                    q.append(n)
        return dist

    # ------------------------------------------------------------------ 生命周期
    def reset(self, seed=None):
        if seed is not None:
            self.rng.seed(seed)
        self.pellets = set()
        self.powers = set()
        for r, row in enumerate(LAYOUT):
            for c, ch in enumerate(row):
                if ch == ".":
                    self.pellets.add((r, c))
                elif ch == "o":
                    self.powers.add((r, c))
        self.total_food = len(self.pellets) + len(self.powers)
        self.score = 0
        self.lives = self.cfg.lives
        self.tick = 0
        self.mode_idx = 0
        self.mode_timer = self.cfg.schedule[0][1]
        self.done = False
        self.win = False
        self._reset_positions()
        return self._obs(), self._info([])

    def _reset_positions(self):
        self.pac = PAC_START
        self.pac_dir = "LEFT"
        self.pac_want = None
        self.life_tick = 0
        self.fright_timer = 0
        self.combo = 0
        self.ghosts = []
        for name, start, corner, release in GHOST_SPECS[: self.cfg.num_ghosts]:
            g = Ghost(name, start, corner, release, pos=start)
            g.state = "hunting" if release == 0 else "house"
            self.ghosts.append(g)

    @property
    def mode(self):
        return self.cfg.schedule[self.mode_idx][0]

    # ------------------------------------------------------------------ 主循环
    def step(self, action):
        if self.done:
            raise RuntimeError("回合已结束，请先调用 reset()")
        name = ACTIONS[action] if isinstance(action, (int, np.integer)) else str(action).upper()
        if name not in ACTIONS:
            raise ValueError(f"未知动作: {action!r}")
        if name != "NOOP":
            self.pac_want = name

        score_before = self.score
        bonus = 0.0
        events = []
        self.tick += 1
        self.life_tick += 1

        # 1. 吃豆人移动
        if self.pac_want and self.pac_walkable(self.neighbor(self.pac, self.pac_want)):
            self.pac_dir = self.pac_want
        nxt = self.neighbor(self.pac, self.pac_dir)
        if self.pac_walkable(nxt):
            self.pac = nxt
        if self.pac in self.pellets:
            self.pellets.discard(self.pac)
            self.score += 10
        elif self.pac in self.powers:
            self.powers.discard(self.pac)
            self.score += 50
            self._frighten()
            events.append("power")

        died = self._collide_all(events)

        # 2. 模式计时 & 幽灵移动
        if not died:
            self._update_mode()
            for g in self.ghosts:
                if g.state == "house":
                    self._house_update(g)
                    continue
                g.acc += self._ghost_speed(g)
                while g.acc >= 1 and not died:
                    g.acc -= 1
                    self._ghost_move(g)
                    if g.state != "house":
                        died = self._collide(g, events)
                if died:
                    break

        if died:
            self.lives -= 1
            bonus += self.cfg.death_penalty
            events.append("death")
            if self.lives <= 0:
                self.done = True
                events.append("game_over")
            else:
                self._reset_positions()

        if not self.pellets and not self.powers:
            self.done = True
            self.win = True
            bonus += self.cfg.win_bonus
            events.append("win")

        terminated = self.done
        truncated = not terminated and self.tick >= self.cfg.max_ticks
        if truncated:
            self.done = True
        reward = float(self.score - score_before) + bonus
        return self._obs(), reward, terminated, truncated, self._info(events)

    # ------------------------------------------------------------------ 幽灵
    def _frighten(self):
        self.fright_timer = self.cfg.frightened_ticks
        self.combo = 0
        for g in self.ghosts:
            if g.state == "hunting":
                g.state = "frightened"
                g.dir = OPPOSITE[g.dir]
                g.acc = 0.0

    def _update_mode(self):
        if self.fright_timer > 0:  # 受惊期间分散/追击计时暂停（原版规则）
            self.fright_timer -= 1
            if self.fright_timer == 0:
                for g in self.ghosts:
                    if g.state == "frightened":
                        g.state = "hunting"
            return
        if self.mode_timer is None:
            return
        self.mode_timer -= 1
        if self.mode_timer <= 0:
            self.mode_idx += 1
            self.mode_timer = self.cfg.schedule[self.mode_idx][1]
            for g in self.ghosts:  # 模式切换时幽灵掉头
                if g.state == "hunting":
                    g.dir = OPPOSITE[g.dir]

    def _house_update(self, g):
        if g.house_timer > 0:
            g.house_timer -= 1
            ready = g.house_timer == 0
        else:
            ready = self.life_tick >= g.release_tick
        if ready:
            g.pos = GHOST_EXIT
            g.dir = "LEFT"
            g.state = "hunting"
            g.acc = 0.0

    def _ghost_speed(self, g):
        if g.state == "eyes":
            return self.cfg.eyes_speed
        speed = self.cfg.frightened_speed if g.state == "frightened" else self.cfg.ghost_speed
        r, c = g.pos
        if r == TUNNEL_ROW and (c <= 5 or c >= WIDTH - 6):
            speed = min(speed, self.cfg.tunnel_speed)
        return speed

    def _ghost_target(self, g):
        pr, pc = self.pac
        dr, dc = DIRS[self.pac_dir]
        if self.mode == "scatter":
            return g.scatter_target
        if g.name == "pinky":
            return pr + 4 * dr, pc + 4 * dc
        if g.name == "inky":
            blinky = next((x for x in self.ghosts if x.name == "blinky"), None)
            if blinky is None:
                return self.pac
            vr, vc = pr + 2 * dr, pc + 2 * dc
            return 2 * vr - blinky.pos[0], 2 * vc - blinky.pos[1]
        if g.name == "clyde":
            if (g.pos[0] - pr) ** 2 + (g.pos[1] - pc) ** 2 <= 64:
                return g.scatter_target
        return self.pac

    def _ghost_move(self, g):
        options = [d for d in GHOST_PRIORITY if self.ghost_walkable(self.neighbor(g.pos, d))]
        if g.state == "eyes":  # 眼睛沿最短路回屋，可以掉头
            d = min(options, key=lambda d: self._exit_dist.get(self.neighbor(g.pos, d), 10**6))
        else:
            forward = [d for d in options if d != OPPOSITE[g.dir]] or options
            if g.state == "frightened":
                d = self.rng.choice(forward)
            else:
                tr, tc = self._ghost_target(g)

                def cost(d):
                    nr, nc = self.neighbor(g.pos, d)
                    return (nr - tr) ** 2 + (nc - tc) ** 2

                d = min(forward, key=cost)  # min 取第一个最小值，天然符合优先级
        g.pos = self.neighbor(g.pos, d)
        g.dir = d
        if g.state == "eyes" and g.pos == GHOST_EXIT:
            g.state = "house"
            g.pos = g.start if g.start != GHOST_EXIT else (14, 13)
            g.house_timer = self.cfg.respawn_ticks
            g.acc = 0.0

    def _collide(self, g, events):
        """返回 True 表示吃豆人死亡。"""
        if g.pos != self.pac or g.state in ("house", "eyes"):
            return False
        if g.state == "frightened":
            self.score += 200 * 2 ** min(self.combo, 3)
            self.combo += 1
            g.state = "eyes"
            g.acc = 0.0
            events.append(f"eat_{g.name}")
            return False
        return True

    def _collide_all(self, events):
        died = False
        for g in self.ghosts:
            died = self._collide(g, events) or died
        return died

    # ------------------------------------------------------------------ 观测
    def state(self, include_map=True):
        from .features import describe
        return describe(self, include_map=include_map)

    def grid(self):
        g = np.zeros((HEIGHT, WIDTH), dtype=np.int8)
        for r, row in enumerate(LAYOUT):
            for c, ch in enumerate(row):
                if ch == "#":
                    g[r, c] = WALL
                elif ch == "-":
                    g[r, c] = DOOR
        for p in self.pellets:
            g[p] = PELLET
        for p in self.powers:
            g[p] = POWER
        for gh in self.ghosts:
            if gh.state != "house":
                g[gh.pos] = {"hunting": GHOST, "frightened": SCARED, "eyes": EYES}[gh.state]
        g[self.pac] = PACMAN
        return g

    def planes(self):
        """(7, H, W) 的 0/1 平面，适合 CNN。通道顺序见 PLANE_NAMES。"""
        g = self.grid()
        out = np.zeros((len(PLANE_NAMES), HEIGHT, WIDTH), dtype=np.float32)
        for i, code in enumerate((WALL, PELLET, POWER, PACMAN, GHOST, SCARED, EYES)):
            out[i] = g == code
        out[0] += g == DOOR
        # 豆子可能被吃豆人/幽灵遮住，单独补上
        for p in self.pellets:
            out[1][p] = 1
        for p in self.powers:
            out[2][p] = 1
        return out

    def render_text(self):
        from .features import ascii_map
        return ascii_map(self)

    def _obs(self):
        t = self.cfg.obs_type
        if t == "state":
            return self.state()
        if t == "grid":
            return self.grid()
        if t == "planes":
            return self.planes()
        if t == "text":
            return self.render_text()
        raise ValueError(f"未知 obs_type: {t}")

    def _info(self, events):
        return {
            "score": self.score,
            "lives": self.lives,
            "tick": self.tick,
            "pellets_left": len(self.pellets) + len(self.powers),
            "win": self.win,
            "events": events,
        }

"""把局面整理成结构化状态，供 Jev 这类“一次前向、做选择题”的模型使用。

Jev 不会自己在地图上搜路，所以这里提前算好每个可走方向的关键信息
（离豆子多远、离危险幽灵多远等），模型只需要在几个方向里挑一个。
"""
from __future__ import annotations

import math
from collections import deque

from .maze import LAYOUT

MAP_LEGEND = {
    "#": "wall", ".": "pellet", "o": "power pellet", "-": "ghost-house door",
    "@": "pac-man", "G": "hunting ghost (deadly)", "f": "frightened ghost (edible)",
    "e": "ghost eyes (harmless)",
}


def ascii_map(env):
    rows = [list(r.replace(".", " ").replace("o", " ")) for r in LAYOUT]
    for r, c in env.pellets:
        rows[r][c] = "."
    for r, c in env.powers:
        rows[r][c] = "o"
    for g in env.ghosts:
        if g.state != "house":
            rows[g.pos[0]][g.pos[1]] = {"hunting": "G", "frightened": "f", "eyes": "e"}[g.state]
    rows[env.pac[0]][env.pac[1]] = "@"
    return "\n".join("".join(r) for r in rows)


def _bfs(env, start, blocked=None):
    dist = {start: 0}
    q = deque([start])
    while q:
        cur = q.popleft()
        for d in ("UP", "DOWN", "LEFT", "RIGHT"):
            n = env.neighbor(cur, d)
            if n not in dist and n != blocked and env.pac_walkable(n):
                dist[n] = dist[cur] + 1
                q.append(n)
    return dist


def _nearest(dist, cells, offset=1):
    best = min((dist[c] for c in cells if c in dist), default=None)
    return None if best is None else best + offset


def move_features(env):
    """对每个可走方向，从“走一步之后的格子”出发做 BFS（不允许立刻退回原位）。"""
    food = env.pellets | env.powers
    hunting = [g.pos for g in env.ghosts if g.state == "hunting"]
    scared = [g.pos for g in env.ghosts if g.state == "frightened"]
    out = {}
    for d in env.legal_moves():
        nxt = env.neighbor(env.pac, d)
        dist = _bfs(env, nxt, blocked=env.pac)
        out[d] = {
            "pellet_steps": _nearest(dist, food),
            "hunting_ghost_steps": _nearest(dist, hunting),
            "frightened_ghost_steps": _nearest(dist, scared),
            "pellets_within_6_steps": sum(1 for c in food if dist.get(c, 99) <= 5),
            "is_current_direction": d == env.pac_dir,
            "reverses_direction": d == {"UP": "DOWN", "DOWN": "UP", "LEFT": "RIGHT", "RIGHT": "LEFT"}[env.pac_dir],
        }
    return out


def describe(env, include_map=True):
    here = _bfs(env, env.pac)
    ghosts = []
    for g in env.ghosts:
        ghosts.append({
            "name": g.name,
            "state": "in_house" if g.state == "house" else g.state,
            "row": g.pos[0],
            "col": g.pos[1],
            "facing": g.dir,
            "steps_from_pacman": None if g.state == "house" else here.get(g.pos),
        })
    state = {
        "game": "Pac-Man",
        "goal": "Eat every pellet without being caught by a hunting ghost.",
        "tick": env.tick,
        "score": env.score,
        "lives": env.lives,
        "pellets_left": len(env.pellets) + len(env.powers),
        "pacman": {"row": env.pac[0], "col": env.pac[1], "facing": env.pac_dir},
        "ghost_mode": env.mode,
        "frightened_ticks_left": env.fright_timer,
        "ghosts": ghosts,
        "moves": move_features(env),
    }
    if include_map:
        state["map"] = ascii_map(env)
        state["map_legend"] = MAP_LEGEND
    return state


def heuristic_score(f, frightened_left=0):
    """手写的方向打分，基线智能体和离线模拟 Jev 共用。越大越好。"""
    s = 0.0
    h = f["hunting_ghost_steps"]
    if h is not None:
        if h <= 1:
            s -= 1000
        elif h <= 3:
            s -= 400 / h
        elif h <= 6:
            s -= 60 / h
    fg = f["frightened_ghost_steps"]
    if fg is not None and fg < frightened_left:
        s += 250 / fg
    p = f["pellet_steps"]
    if p:
        s += 40 / p
    s += 1.5 * f["pellets_within_6_steps"]
    if f["reverses_direction"]:
        s -= 3
    return s


def softmax(scores, temperature=10.0):
    m = max(scores.values())
    ex = {k: math.exp((v - m) / temperature) for k, v in scores.items()}
    z = sum(ex.values())
    return {k: v / z for k, v in ex.items()}

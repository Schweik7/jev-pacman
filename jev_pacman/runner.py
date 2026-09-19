"""两种对局方式：

- turn（回合制）：环境等智能体想好再走，考察决策质量。
- realtime（实时）：环境按固定节拍自己往前走，智能体在后台线程里不停决策，
  想得慢就会错过若干 tick，这时吃豆人沿用上一个指令。考察“又快又准”。
"""
from __future__ import annotations

import statistics
import threading
import time


def _latency_stats(xs):
    if not xs:
        return {}
    xs = sorted(xs)
    return {
        "decisions": len(xs),
        "latency_ms_mean": round(1000 * statistics.fmean(xs), 1),
        "latency_ms_p50": round(1000 * xs[len(xs) // 2], 1),
        "latency_ms_p95": round(1000 * xs[min(len(xs) - 1, int(len(xs) * 0.95))], 1),
    }


def _summary(env, agent, latencies, wall, extra=None):
    out = {
        "agent": getattr(agent, "name", type(agent).__name__),
        "score": env.score,
        "win": env.win,
        "ticks": env.tick,
        "lives_left": env.lives,
        "pellets_eaten": env.total_food - len(env.pellets) - len(env.powers),
        "pellets_total": env.total_food,
        "wall_time_s": round(wall, 1),
        **_latency_stats(latencies),
    }
    if hasattr(agent, "errors"):
        out["api_errors"] = agent.errors
        if agent.last_error:
            out["last_error"] = agent.last_error
    out.update(extra or {})
    return out


def run_turn_based(env, agent, seed=None, recorder=None):
    env.reset(seed=seed)
    latencies = []
    t_start = time.perf_counter()
    if recorder:
        recorder.capture(env, agent.name)
    while True:
        state = env.state(include_map=True)
        t0 = time.perf_counter()
        action = agent.act(state)
        latencies.append(time.perf_counter() - t0)
        *_, terminated, truncated, _ = env.step(action)
        if recorder:
            recorder.capture(env, agent.name)
        if terminated or truncated:
            break
    return _summary(env, agent, latencies, time.perf_counter() - t_start)


def run_realtime(env, agent, tick_ms=150, seed=None, recorder=None):
    env.reset(seed=seed)
    lock = threading.Lock()
    shared = {"state": env.state(), "tick": env.tick, "action": "NOOP"}
    stop = threading.Event()
    latencies, staleness = [], []

    def worker():
        while not stop.is_set():
            with lock:
                state, seen_tick = shared["state"], shared["tick"]
            t0 = time.perf_counter()
            action = agent.act(state)
            latencies.append(time.perf_counter() - t0)
            with lock:
                shared["action"] = action
                staleness.append(shared["tick"] - seen_tick)  # 决策生效时局面已经过去了几个 tick

    th = threading.Thread(target=worker, daemon=True)
    t_start = time.perf_counter()
    th.start()
    next_t = time.perf_counter()
    if recorder:
        recorder.capture(env, agent.name)
    while True:
        with lock:
            action = shared["action"]
        *_, terminated, truncated, _ = env.step(action)
        state = env.state()
        with lock:
            shared["state"], shared["tick"] = state, env.tick
        if recorder:
            recorder.capture(env, agent.name)
        if terminated or truncated:
            break
        next_t += tick_ms / 1000
        time.sleep(max(0.0, next_t - time.perf_counter()))
    stop.set()
    th.join(timeout=5)
    extra = {
        "tick_ms": tick_ms,
        "stale_ticks_mean": round(statistics.fmean(staleness), 2) if staleness else None,
    }
    return _summary(env, agent, latencies, time.perf_counter() - t_start, extra)

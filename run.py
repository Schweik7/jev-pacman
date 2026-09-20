"""命令行入口。

示例：
    python run.py --agent greedy --episodes 5
    python run.py --agent mockjev --mode realtime --tick-ms 150 --gif demo.gif
    python run.py --agent jev --mode realtime --tick-ms 200       # 拿到 Jev 资格后
"""
from __future__ import annotations

import argparse
import json
import statistics

from jev_pacman import PacmanConfig, PacmanEnv, run_realtime, run_turn_based
from jev_pacman.agents import make_agent
from jev_pacman.dotenv import load_dotenv
from jev_pacman.viewer import Tee


def main():
    load_dotenv()  # 项目根目录的 .env（OPENROUTER_API_KEY / TYPESAFE_API_KEY）
    p = argparse.ArgumentParser(description="Jev 吃豆人环境")
    p.add_argument("--agent", default="greedy", choices=["random", "greedy", "mockjev", "jev", "jev-openrouter"])
    p.add_argument("--mode", default="turn", choices=["turn", "realtime"])
    p.add_argument("--tick-ms", type=int, default=150, help="实时模式下每个 tick 的毫秒数")
    p.add_argument("--episodes", type=int, default=1)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--ghosts", type=int, default=4, choices=range(0, 5))
    p.add_argument("--lives", type=int, default=3)
    p.add_argument("--max-ticks", type=int, default=3000)
    p.add_argument("--mock-latency", default="70,500", help="模拟 Jev 的延迟范围（毫秒），如 70,500")
    p.add_argument("--no-map", action="store_true", help="不把 ASCII 地图发给模型，减少输入")
    p.add_argument("--watch", action="store_true", help="在终端里实时观看对局")
    p.add_argument("--watch-ms", type=int, default=60, help="终端画面的最小刷新间隔（毫秒）")
    p.add_argument("--gif", help="把第一局录成 GIF")
    p.add_argument("--gif-every", type=int, default=1, help="每隔几个 tick 录一帧")
    p.add_argument("--gif-ms", type=int, help="GIF 每帧的毫秒数（默认按真实节奏；调小就是加速播放）")
    p.add_argument("--json", help="把结果写入 JSON 文件")
    args = p.parse_args()

    lo, hi = (int(x) for x in args.mock_latency.split(","))
    cfg = PacmanConfig(lives=args.lives, max_ticks=args.max_ticks, num_ghosts=args.ghosts)
    env = PacmanEnv(cfg, seed=args.seed)
    agent = make_agent(args.agent, seed=args.seed, mock_latency_ms=(lo, hi), include_map=not args.no_map)

    results = []
    try:
        for ep in range(args.episodes):
            gif = None
            if args.gif and ep == 0:
                from jev_pacman.render import GifRecorder
                gif = GifRecorder(every=args.gif_every)
            viewer = None
            if args.watch:
                from jev_pacman.viewer import TerminalViewer
                viewer = TerminalViewer(min_frame_ms=args.watch_ms)
            recorder = gif if viewer is None else Tee(viewer, gif)
            seed = args.seed + ep
            try:
                if args.mode == "turn":
                    res = run_turn_based(env, agent, seed=seed, recorder=recorder)
                else:
                    res = run_realtime(env, agent, tick_ms=args.tick_ms, seed=seed, recorder=recorder)
            finally:
                if viewer:
                    viewer.close()
            res["episode"] = ep
            results.append(res)
            print(json.dumps(res, ensure_ascii=False))
            if gif:
                frame_ms = args.gif_ms or (args.tick_ms * args.gif_every if args.mode == "realtime"
                                           else 80 * args.gif_every)
                gif.save(args.gif, frame_ms=frame_ms)
                print(f"GIF 已保存: {args.gif}（{len(gif.frames)} 帧）")
    finally:
        close = getattr(agent, "close", None)
        if callable(close):
            close()

    if len(results) > 1:
        scores = [r["score"] for r in results]
        print(f"\n{args.agent} / {args.mode}: 平均分 {statistics.fmean(scores):.0f}，"
              f"最高 {max(scores)}，通关 {sum(r['win'] for r in results)}/{len(results)}")
    if args.json:
        with open(args.json, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()

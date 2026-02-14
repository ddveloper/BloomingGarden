"""Heuristic baseline for BloomingGarden gym environment.

This script does not depend on the original browser game host and can be used to
continue ML/agent experiments locally.
"""

from __future__ import annotations

import argparse
import collections
import random
import sys
from pathlib import Path
from dataclasses import dataclass
from typing import Iterable, List, Sequence, Tuple

import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "gym-bloomingGarden"))

from gym_bloomingGarden.envs.bloomingGarden_env import bloomingGardenEnv

BOARD_SIZE = 9
EMPTY = 3

Action = Tuple[int, int, int, int]


@dataclass
class EvalWeights:
    immediate_reward: float = 120.0
    future_match_len: float = 10.0
    center_bias: float = 1.2
    empty_cells: float = 1.0


class GreedyPlanner:
    def __init__(self, weights: EvalWeights | None = None):
        self.weights = weights or EvalWeights()

    def choose_action(self, env: bloomingGardenEnv) -> Action:
        actions = list(self._enumerate_actions(env.brd))
        if not actions:
            return (0, 0, 0, 1)

        best = None
        best_score = float("-inf")
        for action in actions:
            value = self._evaluate_action(env.brd, action)
            if value > best_score:
                best_score = value
                best = action
        return best if best is not None else random.choice(actions)

    def _enumerate_actions(self, board: np.ndarray) -> Iterable[Action]:
        flowers = np.argwhere(board != EMPTY)
        for sr, sc in flowers:
            reachable = self._reachable_empties(board, int(sr), int(sc))
            for tr, tc in reachable:
                yield (int(sr), int(sc), tr, tc)

    def _reachable_empties(self, board: np.ndarray, sr: int, sc: int) -> List[Tuple[int, int]]:
        q = collections.deque([(sr, sc)])
        seen = {(sr, sc)}
        targets: List[Tuple[int, int]] = []
        while q:
            r, c = q.popleft()
            for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                nr, nc = r + dr, c + dc
                if not (0 <= nr < BOARD_SIZE and 0 <= nc < BOARD_SIZE):
                    continue
                if (nr, nc) in seen:
                    continue
                seen.add((nr, nc))
                if board[nr][nc] == EMPTY:
                    targets.append((nr, nc))
                    q.append((nr, nc))
        return targets

    def _evaluate_action(self, board: np.ndarray, action: Action) -> float:
        sr, sc, tr, tc = action
        test_board = np.array(board, copy=True)
        test_board[tr][tc] = test_board[sr][sc]
        test_board[sr][sc] = EMPTY

        reward = self._update_brd(test_board, tr, tc)

        empty_cells = np.count_nonzero(test_board == EMPTY)
        center = 4
        center_dist = abs(tr - center) + abs(tc - center)
        center_score = 8 - center_dist
        future = self._max_match_length(test_board)

        return (
            reward * self.weights.immediate_reward
            + future * self.weights.future_match_len
            + center_score * self.weights.center_bias
            + empty_cells * self.weights.empty_cells
        )

    def _max_match_length(self, board: np.ndarray) -> int:
        best = 1
        for r in range(BOARD_SIZE):
            for c in range(BOARD_SIZE):
                val = board[r][c]
                if val == EMPTY:
                    continue
                for dr, dc in ((1, 0), (0, 1), (1, 1), (1, -1)):
                    cnt = 1
                    nr, nc = r + dr, c + dc
                    while 0 <= nr < BOARD_SIZE and 0 <= nc < BOARD_SIZE and board[nr][nc] == val:
                        cnt += 1
                        nr += dr
                        nc += dc
                    best = max(best, cnt)
        return best

    def _update_brd(self, board: np.ndarray, r: int, c: int) -> int:
        gain = 0
        val = board[r][c]
        for dr1, dc1, dr2, dc2 in [(-1, 0, 1, 0), (0, -1, 0, 1), (-1, -1, 1, 1), (-1, 1, 1, -1)]:
            lcnt = 0
            nr, nc = r + dr1, c + dc1
            while 0 <= nr < BOARD_SIZE and 0 <= nc < BOARD_SIZE and board[nr][nc] == val:
                nr, nc = nr + dr1, nc + dc1
                lcnt += 1
            rcnt = 0
            nr, nc = r + dr2, c + dc2
            while 0 <= nr < BOARD_SIZE and 0 <= nc < BOARD_SIZE and board[nr][nc] == val:
                nr, nc = nr + dr2, nc + dc2
                rcnt += 1
            line = lcnt + rcnt + 1
            if line >= 5:
                for i in range(1, lcnt + 1):
                    board[r + i * dr1][c + i * dc1] = EMPTY
                for i in range(1, rcnt + 1):
                    board[r + i * dr2][c + i * dc2] = EMPTY
                gain = [5, 8, 13, 20, 28, 13][line - 5]
                board[r][c] = EMPTY
        return gain


def run_episode(seed: int | None = None, max_steps: int = 1500) -> int:
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    env = bloomingGardenEnv()
    env.reset()
    planner = GreedyPlanner()

    score = 0
    done = False
    steps = 0
    while not done and steps < max_steps:
        action = planner.choose_action(env)
        (_, score), _, done, _ = env.step(action)
        steps += 1
    return int(score)


def benchmark(episodes: int, seed: int) -> Sequence[int]:
    return [run_episode(seed + i) for i in range(episodes)]


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a non-ML baseline for BloomingGarden")
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--seed", type=int, default=123)
    args = parser.parse_args()

    scores = benchmark(args.episodes, args.seed)
    print(f"episodes={args.episodes} min={min(scores)} avg={np.mean(scores):.2f} max={max(scores)}")
    print("scores=", scores)


if __name__ == "__main__":
    main()

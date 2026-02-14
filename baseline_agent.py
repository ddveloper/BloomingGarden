"""High-score focused planner for the local BloomingGarden gym environment.

The planner combines:
1) full legal move generation via BFS reachability,
2) line-clear aware static evaluation,
3) stochastic lookahead over the environment's random flower spawns.

This gives a much stronger non-neural baseline for data generation and policy bootstrapping.
"""

from __future__ import annotations

import argparse
import collections
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple

import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "gym-bloomingGarden"))

from gym_bloomingGarden.envs.bloomingGarden_env import bloomingGardenEnv

BOARD_SIZE = 9
EMPTY = 3
FLOWERS = (0, 1, 2, 4, 5, 6, 7)

Action = Tuple[int, int, int, int]


@dataclass
class PlannerConfig:
    beam_width: int = 12
    lookahead_depth: int = 2
    placement_samples: int = 5
    rollout_steps: int = 2
    immediate_reward_w: float = 250.0
    longest_chain_w: float = 22.0
    mobility_w: float = 0.7
    empty_cells_w: float = 1.1
    center_bias_w: float = 1.0


@dataclass
class Snapshot:
    board: np.ndarray
    coming: Tuple[int, int, int]
    score: int


class HighScorePlanner:
    def __init__(self, config: PlannerConfig | None = None):
        self.cfg = config or PlannerConfig()

    def choose_action(self, env: bloomingGardenEnv) -> Action:
        root = Snapshot(board=np.array(env.brd, copy=True), coming=tuple(env.coming), score=int(env.score))
        actions = self._legal_actions(root.board)
        if not actions:
            return (0, 0, 0, 1)

        candidates = sorted(
            ((self._quick_action_score(root.board, action), action) for action in actions), reverse=True
        )
        top_actions = [a for _, a in candidates[: self.cfg.beam_width]]

        best_action = top_actions[0]
        best_value = float("-inf")
        for action in top_actions:
            next_snap, reward = self._simulate_step(root, action)
            value = reward + self._value(next_snap, depth=self.cfg.lookahead_depth - 1)
            if value > best_value:
                best_value = value
                best_action = action
        return best_action

    def _value(self, snap: Snapshot, depth: int) -> float:
        if depth <= 0:
            return self._state_value(snap.board)

        actions = self._legal_actions(snap.board)
        if not actions:
            return self._state_value(snap.board)

        scored = sorted(
            ((self._quick_action_score(snap.board, action), action) for action in actions), reverse=True
        )
        top_actions = [a for _, a in scored[: self.cfg.beam_width]]

        best = float("-inf")
        for action in top_actions:
            nxt, reward = self._simulate_step(snap, action)
            value = reward + self._value(nxt, depth - 1)
            if value > best:
                best = value
        return best

    def _simulate_step(self, snap: Snapshot, action: Action) -> Tuple[Snapshot, float]:
        """Approximate one game transition with sampled random placements."""
        sr, sc, tr, tc = action
        base_board = np.array(snap.board, copy=True)
        base_board[tr][tc] = base_board[sr][sc]
        base_board[sr][sc] = EMPTY

        immediate_reward = self._update_brd(base_board, tr, tc)
        if immediate_reward > 0:
            next_board = base_board
            next_coming = tuple(np.random.choice(FLOWERS, size=3).tolist())
            return Snapshot(next_board, next_coming, snap.score + immediate_reward), float(immediate_reward)

        # If no line was cleared, sample the random insertion process to estimate expected outcome.
        sample_rewards = []
        sampled_boards: List[np.ndarray] = []
        for _ in range(self.cfg.placement_samples):
            board = np.array(base_board, copy=True)
            reward = 0
            done = False
            for flower in snap.coming:
                empties = np.argwhere(board == EMPTY)
                if len(empties) == 0:
                    done = True
                    break
                idx = np.random.randint(0, len(empties))
                nr, nc = empties[idx]
                board[nr][nc] = flower
                reward += self._update_brd(board, int(nr), int(nc))
            if done:
                sample_rewards.append(float(reward - 400))
            else:
                sample_rewards.append(float(reward))
            sampled_boards.append(board)

        k = int(np.argmax(sample_rewards))
        chosen_board = sampled_boards[k]
        expected_reward = float(np.mean(sample_rewards))
        next_coming = tuple(np.random.choice(FLOWERS, size=3).tolist())

        # short rollout to reduce variance / encourage robust tactical states
        bonus = 0.0
        temp_snap = Snapshot(np.array(chosen_board, copy=True), next_coming, snap.score + int(expected_reward))
        for _ in range(self.cfg.rollout_steps):
            actions = self._legal_actions(temp_snap.board)
            if not actions:
                break
            action2 = max(actions, key=lambda a: self._quick_action_score(temp_snap.board, a))
            temp_snap, r2 = self._simulate_step_no_rollout(temp_snap, action2)
            bonus += r2 * 0.35

        return Snapshot(chosen_board, next_coming, snap.score + int(expected_reward)), expected_reward + bonus

    def _simulate_step_no_rollout(self, snap: Snapshot, action: Action) -> Tuple[Snapshot, float]:
        sr, sc, tr, tc = action
        board = np.array(snap.board, copy=True)
        board[tr][tc] = board[sr][sc]
        board[sr][sc] = EMPTY
        reward = self._update_brd(board, tr, tc)
        if reward == 0:
            for flower in snap.coming:
                empties = np.argwhere(board == EMPTY)
                if len(empties) == 0:
                    reward -= 400
                    break
                nr, nc = empties[np.random.randint(0, len(empties))]
                board[nr][nc] = flower
                reward += self._update_brd(board, int(nr), int(nc))
        next_coming = tuple(np.random.choice(FLOWERS, size=3).tolist())
        return Snapshot(board, next_coming, snap.score + reward), float(reward)

    def _quick_action_score(self, board: np.ndarray, action: Action) -> float:
        sr, sc, tr, tc = action
        test_board = np.array(board, copy=True)
        test_board[tr][tc] = test_board[sr][sc]
        test_board[sr][sc] = EMPTY

        reward = self._update_brd(test_board, tr, tc)
        center_dist = abs(tr - 4) + abs(tc - 4)
        center_score = 8 - center_dist

        return (
            reward * self.cfg.immediate_reward_w
            + self._longest_chain(test_board) * self.cfg.longest_chain_w
            + self._mobility(test_board) * self.cfg.mobility_w
            + np.count_nonzero(test_board == EMPTY) * self.cfg.empty_cells_w
            + center_score * self.cfg.center_bias_w
        )

    def _state_value(self, board: np.ndarray) -> float:
        return (
            self._longest_chain(board) * self.cfg.longest_chain_w
            + self._mobility(board) * self.cfg.mobility_w
            + np.count_nonzero(board == EMPTY) * self.cfg.empty_cells_w
        )

    def _mobility(self, board: np.ndarray) -> int:
        flowers = np.argwhere(board != EMPTY)
        empties = np.count_nonzero(board == EMPTY)
        if empties == 0:
            return 0
        total = 0
        for sr, sc in flowers:
            total += len(self._reachable_empties(board, int(sr), int(sc)))
        return total

    def _longest_chain(self, board: np.ndarray) -> int:
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
                    if cnt > best:
                        best = cnt
        return best

    def _legal_actions(self, board: np.ndarray) -> List[Action]:
        actions: List[Action] = []
        flowers = np.argwhere(board != EMPTY)
        for sr, sc in flowers:
            for tr, tc in self._reachable_empties(board, int(sr), int(sc)):
                actions.append((int(sr), int(sc), tr, tc))
        return actions

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

    def _update_brd(self, board: np.ndarray, r: int, c: int) -> int:
        gain = 0
        val = board[r][c]
        for dr1, dc1, dr2, dc2 in [(-1, 0, 1, 0), (0, -1, 0, 1), (-1, -1, 1, 1), (-1, 1, 1, -1)]:
            lcnt = 0
            nr, nc = r + dr1, c + dc1
            while 0 <= nr < BOARD_SIZE and 0 <= nc < BOARD_SIZE and board[nr][nc] == val:
                nr += dr1
                nc += dc1
                lcnt += 1
            rcnt = 0
            nr, nc = r + dr2, c + dc2
            while 0 <= nr < BOARD_SIZE and 0 <= nc < BOARD_SIZE and board[nr][nc] == val:
                nr += dr2
                nc += dc2
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


def run_episode(seed: int | None = None, max_steps: int = 3000, config: PlannerConfig | None = None) -> int:
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    env = bloomingGardenEnv()
    env.reset()
    planner = HighScorePlanner(config)

    score = 0
    done = False
    steps = 0
    while not done and steps < max_steps:
        action = planner.choose_action(env)
        (_, score), _, done, _ = env.step(action)
        steps += 1
    return int(score)


def benchmark(episodes: int, seed: int, config: PlannerConfig | None = None) -> Sequence[int]:
    return [run_episode(seed + i, config=config) for i in range(episodes)]


def main() -> None:
    parser = argparse.ArgumentParser(description="Run high-score baseline planner for BloomingGarden")
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--beam-width", type=int, default=12)
    parser.add_argument("--depth", type=int, default=2)
    parser.add_argument("--samples", type=int, default=5)
    args = parser.parse_args()

    cfg = PlannerConfig(beam_width=args.beam_width, lookahead_depth=args.depth, placement_samples=args.samples)
    scores = benchmark(args.episodes, args.seed, cfg)
    print(f"episodes={args.episodes} min={min(scores)} avg={np.mean(scores):.2f} max={max(scores)}")
    print("scores=", scores)


if __name__ == "__main__":
    main()

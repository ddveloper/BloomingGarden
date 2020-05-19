import gym
from gym import error, spaces, utils
from gym.utils import seeding

import numpy as np
import collections

class bloomingGardenEnv(gym.Env):
  metadata = {'render.modes': ['human']}

  def __init__(self):
    self.score = 0
    self.flowers = [0, 1, 2, 4, 5, 6, 7]
    self.brd = np.full((9, 9), 3, dtype=np.int)
    self.coming = [3, 3, 3]
    self.reset()
    self.action_space = spaces.Tuple((spaces.Discrete(9), spaces.Discrete(9), spaces.Discrete(9), spaces.Discrete(9)))
    #self.observation_space = spaces.

  def step(self, action):
    reward = 0
    done = np.count_nonzero(self.brd == 3) == 0
    if done:
      return (self.brd, self.coming, self.score), reward, done, {}

    if not self._is_valid_action(action): # action is invalid
      reward = 0
      return (self.brd, self.coming, self.score), reward, done, {}

    # take the action and calculate score
    sr, sc, tr, tc = action
    self.brd[tr][tc] = self.brd[sr][sc]
    self.brd[sr][sc] = 3
    reward += self._update_brd(tr, tc)

    # fill in coming 3 flowers and update score when necessary
    if reward == 0:
      available = np.count_nonzero(self.brd == 3)
      for val in self.coming:
        if available == 0:
          done = True; break
        position = np.random.randint(0, available)
        r0, c0 = np.where(self.brd == 3)
        nr, nc = r0[position], c0[position]
        self.brd[nr][nc] = val
        reward += self._update_brd(nr, nc)
        available -= 1

    # update score
    self.score += reward

    # update comings
    self._update_coming()

    return (self.brd, self.coming, self.score), reward, done, {}

  def reset(self):
    ''' flowers [0-pink, 1-green, 2-yellow, 3-empty, 4-blue
                 5-purple, 6-white, 7-red]
    '''
    self.score = 0
    self.brd = np.full((9, 9), 3, dtype=np.int)
    for i in range(3): # init 3 flowers
      position = np.random.randint(0, 81)
      flower = np.random.choice(self.flowers)
      self.brd[position//9][position%9] = flower
    self._update_coming()

  def render(self, mode='human'):
    brd = '\n     0 1 2 3 4 5 6 7 8\n'
    brd += '     ~ ~ ~ ~ ~ ~ ~ ~ ~ \n'

    for i in range(9):
      row = f' {i} | ' + ' '.join([str(x) if x != 3 else '_' for x in self.brd[i].tolist()]) + '\n'
      brd += row
    print(brd)
    print(self.score, self.coming)

  def close(self):
    self.reset()

  def _is_valid_action(self, action):
    ''' use BFS to check if action (from src -> target)
        is reachable or not
    '''
    sr, sc, tr, tc = action
    if self.brd[sr][sc] == 3: return False
    if sr == tr and sc == tc: return False
    q = collections.deque([(sr,sc)])
    seen = set()
    while len(q):
      r, c = q.popleft()
      if r == tr and c == tc: return True
      for dr,dc in [(1,0),(-1,0),(0,1),(0,-1)]:
        nr, nc = r+dr, c+dc
        if nr >= 0 and nr < 9 and nc >=0 and nc < 9 \
          and self.brd[nr][nc] == 3 and (nr,nc) not in seen:
          q.append((nr,nc)); seen.add((nr,nc))
    return False

  def _update_coming(self):
    self.coming = []
    for i in range(3): self.coming.append(np.random.choice(self.flowers))

  def _update_brd(self, r, c):
    gain = 0
    val = self.brd[r][c]
    # horizontal update
    for dr1,dc1,dr2,dc2 in [(-1,0,1,0), (0,-1,0,1), (-1,-1,1,1),(-1,1,1,-1)]:
      lcnt = 0
      nr, nc = r+dr1, c+dc1
      while nr >= 0 and nr < 9 and nc >=0 and nc < 9 \
        and self.brd[nr][nc] == val:
        nr, nc = nr+dr1, nc+dc1; lcnt += 1
      rcnt = 0
      nr, nc = r+dr2, c+dc2
      while nr >= 0 and nr < 9 and nc >=0 and nc < 9 \
        and self.brd[nr][nc] == val:
        nr, nc = nr+dr2, nc+dc2; rcnt += 1
      if lcnt + rcnt + 1 >= 5: # can be clean
        for i in range(1, lcnt+1): self.brd[r+i*dr1][c+i*dc1] = 3
        for i in range(1, rcnt+1): self.brd[r+i*dr2][c+i*dc2] = 3
        gain = [5,8,13,20,28,13][lcnt+rcnt+1-5]
      if gain: self.brd[r][c] = 3
    return gain

  # def _get_next_pos(self, pos):
  #   for i in range(81):
  #     r, c = (pos+i)//9, (pos+i)%9
  #     if self.brd[r][c] == 3:
  #       return r, c
  #   raise error.InvalidAction(f"failed to get next pos - {r},{c}")

      
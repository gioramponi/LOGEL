"""Grid world MDP from 'LOGEL' paper.

      N(S)  N      N      R      R
---------------------------------------
      N     N      N      N      R
---------------------------------------
      N     N      V      N      N
---------------------------------------
      P     N      N      N      N
---------------------------------------
      P     P      N      N      G

where R = -3, V = -5, N = -1, G = 7, P=0

"""
from gym.envs.classic_control import rendering
import numpy as np


class Grid(object):
    """This class implements a grid MDP."""

    def __init__(self, size, reward_weights=None, stochastic=False):
        self.size = size
        self.noise = None
        if stochastic:
            self.noise = np.ones((size, size, 4, 4))
            self.noise /= self.noise.sum(3, keepdims=True)

        self.mid = int((self.size - 1) / 2)
        self.start = (0, 0)
        self.color_grid = self.create_color_grid()
        self.state = self.reset()
        self.viewer = None
        if reward_weights is None:
            self.reward_weights = np.array([0, -5,-1,-3, 7])
        else:
            self.reward_weights = reward_weights

    def reset(self):
        self.state = self.start
        return self.state

    def create_color_grid(self):
        color_grid = np.array([['N', 'N', 'N', 'R', 'R'],
                               ['N', 'N', 'N', 'N', 'R'],
                               ['N', 'N', 'V', 'N', 'N'],
                               ['P', 'N', 'N', 'N', 'N'],
                               ['P', 'P', 'N', 'N', 'G']])
        return color_grid

    def transition(self, state, action):
        """Transition p(s'|s,a)."""
        if state[0] == self.size - 1 and state[1] == self.size-1:# or state == (self.mid, self.mid):
            return self.reset()
        else:
            x, y = state
            if self.noise is not None and np.random.rand() > 0.7:
                d = np.random.choice(range(4), p=self.noise[x, y, action])
            else:
                d = action

        directions = np.array([[1, -1, 0, 0], [0, 0, -1, 1]])
        dx, dy = directions[:, d]
        x_ = max(0, min(self.size - 1, x + dx))
        y_ = max(0, min(self.size - 1, y + dy))
        return (x_, y_)

    def reward(self, state):
        """Reward depends on the color of the state"""
        if self.color_grid[state[0], state[1]] == 'R':
            return self.reward_weights[0]
        if self.color_grid[state[0], state[1]] == 'V':
            return self.reward_weights[1]
        if self.color_grid[state[0], state[1]] == 'N':
            return self.reward_weights[2]
        if self.color_grid[state[0], state[1]] == 'P':
            return self.reward_weights[3]
        if self.color_grid[state[0], state[1]] == 'G':
            return self.reward_weights[4]


    def get_reward_vector(self, state):
        feat = np.zeros(5)
        state = self._intToCouple(state)
        if self.color_grid[state[0], state[1]] == 'R':
            feat[0] = 1
            return feat
        if self.color_grid[state[0], state[1]] == 'V':
            feat[1] = 1
            return feat
        if self.color_grid[state[0], state[1]] == 'N':
            feat[2] = 1
            return feat
        if self.color_grid[state[0], state[1]] == 'P':
            feat[3] = 1
            return feat
        if self.color_grid[state[0], state[1]] == 'G':
            feat[4] = 1
            return feat
        return feat

    def step(self, state, action):
        state = self.transition(state, action)
        reward = self.reward(state)
        reward_vect = self.get_reward_vector(self._coupleToInt(state[0], state[1]))
        self.state = state
        done = False
        if self.color_grid[state[0], state[1]] == 'G':
            done = False
        return state, reward, reward_vect

    def get_reward(self, state):
        return np.dot(self.get_reward_vector(state), self.reward_weights)

    def make_tables(self):
        """Returns tabular version of reward and transition functions r and p.
    """
        r = np.zeros((self.size * self.size, 4))
        p = np.zeros((self.size * self.size, 4, self.size * self.size))
        directions = np.array([[1, -1, 0, 0], [0, 0, -1, 1]])
        for x in range(self.size):
            for y in range(self.size):
                for a in range(4):
                    i = x * self.size + y
                    r[i, a] = self.reward((x, y))
                    if (x, y) == (self.size - 1, self.size - 1) or \
                            (x, y) == (self.mid, self.mid):
                        p[i, a, 0] = 1
                    else:
                        for d in range(4):
                            dx, dy = directions[:, d]
                            x_ = max(0, min(self.size - 1, x + dx))
                            y_ = max(0, min(self.size - 1, y + dy))
                            j = x_ * self.size + y_
                            if self.noise is not None:
                                p[i, a, j] += 0.3 * self.noise[x, y, a, d] + 0.7 * int(a == d)
                            else:
                                p[i, a, j] += int(a == d)
        return r, p

    def make_tables_gpomdp(self):
        """Returns tabular version of reward and transition functions r and p.
      r = A*S
      r_f = A*S*S
      p = A*S*S
    """
        r = np.zeros((4, self.size * self.size))
        r_f = np.zeros((4, self.size * self.size, 5))#self.size * self.size))
        p = np.zeros((4, self.size * self.size, self.size * self.size))
        directions = np.array([[1, -1, 0, 0], [0, 0, -1, 1]])
        for x in range(self.size):
            for y in range(self.size):
                for a in range(4):
                    i = x * self.size + y
                    r_f[a, i] = self.get_reward_vector(i)
                    r[a, i] = np.dot(r_f[a, i], self.reward_weights)
                    if (x, y) == (self.size - 1, self.size - 1) or \
                            (x, y) == (self.mid, self.mid):
                        p[a, i, 0] = 1
                    else:
                        for d in range(4):
                            dx, dy = directions[:, d]
                            x_ = max(0, min(self.size - 1, x + dx))
                            y_ = max(0, min(self.size - 1, y + dy))
                            j = x_ * self.size + y_
                            if self.noise is not None:
                                p[a, i, j] += 0.3 * self.noise[x, y, a, d] + 0.7 * int(a == d)
                            else:
                                p[a, i, j] += int(a == d)
        return r, p, r_f

    def phi(self, s, a):
        phi = np.zeros(self.size * self.size * 3)
        phi[s * 3 + a] = 1
        return phi

    def _coupleToInt(self, x, y):
        return y + x * self.size

    def _intToCouple(self, n):
        return int(np.floor(n / self.size)), int(n % self.size)

    def _render(self, mode='human', close=False):
        if close:
            if self.viewer is not None:
                self.viewer.close()
                self.viewer = None
            return

        if self.viewer is None:
            self.viewer = rendering.Viewer(500, 500)
            self.viewer.set_bounds(0, self.size, 0, self.size)

        # Draw the grid
        for i in range(self.size):
            self.viewer.draw_line((0, i), (self.size, i))
            self.viewer.draw_line((i, 0), (i, self.size))
        goal = self.viewer.draw_circle(radius=0.5)
        goal.set_color(0, 0.8, 0)
        goal.add_attr(rendering.Transform(translation=(self.size-1 + 0.5, self.mid + 0.5)))

        agent = self.viewer.draw_circle(radius=0.4)
        agent.set_color(.8, 0, 0)
        agent_x, agent_y = self.state
        transform = rendering.Transform(translation=(agent_x + 0.5, agent_y + 0.5))
        agent.add_attr(transform)

        return self.viewer.render(return_rgb_array=mode == 'rgb_array')



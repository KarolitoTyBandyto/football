import gym
import numpy as np
from gym import spaces


class FootballGameState:
    def __init__(self, height=13, width=13):
        self.height = height
        self.width = width
        self.board = np.full((height, width), " ", dtype=str)
        self.lines = set()
        self.ball_position = None
        self.goals_position = []

    def initialize_board(self):

        goal_width = self.width // 3
        start = (self.width - goal_width) // 2
        end = start + goal_width

        self.board[0, start : end + 1] = "─"
        self.board[0, start] = "┌"
        self.board[0, end] = "┐"
        self.board[-1, start : end + 1] = "─"
        self.board[-1, start] = "└"
        self.board[-1, end] = "┘"

        for i in range(start + 1, end):
            self.goals_position.append((i, 0))
            self.goals_position.append((i, self.height - 1))

        self.ball_position = (self.width // 2, self.height // 2)
        self.board[self.ball_position[1], self.ball_position[0]] = "B"

    def add_line(self, start, end):
        self.lines.add((start, end))

    def move_ball(self, new_position):
        if self.ball_position:
            self.board[self.ball_position[1], self.ball_position[0]] = " "
        self.ball_position = new_position
        self.board[new_position[1], new_position[0]] = "B"

    def _update_board_with_lines(self):
        self.board[np.where((self.board == "─") | (self.board == "│"))] = " "

        for start, end in self.lines:
            x, y = start
            dx = np.sign(end[0] - start[0])
            dy = np.sign(end[1] - start[1])
            while (x, y) != end:
                x += dx
                y += dy
                if self.board[y, x] == " ":
                    self.board[y, x] = "─" if dx != 0 else "│"

        self.board[self.ball_position[1], self.ball_position[0]] = "B"

    def get_state_representation(self):
        state = np.zeros((self.height, self.width, 3), dtype=np.float32)


        state[self.ball_position[1], self.ball_position[0], 0] = 1


        state[:, :, 1] = np.where(
            np.isin(self.board, ["─", "│", "┌", "┐", "└", "┘"]), 1, 0
        )


        for x, y in self.goals_position:
            state[y, x, 2] = 1

        return state

    def __str__(self):
        return "\n".join(["".join(row) for row in self.board])


class FootballGameEnv(gym.Env):
    def __init__(self, height=13, width=13):
        super(FootballGameEnv, self).__init__()
        self.height = height
        self.width = width
        self.board = np.full((height, width), " ", dtype=str)

        self.action_space = spaces.Discrete(8)
        self.observation_space = spaces.Box(
            low=0, high=1, shape=(height, width, 4), dtype=np.float32
        )

        self.game_state = FootballGameState(height, width)
        self.turn = None
        self.winner = None
        self.extra_move_granted = None
        self.last_move_hit_post = None
        self.player1_goals = []
        self.player2_goals = []
        self.reset()

    def _get_action_from_positions(self, start, end):
        dx = end[0] - start[0]
        dy = end[1] - start[1]
        directions = [
            (0, -1),
            (1, -1),
            (1, 0),
            (1, 1),
            (0, 1),
            (-1, 1),
            (-1, 0),
            (-1, -1),
        ]
        return directions.index((np.sign(dx), np.sign(dy)))

    def reset(self):
        self.game_state = FootballGameState(self.height, self.width)
        self.game_state.initialize_board()
        self.player1_goals = [
            pos for pos in self.game_state.goals_position if pos[1] == self.height - 1
        ]
        self.player2_goals = [
            pos for pos in self.game_state.goals_position if pos[1] == 0
        ]
        self.turn = 1
        self.winner = None
        self.extra_move_granted = False
        self.last_move_hit_post = False

        return self._get_observation()

    def get_legal_actions(self):
        legal_actions = []
        current_position = self.game_state.ball_position
        for action in range(8):
            end = self._get_end_position(current_position, action)
            if self.is_legal_move(current_position, end):
                legal_actions.append(action)
        return legal_actions

    def step(self, action):
        legal_actions = self.get_legal_actions()
        if action not in legal_actions:
            return self._get_observation(), -1, False, {"invalid_action": True}
        start = self.game_state.ball_position
        end = self._get_end_position(start, action)

        if not self.make_move(start, end):
            return self._get_observation(), -1, False, {"invalid_move": True}

        reward = self._calculate_reward(end, self.turn)


        if not self.has_legal_moves():
            self.winner = 3 - self.turn
            reward = -10

        done = self.winner is not None

        extra_move_info = {
            "extra_move": self.extra_move_granted,
            "reason": (
                "hit_post"
                if self.last_move_hit_post
                else "connected_line" if self.extra_move_granted else None
            ),
        }

        if not self.extra_move_granted:
            self.turn = 3 - self.turn

        return self._get_observation(), reward, done, extra_move_info

    def render(self, mode="human"):
        print(str(self.game_state))
        print()

    def _get_observation(self):
        state = self.game_state.get_state_representation()
        player_layer = np.full((self.height, self.width, 1), self.turn - 1, dtype=np.float32)
        return np.concatenate([state, player_layer], axis=2)

    def _calculate_reward(self, end, player):
        base_reward = 0

        goal_result = self.check_goal(end)
        if goal_result == "correct_goal":
            base_reward += 500 if player == self.turn else -500
        elif goal_result == "own_goal":
            base_reward -= 500 if player == self.turn else 500
        elif self.winner is not None:
            base_reward -=  200 if player == self.turn else 200
        elif self.extra_move_granted:
            base_reward += 2

        base_reward += self._reward_for_goal_proximity(end, player)

        return base_reward
    def has_legal_moves(self):
        current_position = self.game_state.ball_position
        for neighbor in self.get_neighbours(*current_position):
            if self.is_legal_move(current_position, neighbor):
                return True
        return False

    def _reward_for_potential_moves(self, position):
        potential_reward = 0
        visited = set()

        def dfs(pos, depth):
            if depth == 0 or pos in visited:
                return 0

            visited.add(pos)
            local_reward = 0

            for neighbor in self.get_neighbours(*pos):
                if self.is_legal_move(pos, neighbor):
                    local_reward += 1 + dfs(neighbor, depth - 1)

            visited.remove(pos)
            return local_reward

        potential_reward = dfs(position, 3)
        return potential_reward / 10

    def _reward_for_goal_proximity(self, position, player):
        opponent_goals = self.player2_goals if player == 1 else self.player1_goals
        distances = [abs(position[1] - goal[1]) for goal in opponent_goals]
        distance_to_goal = min(distances)

        max_distance = self.height - 1
        proximity_reward = (max_distance - distance_to_goal) / max_distance

        return proximity_reward

    def _is_valid_action(self, action):
        return 0 <= action < 8

    def _get_end_position(self, start, action):
        directions = [
            (0, -1),
            (1, -1),
            (1, 0),
            (1, 1),
            (0, 1),
            (-1, 1),
            (-1, 0),
            (-1, -1),
        ]
        dx, dy = directions[action]
        return (start[0] + dx, start[1] + dy)

    def make_move(self, start, end):
        self.last_move_hit_post = False
        if not self.is_legal_move(start, end):
            return False

        goal_result = self.check_goal(end)
        if goal_result == "correct_goal":
            self.winner = self.turn
            self.extra_move_granted = False
        elif goal_result == "own_goal":
            self.winner = 3 - self.turn
            self.extra_move_granted = False
        else:
            self.extra_move_granted = self.is_extra_move_granted(start, end)

        old_position = self.game_state.ball_position
        self.game_state.move_ball(end)
        self.game_state.add_line(old_position, end)
        return True

    def get_neighbours(self, x: int, y: int):
        return [
            (x + dx, y + dy)
            for dx, dy in [
                (1, 0),
                (-1, 0),
                (0, 1),
                (0, -1),
                (1, 1),
                (-1, 1),
                (1, -1),
                (-1, -1),
            ]
            if 0 <= x + dx < self.width and 0 <= y + dy < self.height
        ]

    def is_legal_move(self, start, end):
        neighbor_squares = self.get_neighbours(*start)

        if not (
            0 <= start[0] < self.width
            and 0 <= start[1] < self.height
            and 0 <= end[0] < self.width
            and 0 <= end[1] < self.height
        ):
            return False

        if (end[0], end[1]) not in neighbor_squares:
            return False

        if (start, end) in self.game_state.lines or (
            end,
            start,
        ) in self.game_state.lines:
            return False

        x, y = start
        dx = np.sign(end[0] - start[0])
        dy = np.sign(end[1] - start[1])

        while (x, y) != end:
            x += dx
            y += dy
            if self.game_state.board[y, x] not in [
                " ",
                "B",
                "─",
                "│",
                "┌",
                "┐",
                "└",
                "┘",
            ]:
                return False

        return True

    def check_goal(self, end):
        if end in self.game_state.goals_position:
            if (self.turn == 1 and end in self.player2_goals) or (
                self.turn == 2 and end in self.player1_goals
            ):
                return "correct_goal"
            else:
                return "own_goal"
        return False

    def is_post(self, end):
        return self.game_state.board[end[1], end[0]] in ["┌", "┐", "└", "┘"]

    def is_extra_move_granted(self, start, end):

        hit_post_criteria = self.is_post(end)

        if hit_post_criteria:
            self.last_move_hit_post = True
            return True


        hit_line_criteria = any(
            (
                (end, neighbor) in self.game_state.lines
                or (neighbor, end) in self.game_state.lines
            )
            and neighbor != start
            for neighbor in self.get_neighbours(*end)
        )

        return hit_line_criteria



if __name__ == "__main__":
    env = FootballGameEnv()
    obs = env.reset()
    print("Initial state:")
    env.render()


    for i in range(5):
        action = env.action_space.sample()
        obs, reward, done, info = env.step(action)
        print(f"\nAfter action {action}:")
        env.render()
        print(f"Reward: {reward}, Done: {done}, Info: {info}")

        if done:
            break

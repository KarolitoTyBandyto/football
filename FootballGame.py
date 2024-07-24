from dataclasses import dataclass, field
from typing import List, Tuple, Set
import numpy as np


@dataclass
class FootballGame:
    pitch_height: int
    pitch_width: int
    board: np.ndarray = field(init=False)
    turn: int = 1
    winner: int = None
    lines: Set[Tuple[Tuple[int, int], Tuple[int, int]]] = field(default_factory=set)
    extra_move_granted: bool = field(default=False)
    goals_position: List[Tuple[int, int]] = field(init=False)
    last_move_hit_post: bool = field(default=False)

    def __post_init__(self):
        self.board = np.full((self.pitch_height, self.pitch_width), " ", dtype=str)
        self.goals_position = []
        self._add_goals()
        self.place_on_board(self.pitch_width // 2, self.pitch_height // 2, "B")

    def _add_goals(self):
        goal_width = self.pitch_width // 3
        start = (self.pitch_width - goal_width) // 2
        end = start + goal_width

        # Add top goal
        self.board[0, start : end + 1] = "─"
        self.board[0, start] = "┌"
        self.board[0, end] = "┐"

        # Add bottom goal
        self.board[-1, start : end + 1] = "─"
        self.board[-1, start] = "└"
        self.board[-1, end] = "┘"
        for i in range(start + 1, end):

            self.goals_position.append((i, 0))
            self.goals_position.append((i, self.pitch_height - 1))
        print(self.goals_position)

    def place_on_board(self, x: int, y: int, symbol: str):
        if 0 <= x < self.pitch_width and 0 <= y < self.pitch_height:
            self.board[y, x] = symbol
        else:
            raise ValueError(f"Invalid position: ({x}, {y})")

    def make_move(self, start: Tuple[int, int], end: Tuple[int, int]) -> bool:
        self.last_move_hit_post = False
        if not self.is_legal_move(start, end):
            return False

        piece = self.board[start[1], start[0]]
        self.board[start[1], start[0]] = " "
        self.board[end[1], end[0]] = piece

        self.lines.add((start, end))

        x, y = start
        dx = np.sign(end[0] - start[0])
        dy = np.sign(end[1] - start[1])

        while (x, y) != end:
            x += dx
            y += dy
            if self.board[y, x] == " ":
                self.board[y, x] = "─" if dx != 0 else "│"
        print("winner")
        print(self.winner)
        print(self.get_neighbours(*start))
        print(end)
        if self.check_goal(end) and end in self.get_neighbours(*start):
            self.winner = self.turn

        return True

    def get_neighbours(self, x: int, y: int) -> List[Tuple[int, int]]:
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
            if 0 <= x + dx < self.pitch_width and 0 <= y + dy < self.pitch_height
        ]

    def is_legal_move(self, start: Tuple[int, int], end: Tuple[int, int]) -> bool:
        print("legal moves")
        print(self.board)
        neighbor_squares = self.get_neighbours(*start)

        if not (
            0 <= start[0] < self.pitch_width
            and 0 <= start[1] < self.pitch_height
            and 0 <= end[0] < self.pitch_width
            and 0 <= end[1] < self.pitch_height
        ):
            return False

        # Check if the end square is a neighbor of the start square or in a goal
        print(neighbor_squares)
        if (end[0], end[1]) not in neighbor_squares:
            return False
        print("hey")
        # Check if the move has already been made
        if (start, end) in self.lines or (end, start) in self.lines:
            return False

        x, y = start
        dx = np.sign(end[0] - start[0])
        dy = np.sign(end[1] - start[1])

        while (x, y) != end:
            x += dx
            y += dy
            if self.board[y, x] not in [" ", "B", "─", "│", "┌", "┐", "└", "┘"]:
                return False

        extra_move = self.is_extra_move_granted(end=end)

        print(f"extra move granted: {extra_move}")
        self.extra_move_granted = extra_move

        return True

    def check_goal(self, end: Tuple[int, int]) -> bool:
        if end in self.goals_position and not self.last_move_hit_post:
            return True
        return False

    def is_post(self, end: Tuple[int, int]) -> bool:
        return self.board[end[1], end[0]] in ["┌", "┐", "└", "┘"]

    def is_extra_move_granted(self,end):
        hit_line_criteria = any(
            (end, neighbor) in self.lines or (neighbor, end) in self.lines
            for neighbor in self.get_neighbours(*end)
        )
        hit_post_criteria = self.is_post(end)
        if hit_post_criteria:
            self.last_move_hit_post = True
        if hit_post_criteria or hit_line_criteria:
            return True

        return False

    def get_legal_moves(self, start: Tuple[int, int]) -> List[Tuple[int, int]]:
        legal_moves = []
        for dx, dy in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
            x, y = start[0] + dx, start[1] + dy
            while 0 <= x < self.pitch_width and 0 <= y < self.pitch_height:
                if self.is_legal_move(start, (x, y)):
                    legal_moves.append((x, y))
                if self.board[y, x] != " ":
                    break
                x += dx
                y += dy
        return legal_moves

    def get_lines(self) -> List[dict]:
        return [
            {
                "from": {"row": start[1], "col": start[0]},
                "to": {"row": end[1], "col": end[0]},
            }
            for start, end in self.lines
        ]

    def print_board(self):
        for row in self.board:
            print("│", end="")
            for cell in row:
                if cell == " ":
                    print("·", end=" ")
                elif cell == "B":
                    print("●", end=" ")
                elif cell == "1":
                    print("△", end=" ")
                elif cell == "2":
                    print("▲", end=" ")
                else:
                    print(cell, end=" ")
            print("│")
        print()


def main():
    game = FootballGame(9, 13)
    game.print_board()


if __name__ == "__main__":
    main()

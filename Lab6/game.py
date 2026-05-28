import numpy as np
from collections import deque

class GomokuGame:
    """
    Класс, управляющий логикой игры Гомоку (5 в ряд) на доске 15x15.
    """
    def __init__(self, board_size=15, win_length=5):
        self.board_size = board_size
        self.win_length = win_length
        self.board = None
        self.current_player = None
        self.game_over = False
        self.winner = None
        self.reset()

    def reset(self):
        """Сбрасывает состояние игры для новой партии."""
        self.board = np.zeros((self.board_size, self.board_size), dtype=int)
        self.current_player = 1  # 1 - первый игрок (крестики/черные), -1 - второй (нолики/белые)
        self.game_over = False
        self.winner = None
        return self._get_state()

    def _get_state(self):
        """
        Возвращает текущее состояние игры в виде, понятном для RL-агента.
        """
        return {
            'board': self.board.copy(),
            'current_player': self.current_player,
            'game_over': self.game_over,
            'winner': self.winner
        }

    def get_valid_moves(self):
        """
        Возвращает список всех допустимых ходов (клеток, где значение равно 0).
        """
        return list(zip(*np.where(self.board == 0)))

    def make_move(self, row, col):
        """
        Выполняет ход, если он допустим, и обновляет состояние игры.
        Возвращает: (новая_доска, награда, игра_закончена, информация)
        """
        if not (0 <= row < self.board_size and 0 <= col < self.board_size):
            return self._get_state(), -10, True, {'error': 'invalid_move', 'winner': self.current_player * -1}

        if self.board[row, col] != 0:
            return self._get_state(), -10, True, {'error': 'invalid_move', 'winner': self.current_player * -1}

        # Выполняем ход
        self.board[row, col] = self.current_player

        # Проверяем, не привел ли ход к победе
        if self._check_win(row, col):
            self.game_over = True
            self.winner = self.current_player
            # Победа приносит +1 награду
            return self._get_state(), 1, True, {'winner': self.winner}

        # Проверяем, не закончилась ли игра ничьей (доска заполнена)
        if len(self.get_valid_moves()) == 0:
            self.game_over = True
            self.winner = 0
            # Ничья не дает награды
            return self._get_state(), 0, True, {'winner': 0}

        # Смена игрока
        self.current_player *= -1
        return self._get_state(), 0, False, {}

    def _check_win(self, row, col):
        """Проверяет, есть ли 5 в ряд вокруг последнего поставленного камня."""
        player = self.board[row, col]
        if player == 0:
            return False

        # Направления для проверки: горизонталь, вертикаль, две диагонали
        directions = [(0, 1), (1, 0), (1, 1), (1, -1)]

        for dr, dc in directions:
            count = 1
            # Проверка в положительном направлении
            for step in range(1, self.win_length):
                r, c = row + dr * step, col + dc * step
                if not (0 <= r < self.board_size and 0 <= c < self.board_size):
                    break
                if self.board[r, c] == player:
                    count += 1
                else:
                    break

            # Проверка в отрицательном направлении
            for step in range(1, self.win_length):
                r, c = row - dr * step, col - dc * step
                if not (0 <= r < self.board_size and 0 <= c < self.board_size):
                    break
                if self.board[r, c] == player:
                    count += 1
                else:
                    break

            if count >= self.win_length:
                return True
        return False

    def render(self):
        """Выводит доску в консоль для визуального контроля (полезно при отладке)."""
        symbols = {0: '.', 1: 'X', -1: 'O'}
        print('  ' + ' '.join(str(i % 10) for i in range(self.board_size)))
        for r in range(self.board_size):
            print(str(r % 10) + ' ' + ' '.join(symbols[self.board[r, c]] for c in range(self.board_size)))
        print("\n")
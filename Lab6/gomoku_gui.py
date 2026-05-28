import tkinter as tk
from tkinter import messagebox, ttk
import numpy as np
from game import GomokuGame


class HeuristicGomokuAgent:
    def __init__(self, board_size=15, win_length=5):
        self.board_size = board_size
        self.win_length = win_length
        self.pattern_weights = {
            2: 10,
            3: 100,
            4: 10000,
            5: 1000000
        }

    def choose_action(self, state, valid_moves, player):
        best_move = None
        best_score = -1
        for r, c in valid_moves:
            attack = self._evaluate_move(state['board'], r, c, player)
            defense = self._evaluate_move(state['board'], r, c, -player)
            score = attack + defense * 0.8
            if score > best_score:
                best_score = score
                best_move = (r, c)
        return best_move

    def _evaluate_move(self, board, row, col, player):
        if board[row, col] != 0:
            return 0
        total = 0
        directions = [(1, 0), (0, 1), (1, 1), (1, -1)]
        for dr, dc in directions:
            length = 1
            step = 1
            while True:
                nr, nc = row + dr * step, col + dc * step
                if 0 <= nr < self.board_size and 0 <= nc < self.board_size and board[nr, nc] == player:
                    length += 1
                    step += 1
                else:
                    break
            step = 1
            while True:
                nr, nc = row - dr * step, col - dc * step
                if 0 <= nr < self.board_size and 0 <= nc < self.board_size and board[nr, nc] == player:
                    length += 1
                    step += 1
                else:
                    break
            if length >= 2:
                l = min(length, 5)
                total += self.pattern_weights[l]
        return total


class GomokuGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("5 в ряд — эвристический ИИ")
        self.root.resizable(False, False)
        self.root.configure(bg='#2c3e50')

        self.game = GomokuGame(board_size=15, win_length=5)

        self.agent_type = tk.StringVar(value="heuristic")
        frame_top = tk.Frame(root, bg='#2c3e50')
        frame_top.pack(pady=10)

        tk.Label(frame_top, text="Тип ИИ:", font=('Segoe UI', 11, 'bold'), fg='white', bg='#2c3e50').pack(side=tk.LEFT, padx=5)

        style = ttk.Style()
        style.theme_use('clam')
        style.configure('TRadiobutton', background='#2c3e50', foreground='white', font=('Segoe UI', 10))
        style.map('TRadiobutton', background=[('active', '#34495e')])

        rb_heuristic = ttk.Radiobutton(frame_top, text="Эвристический (умный)", variable=self.agent_type,
                                       value="heuristic", command=self.change_agent)
        rb_heuristic.pack(side=tk.LEFT, padx=5)

        rb_qlearning = ttk.Radiobutton(frame_top, text="Q-learning (глупый)", variable=self.agent_type,
                                       value="qlearning", command=self.change_agent)
        rb_qlearning.pack(side=tk.LEFT, padx=5)

        self.agent = HeuristicGomokuAgent()
        self.agent_ql = self._load_qlearning_agent()

        self.cell_size = 38
        self.board_size = self.game.board_size
        self.margin = 25
        self.width = self.margin * 2 + self.cell_size * self.board_size
        self.height = self.width

        self.canvas = tk.Canvas(root, width=self.width, height=self.height, bg='#f7f3e8', highlightthickness=0)
        self.canvas.pack(pady=10, padx=10)

        self.info_frame = tk.Frame(root, bg='#2c3e50')
        self.info_frame.pack(pady=5)

        self.info_label = tk.Label(self.info_frame, text="Ваш ход (X)", font=('Segoe UI', 14, 'bold'),
                                   fg='#ecf0f1', bg='#2c3e50')
        self.info_label.pack()

        self.reset_btn = tk.Button(root, text="Новая игра", font=('Segoe UI', 11, 'bold'),
                                   bg='#e67e22', fg='white', activebackground='#d35400',
                                   activeforeground='white', relief=tk.FLAT, padx=20, pady=5,
                                   command=self.reset_game)
        self.reset_btn.pack(pady=10)

        self.canvas.bind("<Button-1>", self.on_click)
        self.draw_board()

    def _load_qlearning_agent(self):
        try:
            from strong_agent import StrongQLearningAgent
            agent = StrongQLearningAgent(board_size=15, win_length=5, epsilon=0)
            agent.load("strong_weights_final.pkl")
            print("Q-learning агент загружен (может играть плохо).")
            return agent
        except Exception as e:
            print(f"Не удалось загрузить Q-learning агента: {e}")
            return None

    def change_agent(self):
        if self.agent_type.get() == "heuristic":
            self.agent = HeuristicGomokuAgent()
            self.info_label.config(text="ИИ: эвристический (умный). Ваш ход (X)")
        else:
            if self.agent_ql is not None:
                self.agent = self.agent_ql
                self.info_label.config(text="ИИ: Q-learning (глупый). Ваш ход (X)")
            else:
                messagebox.showerror("Ошибка", "Файл весов Q-learning не найден.\nОставлен эвристический ИИ.")
                self.agent_type.set("heuristic")
                self.agent = HeuristicGomokuAgent()
        self.reset_game()

    def draw_board(self):
        self.canvas.delete("all")
        for i in range(self.board_size + 1):
            x = self.margin + i * self.cell_size
            self.canvas.create_line(x, self.margin, x, self.height - self.margin, fill='#8b5a2b', width=1)
            y = self.margin + i * self.cell_size
            self.canvas.create_line(self.margin, y, self.width - self.margin, y, fill='#8b5a2b', width=1)

        for row in range(self.board_size):
            for col in range(self.board_size):
                val = self.game.board[row, col]
                if val == 0:
                    continue
                x = self.margin + col * self.cell_size + self.cell_size // 2
                y = self.margin + row * self.cell_size + self.cell_size // 2
                if val == 1:
                    fill_color = '#1e1e1e'
                    outline = '#333333'
                    shadow_offset = 2
                else:
                    fill_color = '#f9f9f9'
                    outline = '#aaaaaa'
                    shadow_offset = 2
                r = self.cell_size // 2 - 5
                self.canvas.create_oval(x - r + shadow_offset, y - r + shadow_offset,
                                        x + r + shadow_offset, y + r + shadow_offset,
                                        fill='#cccccc', outline='')
                self.canvas.create_oval(x - r, y - r, x + r, y + r,
                                        fill=fill_color, outline=outline, width=1)

    def on_click(self, event):
        if self.game.game_over:
            return
        col = (event.x - self.margin) // self.cell_size
        row = (event.y - self.margin) // self.cell_size
        if 0 <= row < self.board_size and 0 <= col < self.board_size:
            if self.game.board[row, col] == 0 and self.game.current_player == 1:
                self.make_human_move(row, col)

    def make_human_move(self, row, col):
        _, _, done, info = self.game.make_move(row, col)
        self.draw_board()
        if done:
            self.game_over(info.get('winner'))
            return
        self.root.after(10, self.make_ai_move)

    def make_ai_move(self):
        if self.game.game_over:
            return
        valid_moves = self.game.get_valid_moves()
        state = self.game._get_state()
        action = self.agent.choose_action(state, valid_moves, self.game.current_player)
        if action:
            _, _, done, info = self.game.make_move(action[0], action[1])
            self.draw_board()
            if done:
                self.game_over(info.get('winner'))

    def game_over(self, winner):
        if winner == 1:
            msg = "Вы выиграли!"
        elif winner == -1:
            msg = "ИИ выиграл!"
        else:
            msg = "Ничья!"
        self.info_label.config(text=msg)
        messagebox.showinfo("Игра окончена", msg)

    def reset_game(self):
        self.game.reset()
        if self.agent_type.get() == "heuristic":
            self.info_label.config(text="ИИ: эвристический (умный). Ваш ход (X)")
        else:
            self.info_label.config(text="ИИ: Q-learning (глупый). Ваш ход (X)")
        self.draw_board()


if __name__ == "__main__":
    root = tk.Tk()
    app = GomokuGUI(root)
    root.mainloop()
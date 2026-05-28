import numpy as np
import pickle
import random
from game import GomokuGame

class StrongQLearningAgent:
    def __init__(self, board_size=15, win_length=5, epsilon=0.2, alpha=0.001, gamma=0.9):
        self.board_size = board_size
        self.win_length = win_length
        self.epsilon = epsilon
        self.alpha = alpha
        self.gamma = gamma
        # 7 признаков: свои длины 2,3,4 + чужие длины 2,3,4 + центральность
        self.weights = np.random.randn(7) * 0.01
        self.prev_features = None

    def _count_lines_global(self, board, player):
        """Возвращает количество линий длины 2,3,4 для игрока на всей доске."""
        counts = [0.0, 0.0, 0.0]  # 2,3,4
        directions = [(1,0), (0,1), (1,1), (1,-1)]
        for r in range(self.board_size):
            for c in range(self.board_size):
                if board[r, c] != player:
                    continue
                for dr, dc in directions:
                    length = 1
                    # считаем в положительном направлении
                    for step in range(1, self.win_length):
                        nr, nc = r + dr*step, c + dc*step
                        if 0 <= nr < self.board_size and 0 <= nc < self.board_size and board[nr, nc] == player:
                            length += 1
                        else:
                            break
                    # если длина >=2, добавляем вклады
                    for l in [2,3,4]:
                        if length >= l:
                            counts[l-2] += 1
        # Нормализация (максимум – очень грубо, но чтобы признаки не росли бесконечно)
        max_count = self.board_size * self.board_size * 4
        counts = [c / max_count for c in counts]
        return counts

    def extract_features(self, state, action, player):
        """Признаки после гипотетического хода action."""
        board = state['board'].copy()
        r, c = action
        board[r, c] = player
        my_counts = self._count_lines_global(board, player)
        opp_counts = self._count_lines_global(board, -player)
        feats = my_counts + opp_counts
        # Центральность
        center = self.board_size // 2
        dist = abs(r - center) + abs(c - center)
        centrality = 1.0 - dist / (self.board_size * 2)
        feats.append(centrality)
        return np.array(feats, dtype=np.float32)

    def predict_q(self, state, action, player):
        return np.dot(self.weights, self.extract_features(state, action, player))

    def choose_action(self, state, valid_moves, player):
        if random.random() < self.epsilon:
            return random.choice(valid_moves)
        best = None
        best_q = -np.inf
        for a in valid_moves:
            q = self.predict_q(state, a, player)
            if q > best_q:
                best_q = q
                best = a
        return best if best else random.choice(valid_moves)

    def learn(self, reward, done, next_state, next_player):
        if self.prev_features is None:
            return
        if not done:
            next_moves = list(zip(*np.where(next_state['board'] == 0)))
            if next_moves:
                max_q = max(self.predict_q(next_state, a, next_player) for a in next_moves)
            else:
                max_q = 0.0
            target = reward + self.gamma * max_q
        else:
            target = reward
        q_prev = np.dot(self.weights, self.prev_features)
        error = np.clip(target - q_prev, -1.0, 1.0)   # ограничиваем ошибку
        self.weights += self.alpha * error * self.prev_features
        self.weights = np.clip(self.weights, -1.0, 1.0)  # ограничиваем веса

    def set_prev(self, state, action, player):
        self.prev_features = self.extract_features(state, action, player)

    def reset(self):
        self.prev_features = None

    def save(self, filename):
        with open(filename, 'wb') as f:
            pickle.dump(self.weights, f)
        print(f"Saved {filename}")

    def load(self, filename):
        try:
            with open(filename, 'rb') as f:
                self.weights = pickle.load(f)
            print(f"Loaded {filename}, weights sum: {self.weights.sum():.4f}")
        except Exception as e:
            print(f"Load failed: {e}, using random weights.")

def train_strong(episodes=3000, save_freq=500):
    env = GomokuGame(15, 5)
    agent = StrongQLearningAgent(alpha=0.001)
    wins_p1, wins_p2, draws = 0, 0, 0
    print("Training strong agent... (this will take hours)", flush=True)
    for ep in range(1, episodes+1):
        state = env.reset()
        done = False
        agent.reset()
        while not done:
            player = env.current_player
            moves = env.get_valid_moves()
            action = agent.choose_action(state, moves, player)
            if action:
                agent.set_prev(state, action, player)
                next_state, reward, done, info = env.make_move(action[0], action[1])
                agent.learn(reward, done, next_state, -player)
                state = next_state
        w = info.get('winner')
        if w == 1: wins_p1 += 1
        elif w == -1: wins_p2 += 1
        elif w == 0: draws += 1
        if ep % save_freq == 0:
            wr1 = wins_p1/ep*100
            wr2 = wins_p2/ep*100
            print(f"Ep {ep}/{episodes} | P1:{wins_p1}({wr1:.1f}%) | P2:{wins_p2}({wr2:.1f}%) | Draws:{draws}", flush=True)
            agent.save(f"strong_weights_ep{ep}.pkl")
    agent.save("strong_weights_final.pkl")
    print(f"Done. Final: P1={wins_p1/episodes*100:.1f}%, P2={wins_p2/episodes*100:.1f}%, Draws={draws}")

def play_strong(weights_file="strong_weights_final.pkl"):
    env = GomokuGame(15, 5)
    agent = StrongQLearningAgent(epsilon=0)
    agent.load(weights_file)
    state = env.reset()
    done = False
    print("You are X, AI is O. Enter 'row col'")
    env.render()
    while not done:
        if env.current_player == 1:
            try:
                r, c = map(int, input("Your move: ").split())
                if (r, c) not in env.get_valid_moves():
                    print("Invalid move")
                    continue
                action = (r, c)
            except:
                print("Invalid input")
                continue
        else:
            valid = env.get_valid_moves()
            action = agent.choose_action(state, valid, env.current_player)
            print(f"AI move: {action}")
        next_state, _, done, info = env.make_move(action[0], action[1])
        state = next_state
        env.render()
        if done:
            w = info.get('winner')
            if w == 1:
                print("You win!")
            elif w == -1:
                print("AI wins!")
            else:
                print("Draw")

if __name__ == "__main__":
    train_strong(episodes=3000, save_freq=500)
import numpy as np

class KMeansRLOptimizer:
    def __init__(self, k_range=(2, 8), lr=0.1, gamma=0.9, epsilon=0.2):
        self.k_values = list(range(k_range[0], k_range[1] + 1))
        self.q_table = {k: 0.0 for k in self.k_values}
        self.lr = lr
        self.gamma = gamma
        self.epsilon = epsilon

    def select_k(self):
        if np.random.rand() < self.epsilon:
            return np.random.choice(self.k_values)
        return max(self.q_table, key=self.q_table.get)

    def update(self, k, reward):
        best_future = max(self.q_table.values())
        self.q_table[k] += self.lr * (reward + self.gamma * best_future - self.q_table[k])


def optimize_k_rl(X_scaled, evaluate_fn, episodes=10):
    agent = KMeansRLOptimizer()

    for _ in range(episodes):
        k = agent.select_k()
        reward = evaluate_fn(k)  # silhouette score
        agent.update(k, reward)

    best_k = max(agent.q_table, key=agent.q_table.get)
    print(f"🤖 RL Selected K: {best_k}")
    return best_k
import numpy as np

class MatrixGameSolver:
    def __init__(self, payoff_matrix, random_seed=42, max_iter=10000, epsilon=1e-6):
        self.A = np.array(payoff_matrix, dtype=float)
        self.m, self.n = self.A.shape
        np.random.seed(random_seed)
        self.max_iter = max_iter
        self.epsilon = epsilon
        self.value = None
        self.strategy1 = None
        self.strategy2 = None
        self.history = []
        
    def solve(self):
        freq1 = np.zeros(self.m)
        freq2 = np.zeros(self.n)
        
        p = np.ones(self.m) / self.m
        q = np.ones(self.n) / self.n
        
        self.history = []
        
        for t in range(1, self.max_iter + 1):
            expected_payoff1 = self.A @ q
            row_choice = np.argmax(expected_payoff1)
            
            expected_payoff2 = p @ self.A
            col_choice = np.argmin(expected_payoff2)
            
            freq1[row_choice] += 1
            freq2[col_choice] += 1
            
            p = freq1 / t
            q = freq2 / t
            
            v_lower = np.min(p @ self.A)
            v_upper = np.max(self.A @ q)
            v_curr = (v_lower + v_upper) / 2
            
            self.history.append({
                'iteration': t,
                'v_lower': v_lower,
                'v_upper': v_upper,
                'v_curr': v_curr,
                'gap': v_upper - v_lower
            })
            
            if t > 100 and (v_upper - v_lower) < self.epsilon:
                print(f"Сошлось за {t} итераций (точность: {v_upper - v_lower:.2e})")
                break
        
        self.strategy1 = p
        self.strategy2 = q
        self.value = self.history[-1]['v_curr'] if self.history else 0
        
        return self.strategy1, self.strategy2, self.value
    
    def check_solution(self):
        if self.strategy1 is None:
            raise ValueError("Сначала вызовите solve()")
        
        payoff = self.strategy1 @ self.A @ self.strategy2
        min_guarantee = np.min(self.strategy1 @ self.A)
        max_loss = np.max(self.A @ self.strategy2)
        
        print(f"Цена игры (теоретическая) = {self.value:.6f}")
        print(f"Фактический выигрыш = {payoff:.6f}")
        print(f"Нижняя цена (maxmin) = {min_guarantee:.6f}")
        print(f"Верхняя цена (minmax) = {max_loss:.6f}")
        print(f"Зазор = {max_loss - min_guarantee:.2e}")
        
        return abs(payoff - self.value) < 1e-5
    
    def get_convergence_history(self):
        iterations = [h['iteration'] for h in self.history]
        v_lower = [h['v_lower'] for h in self.history]
        v_upper = [h['v_upper'] for h in self.history]
        return iterations, v_lower, v_upper
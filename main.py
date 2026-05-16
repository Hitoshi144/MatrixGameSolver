import numpy as np
from solver import MatrixGameSolver
from test import get_test_games, reference_solution_scipy
from plots import plot_heatmap, plot_strategy_comparison, plot_convergence

def run_experiment(game_dict):
    print("\n" + "#"*60)
    print(f"Игра: {game_dict['name']}")
    print(f"Комментарий: {game_dict['comment']}")
    print("Матрица выигрышей:")
    print(game_dict['matrix'])
    
    A = game_dict['matrix']
    
    plot_heatmap(A, title=f"Платёжная матрица: {game_dict['name']}")
    
    print("\nРешаем методом Брауна-Робинсона...")
    solver = MatrixGameSolver(A, random_seed=42, max_iter=5000, epsilon=1e-6)
    p, q, value = solver.solve()
    
    print(f"\n### Авторское решение (Браун-Робинсон) ###")
    print(f"Цена игры v = {value:.6f}")
    print(f"Стратегия игрока 1: {np.round(p, 4)}")
    print(f"Стратегия игрока 2: {np.round(q, 4)}")
    
    solver.check_solution()
    
    iterations, v_lower, v_upper = solver.get_convergence_history()
    plot_convergence(iterations, v_lower, v_upper, 
                     true_value=game_dict.get('expected_value'), 
                     title=f"Сходимость: {game_dict['name']}")
    
    p_ref, value_ref = reference_solution_scipy(A)
    if p_ref is not None:
        print(f"\n### Эталонное решение (scipy.optimize.linprog) ###")
        print(f"v_ref = {value_ref:.6f}")
        print(f"p_ref = {np.round(p_ref, 4)}")
        print(f"Абсолютная ошибка: {abs(value - value_ref):.2e}")
        
        plot_strategy_comparison({
            'Метод Брауна-Робинсона': p,
            'Scipy (симплекс)': p_ref
        }, game_dict['name'])
    
    return {'value': value, 'p': p, 'q': q}

def analyze_parameter_influence():
    A = np.array([[1, -1], [-1, 1]])
    max_iter_values = [100, 500, 1000, 5000]
    results = []
    
    print("\n" + "#"*60)
    print("Исследование влияния числа итераций на точность")
    
    for max_iter in max_iter_values:
        solver = MatrixGameSolver(A, random_seed=42, max_iter=max_iter, epsilon=1e-10)
        _, _, v = solver.solve()
        gap = solver.history[-1]['gap'] if solver.history else 1.0
        results.append((max_iter, v, gap))
        print(f"max_iter={max_iter:5d} -> v={v:.6f}, зазор={gap:.2e}")
    
    print("\nВывод: с ростом числа итераций точность повышается.")

def main():
    games = get_test_games()
    results = []
    for game in games:
        res = run_experiment(game)
        results.append(res)
    
    analyze_parameter_influence()
    
    print("\n" + "#"*60)
    print("ИТОГОВАЯ СВОДКА ПО ВСЕМ ИГРАМ")
    for i, game in enumerate(games):
        print(f"{i+1}. {game['name']}: v = {results[i]['value']:.6f}")
    
    print("\nПроект успешно завершён.")

if __name__ == "__main__":
    main()
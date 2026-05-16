import numpy as np
from scipy.optimize import linprog

def get_test_games():
    games = []
    
    A1 = np.array([[4, 2],
                   [3, 1]])
    games.append({
        'name': 'Седловая точка',
        'matrix': A1,
        'expected_value': 2.0,
        'comment': 'Чистые стратегии'
    })
    
    A2 = np.array([[1, -1],
                   [-1, 1]])
    games.append({
        'name': 'Орлянка',
        'matrix': A2,
        'expected_value': 0.0,
        'comment': 'Смешанные стратегии (0.5,0.5)'
    })
    
    A3 = np.array([[3, -1, -3],
                   [-2, 4, -1],
                   [-5, -6, 2]])
    games.append({
        'name': 'Учебный пример 3x3',
        'matrix': A3,
        'expected_value': None,
        'comment': 'Нет седловой точки'
    })
    
    A4 = np.array([[2, 2],
                   [3, 1]])
    games.append({
        'name': 'Вырожденная',
        'matrix': A4,
        'expected_value': 2.0,
        'comment': 'Множество оптимальных стратегий'
    })
    
    return games

def reference_solution_scipy(A):
    m, n = A.shape
    
    c = np.ones(m)
    A_ub = -A.T
    b_ub = -np.ones(n)
    bounds = [(0, None)] * m
    
    res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
    if res.success:
        value = 1.0 / res.fun
        strategy = res.x * value
        return strategy, value
    else:
        return None, None

def get_expected_results():
    games = get_test_games()
    results = []
    
    for game in games:
        A = game['matrix']
        if game['expected_value'] is None:
            _, val = reference_solution_scipy(A)
            game['expected_value'] = val
        results.append(game)
    
    return results
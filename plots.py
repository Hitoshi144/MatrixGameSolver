import matplotlib.pyplot as plt
import numpy as np

def plot_convergence(iterations, v_lower, v_upper, true_value=None, title="Сходимость метода Брауна-Робинсона"):
    plt.figure(figsize=(12, 6))
    
    plt.plot(iterations, v_lower, 'b-', linewidth=2, label='Нижняя оценка (maxmin)')
    plt.plot(iterations, v_upper, 'r-', linewidth=2, label='Верхняя оценка (minmax)')
    
    if true_value is not None:
        plt.axhline(y=true_value, color='g', linestyle='--', 
                    label=f'Истинная цена игры = {true_value:.4f}')
    
    plt.xlabel('Номер итерации')
    plt.ylabel('Оценка цены игры')
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    if np.array(v_lower[-100:]).min() > 0:
        plt.yscale('log')
    
    plt.show()

def plot_strategy_comparison(strategies_dict, game_name):
    n_methods = len(strategies_dict)
    fig, axes = plt.subplots(1, n_methods, figsize=(5 * n_methods, 4))
    
    if n_methods == 1:
        axes = [axes]
    
    for idx, (name, strat) in enumerate(strategies_dict.items()):
        axes[idx].bar(range(len(strat)), strat, alpha=0.7, color='steelblue')
        axes[idx].set_title(name)
        axes[idx].set_xlabel('Индекс стратегии')
        axes[idx].set_ylabel('Вероятность')
        axes[idx].set_ylim([0, 1])
        axes[idx].grid(True, alpha=0.3)
    
    plt.suptitle(f'Сравнение оптимальных стратегий: {game_name}')
    plt.tight_layout()
    plt.show()

def plot_heatmap(A, title="Платёжная матрица"):
    plt.figure(figsize=(8, 6))
    im = plt.imshow(A, cmap='coolwarm', interpolation='nearest')
    plt.colorbar(im, label='Выигрыш игрока 1')
    plt.title(title)
    plt.xlabel('Стратегии игрока 2')
    plt.ylabel('Стратегии игрока 1')
    
    for i in range(A.shape[0]):
        for j in range(A.shape[1]):
            plt.text(j, i, f'{A[i,j]:.1f}', 
                     ha='center', va='center', 
                     color='white' if abs(A[i,j]) > 2 else 'black')
    plt.show()
import numpy as np
import matplotlib.pyplot as plt


# Исходная функция и её производные
def f(x):
    return 4 ** x - 5 * x - 2


def df(x):
    return 4 ** x * np.log(4) - 5


# Функция phi(x) для метода простой итерации
def phi(x):
    return np.log(5 * x + 2) / np.log(4)


# Параметры задачи
eps = 1e-3
x_true = 1.71383  # Точное значение корня для расчета реальной погрешности


# 1. МЕТОД НЬЮТОНА
def newton_method(x0, eps):
    history = []
    x = x0
    k = 0
    while True:
        fx = f(x)
        dfx = df(x)
        dx = -fx / dfx
        error = abs(x - x_true)
        history.append((k, x, error))

        if abs(dx) < eps:
            break
        x = x + dx
        k += 1
    return history


# 2. МЕТОД ПРОСТОЙ ИТЕРАЦИИ
def simple_iteration(x0, eps):
    history = []
    x = x0
    k = 0
    q = 0.52  # Константа сжатия
    while True:
        x_next = phi(x)
        error = abs(x_next - x_true)
        history.append((k, x, error))

        # Условие остановки с учетом константы сжатия q
        if (q / (1 - q)) * abs(x_next - x) < eps:
            history.append((k + 1, x_next, abs(x_next - x_true)))
            break
        x = x_next
        k += 1
    return history


# Запуск методов
history_newton = newton_method(x0=2.0, eps=eps)
history_si = simple_iteration(x0=1.5, eps=eps)

# Вывод результатов в консоль
print("--- МЕТОД НЬЮТОНА ---")
print(f"{'k':<5}{'x^(k)':<12}{'Погрешность':<12}")
for k, x, err in history_newton:
    print(f"{k:<5}{x:<12.5f}{err:<12.5f}")

print("\n--- МЕТОД ПРОСТОЙ ИТЕРАЦИИ ---")
print(f"{'k':<5}{'x^(k)':<12}{'Погрешность':<12}")
for k, x, err in history_si:
    print(f"{k:<5}{x:<12.5f}{err:<12.5f}")

# --- ПОСТРОЕНИЕ ГРАФИКОВ ---
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# График 1: Графическое отделение корня
x_arr = np.linspace(0, 2.5, 500)
ax1.plot(x_arr, 4 ** x_arr, label=r'$y = 4^x$', color='red')
ax1.plot(x_arr, 5 * x_arr + 2, label=r'$y = 5x + 2$', color='green')
ax1.axvline(1.714, color='gray', linestyle='--', label='Корень ~1.714')
ax1.set_title('Графическое отделение корня')
ax1.set_xlabel('x')
ax1.set_ylabel('y')
ax1.grid(True)
ax1.legend()

# График 2: Зависимость погрешности от количества итераций
k_n, _, err_n = zip(*history_newton)
k_si, _, err_si = zip(*history_si)

ax2.plot(k_n, err_n, marker='o', color='blue', label='Метод Ньютона')
ax2.plot(k_si, err_si, marker='s', color='orange', label='Метод простой итерации')
ax2.set_yscale('log')  # Логарифмический масштаб для наглядности
ax2.set_title('Зависимость погрешности от шага (k)')
ax2.set_xlabel('Количество итераций (k)')
ax2.set_ylabel('Погрешность $|x^{(k)} - x^*|$ (log scale)')
ax2.grid(True, which="both", ls="--")
ax2.legend()

plt.tight_layout()
plt.show()
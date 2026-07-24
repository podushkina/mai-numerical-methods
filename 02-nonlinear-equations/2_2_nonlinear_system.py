import numpy as np
import matplotlib.pyplot as plt


def F(x):
    return np.array([
        3 * x[0] - np.cos(x[1]),
        3 * x[1] - np.exp(x[0])
    ])


def J(x):
    return np.array([
        [3, np.sin(x[1])],
        [-np.exp(x[0]), 3]
    ])


def Phi(x):
    return np.array([
        np.cos(x[1]) / 3,
        np.exp(x[0]) / 3
    ])


def newton_system(x0, eps, x_exact_calc=None):
    history = []
    x = x0.copy()
    k = 0
    while True:
        fx = F(x)
        jx = J(x)
        dx = np.linalg.solve(jx, -fx)

        if x_exact_calc is not None:
            error = np.max(np.abs(x - x_exact_calc))
        else:
            error = np.max(np.abs(dx))

        history.append((k, x.copy(), error))

        if np.max(np.abs(dx)) < eps:
            break
        x += dx
        k += 1
    return history


def simple_iteration_system(x0, eps, x_exact_calc):
    history = []
    x = x0.copy()
    k = 0
    q = 0.55
    while True:
        x_next = Phi(x)
        error = np.max(np.abs(x_next - x_exact_calc))
        history.append((k, x.copy(), error))

        if (q / (1 - q)) * np.max(np.abs(x_next - x)) < eps:
            history.append((k + 1, x_next.copy(), np.max(np.abs(x_next - x_exact_calc))))
            break
        x = x_next
        k += 1
    return history


eps = 1e-3
x0 = np.array([0.5, 0.5])

history_perfect = newton_system(x0, eps=1e-9)
x_exact = history_perfect[-1][1]

history_newton = newton_system(x0, eps, x_exact_calc=x_exact)
history_si = simple_iteration_system(x0, eps, x_exact_calc=x_exact)

print("--- МЕТОД НЬЮТОНА СИСТЕМ ---")
print(f"{'k':<5}{'x1':<12}{'x2':<12}{'Погрешность':<12}")
for k, x_val, err in history_newton:
    print(f"{k:<5}{x_val[0]:<12.5f}{x_val[1]:<12.5f}{err:<12.5f}")

print("\n--- МЕТОД ПРОСТОЙ ИТЕРАЦИИ СИСТЕМ ---")
print(f"{'k':<5}{'x1':<12}{'x2':<12}{'Погрешность':<12}")
for k, x_val, err in history_si:
    print(f"{k:<5}{x_val[0]:<12.5f}{x_val[1]:<12.5f}{err:<12.5f}")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

x1_vals = np.linspace(0.0, 0.6, 300)
x2_vals = np.linspace(0.0, 0.8, 300)
X1, X2 = np.meshgrid(x1_vals, x2_vals)

Z1 = 3 * X1 - np.cos(X2)
Z2 = 3 * X2 - np.exp(X1)

ax1.contour(X1, X2, Z1, levels=[0], colors='blue')
ax1.contour(X1, X2, Z2, levels=[0], colors='green')

ax1.plot([], [], color='blue', label=r'$3x_1 - \cos(x_2) = 0$')
ax1.plot([], [], color='green', label=r'$3x_2 - e^{x_1} = 0$')
ax1.plot(x_exact[0], x_exact[1], 'ro', markersize=8, label=f'Корень ({x_exact[0]:.3f}, {x_exact[1]:.3f})')

ax1.set_title('Графическое решение системы (пересечение линий)')
ax1.set_xlabel('x1')
ax1.set_ylabel('x2')
ax1.grid(True, linestyle=':')
ax1.legend(loc='upper right')

k_n, _, err_n = zip(*history_newton)
k_si, _, err_si = zip(*history_si)

ax2.plot(k_n, err_n, marker='o', color='blue', label='Метод Ньютона')
ax2.plot(k_si, err_si, marker='s', color='orange', label='Метод простой итерации')
ax2.set_yscale('log')

ax2.set_title('Зависимость погрешности от номера итерации')
ax2.set_xlabel('Количество итераций (k)')
ax2.set_ylabel('Погрешность ||x^(k) - x*|| (log scale)')
ax2.grid(True, which="both", ls="--")
ax2.legend()

plt.tight_layout()
plt.show()
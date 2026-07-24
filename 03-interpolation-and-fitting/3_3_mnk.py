import numpy as np
import matplotlib.pyplot as plt

x = np.array([-3.0, -2.0, -1.0, 0.0, 1.0, 2.0])
y = np.array([-2.9502, -1.8647, -0.63212, 1.0, 3.7183, 9.3891])
N = len(x)

print("--- Расчеты МНК ---")

A1 = np.array([
    [N, np.sum(x)],
    [np.sum(x), np.sum(x**2)]
])
B1 = np.array([np.sum(y), np.sum(x * y)])

a0, a1 = np.linalg.solve(A1, B1)
print(f"\n1-я степень: F1(x) = {a0:.4f} + {a1:.4f}*x")

f1_vals = a0 + a1 * x
Phi1 = np.sum((f1_vals - y) ** 2)
print(f"Сумма квадратов ошибок (Phi1): {Phi1:.4f}")

A2 = np.array([
    [N, np.sum(x), np.sum(x**2)],
    [np.sum(x), np.sum(x**2), np.sum(x**3)],
    [np.sum(x**2), np.sum(x**3), np.sum(x**4)]
])
B2 = np.array([np.sum(y), np.sum(x * y), np.sum(x**2 * y)])

a0_2, a1_2, a2_2 = np.linalg.solve(A2, B2)
print(f"\n2-я степень: F2(x) = {a0_2:.4f} + {a1_2:.4f}*x + {a2_2:.4f}*x^2")

f2_vals = a0_2 + a1_2 * x + a2_2 * x**2
Phi2 = np.sum((f2_vals - y) ** 2)
print(f"Сумма квадратов ошибок (Phi2): {Phi2:.4f}")

x_smooth = np.linspace(min(x) - 0.5, max(x) + 0.5, 200)
y_smooth_f1 = a0 + a1 * x_smooth
y_smooth_f2 = a0_2 + a1_2 * x_smooth + a2_2 * x_smooth**2

plt.figure(figsize=(9, 6))
plt.scatter(x, y, color='black', zorder=5, label='Исходные точки (таблица)')
plt.plot(x_smooth, y_smooth_f1, color='blue', linestyle='-', label=f'1-я степень (Phi={Phi1:.4f})')
plt.plot(x_smooth, y_smooth_f2, color='red', linestyle='--', label=f'2-я степень (Phi={Phi2:.4f})')

plt.title('Приближение функции методом наименьших квадратов (МНК)')
plt.xlabel('X')
plt.ylabel('Y')
plt.grid(True, linestyle=':')
plt.legend()
plt.show()
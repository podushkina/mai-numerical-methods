import numpy as np


def f(x):
    return np.exp(x) + x


def lagrange_poly(x, nodes, y_nodes):
    res = 0
    n = len(nodes)
    for i in range(n):
        p = 1
        for j in range(n):
            if i != j:
                # L_i(x) = \prod (x - x_j) / (x_i - x_j)
                p *= (x - nodes[j]) / (nodes[i] - nodes[j])

        # P_n(x) = \sum (y_i * L_i(x))
        res += y_nodes[i] * p
    return res


def divided_differences(nodes, y_nodes):
    n = len(nodes)
    coef = np.zeros((n, n))
    coef[:, 0] = y_nodes

    for j in range(1, n):
        for i in range(n - j):
            # f(x_0,...,x_k) = (f(x_1,...,x_k) - f(x_0,...,x_{k-1})) / (x_k - x_0)
            coef[i, j] = (coef[i + 1, j - 1] - coef[i, j - 1]) / (nodes[i + j] - nodes[i])

    return coef[0, :]


def newton_poly(x, nodes, coef):
    n = len(nodes)
    res = coef[0]
    p = 1
    for i in range(1, n):
        # (x - x_0)(x - x_1)...(x - x_{i-1})
        p *= (x - nodes[i - 1])

        # P_n(x) = c_0 + c_1(x-x_0) + c_2(x-x_0)(x-x_1) + ...
        res += coef[i] * p
    return res


x_star = -0.5
y_exact = f(x_star)

nodes_cases = {
    "а) Фиксированные целые узлы": np.array([-2.0, -1.0, 0.0, 1.0]),
    "б) Узел 0 смещен в точку 0.2": np.array([-2.0, -1.0, 0.2, 1.0])
}

print(f"Истинное значение f(x*) в точке {x_star}: {y_exact:.6f}\n")

for case_name, nodes in nodes_cases.items():
    print(f"=== Набор {case_name} ===")
    print(f"Используемые узлы (x_i): {nodes}")

    y_nodes = f(nodes)
    print(f"Значения в узлах (y_i):   {np.round(y_nodes, 5)}")

    y_lagr = lagrange_poly(x_star, nodes, y_nodes)
    err_lagr = abs(y_exact - y_lagr)

    coef_newt = divided_differences(nodes, y_nodes)
    y_newt = newton_poly(x_star, nodes, coef_newt)
    err_newt = abs(y_exact - y_newt)

    print(f"Коэффициенты Ньютона (c_i): {np.round(coef_newt, 5)}")

    print(f"{'Метод':<12}{'Значение в x*':<16}{'Абс. Погрешность':<16}")
    print(f"{'Лагранж':<12}{y_lagr:<16.6f}{err_lagr:<16.2e}")
    print(f"{'Ньютон':<12}{y_newt:<16.6f}{err_newt:<16.2e}")
    print("-" * 50 + "\n")
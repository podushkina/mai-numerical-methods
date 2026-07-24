import math

# --- 1. Исходные данные Варианта 17 ---
x0, x_end = 0.0, 1.0
h = 0.1


def exact_sol(x):
    return x - 3 + 1.0 / (x + 1)


# Защита от деления на ноль при x = 1.0
def p_func(x):
    denom = x ** 2 - 1.0
    if abs(denom) < 1e-15:
        denom = -1e-15 if denom <= 0 else 1e-15
    return (x - 3.0) / denom


def q_func(x):
    denom = x ** 2 - 1.0
    if abs(denom) < 1e-15:
        denom = -1e-15 if denom <= 0 else 1e-15
    return -1.0 / denom


def f_func(x):
    return 0.0


def f_z(x, y, z):
    return f_func(x) - p_func(x) * z - q_func(x) * y


# --- 2. Интегратор Рунге-Кутты 4-го порядка ---
def rk4_step(x, y, z, h_step):
    k1 = h_step * z
    l1 = h_step * f_z(x, y, z)

    k2 = h_step * (z + 0.5 * l1)
    l2 = h_step * f_z(x + 0.5 * h_step, y + 0.5 * k1, z + 0.5 * l1)

    k3 = h_step * (z + 0.5 * l2)
    l3 = h_step * f_z(x + 0.5 * h_step, y + 0.5 * k2, z + 0.5 * l2)

    k4 = h_step * (z + l3)
    l4 = h_step * f_z(x + h_step, y + k3, z + l3)

    return y + (k1 + 2 * k2 + 2 * k3 + k4) / 6.0, z + (l1 + 2 * l2 + 2 * l3 + l4) / 6.0


def solve_ivp_rk4(y_start, z_start, h_step):
    N_steps = int(round((x_end - x0) / h_step))
    x_arr = [x0 + i * h_step for i in range(N_steps + 1)]
    y_arr = [0.0] * (N_steps + 1)
    z_arr = [0.0] * (N_steps + 1)

    y_arr[0], z_arr[0] = y_start, z_start
    for k in range(N_steps):
        y_arr[k + 1], z_arr[k + 1] = rk4_step(x_arr[k], y_arr[k], z_arr[k], h_step)
    return x_arr, y_arr, z_arr


# --- 3. МЕТОД СТРЕЛЬБЫ ---
def shooting_method(h_step):
    def phi(eta):
        _, y_res, z_res = solve_ivp_rk4(eta, 0.0, h_step)
        return z_res[-1] + y_res[-1] + 0.75

    eta0, eta1 = -2.0, -1.0
    phi0, phi1 = phi(eta0), phi(eta1)
    eps = 1e-7

    while abs(phi1) > eps:
        if abs(phi1 - phi0) < 1e-12:
            break
        eta_next = eta1 - (eta1 - eta0) / (phi1 - phi0) * phi1
        eta0, eta1 = eta1, eta_next
        phi0, phi1 = phi1, phi(eta1)

    x_arr, y_arr, _ = solve_ivp_rk4(eta1, 0.0, h_step)
    return x_arr, y_arr


# --- 4. КОНЕЧНО-РАЗНОСТНЫЙ МЕТОД ---
def finite_difference_method(h_step):
    N = int(round((x_end - x0) / h_step))
    x_arr = [x0 + i * h_step for i in range(N + 1)]

    A = [0.0] * (N + 1)
    B = [0.0] * (N + 1)
    C = [0.0] * (N + 1)
    D = [0.0] * (N + 1)

    C[0] = -1.0
    B[0] = 1.0
    D[0] = 0.0

    for k in range(1, N):
        pk = p_func(x_arr[k])
        qk = q_func(x_arr[k])
        fk = f_func(x_arr[k])

        A[k] = 1.0 - (pk * h_step) / 2.0
        C[k] = -2.0 + (qk * h_step ** 2)
        B[k] = 1.0 + (pk * h_step) / 2.0
        D[k] = fk * h_step ** 2

    A[N] = -1.0
    C[N] = 1.0 + h_step
    D[N] = -0.75 * h_step

    alpha = [0.0] * (N + 1)
    beta = [0.0] * (N + 1)

    alpha[1] = -B[0] / C[0]
    beta[1] = D[0] / C[0]

    for k in range(1, N):
        denom = C[k] + A[k] * alpha[k]
        alpha[k + 1] = -B[k] / denom
        beta[k + 1] = (D[k] - A[k] * beta[k]) / denom

    y_arr = [0.0] * (N + 1)
    y_arr[N] = (D[N] - A[N] * beta[N]) / (C[N] + A[N] * alpha[N])

    for k in range(N - 1, -1, -1):
        y_arr[k] = alpha[k + 1] * y_arr[k + 1] + beta[k + 1]

    return x_arr, y_arr


# --- 5. Вычисление и вывод результатов ---
if __name__ == "__main__":
    x_pts, y_shoot = shooting_method(h)
    _, y_fdm = finite_difference_method(h)

    # Расчет Рунге-Ромберга на двойном шаге 2h = 0.2
    _, y_shoot_2h = shooting_method(h * 2)
    _, y_fdm_2h = finite_difference_method(h * 2)

    print("=== РЕЗУЛЬТАТЫ РЕШЕНИЯ КРАЕВОЙ ЗАДАЧИ (ВАРИАНТ 17) ===")
    print(f"{'x':<6}{'Exact':<12}{'Shooting':<12}{'FDM':<12}{'Err Shoot':<12}{'Err FDM':<12}{'R-R Shoot':<12}{'R-R FDM':<12}")
    print("-" * 96)

    for i, x in enumerate(x_pts):
        exact = exact_sol(x)
        err_shoot = abs(exact - y_shoot[i])
        err_fdm = abs(exact - y_fdm[i])

        rr_shoot, rr_fdm = "-", "-"
        if i % 2 == 0:
            idx_2h = i // 2
            # Метод стрельбы (РК4): p=4, делитель 15.0
            rr_shoot = f"{abs(y_shoot[i] - y_shoot_2h[idx_2h]) / 15.0:.2e}"
            # КРМ: p=1 на границе, делитель 1.0
            rr_fdm = f"{abs(y_fdm[i] - y_fdm_2h[idx_2h]) / 1.0:.2e}"

        print(f"{x:<6.1f}{exact:<12.6f}{y_shoot[i]:<12.6f}{y_fdm[i]:<12.6f}"
              f"{err_shoot:<12.2e}{err_fdm:<12.2e}{rr_shoot:<12}{rr_fdm:<12}")
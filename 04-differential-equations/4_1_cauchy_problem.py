import math


def exact_sol(x):
    return x + 1 + math.exp(x)


def f_z(x, y, z):
    return ((x + 1) * z - y) / x


def runge_kutta_step(x, y, z, h):
    k1 = h * z
    l1 = h * f_z(x, y, z)

    k2 = h * (z + 0.5 * l1)
    l2 = h * f_z(x + 0.5 * h, y + 0.5 * k1, z + 0.5 * l1)

    k3 = h * (z + 0.5 * l2)
    l3 = h * f_z(x + 0.5 * h, y + 0.5 * k2, z + 0.5 * l2)

    k4 = h * (z + l3)
    l4 = h * f_z(x + h, y + k3, z + l3)

    dy = (k1 + 2 * k2 + 2 * k3 + k4) / 6
    dz = (l1 + 2 * l2 + 2 * l3 + l4) / 6
    return y + dy, z + dz


def solve_euler(x0, y0, z0, x_end, h):
    N = int(round((x_end - x0) / h))
    x_arr = [x0 + i * h for i in range(N + 1)]
    y_arr = [0.0] * (N + 1)
    z_arr = [0.0] * (N + 1)
    y_arr[0], z_arr[0] = y0, z0

    for k in range(N):
        y_arr[k + 1] = y_arr[k] + h * z_arr[k]
        z_arr[k + 1] = z_arr[k] + h * f_z(x_arr[k], y_arr[k], z_arr[k])
    return x_arr, y_arr


def solve_rk4(x0, y0, z0, x_end, h):
    N = int(round((x_end - x0) / h))
    x_arr = [x0 + i * h for i in range(N + 1)]
    y_arr = [0.0] * (N + 1)
    z_arr = [0.0] * (N + 1)
    y_arr[0], z_arr[0] = y0, z0

    for k in range(N):
        y_arr[k + 1], z_arr[k + 1] = runge_kutta_step(x_arr[k], y_arr[k], z_arr[k], h)
    return x_arr, y_arr


def solve_adams(x0, y0, z0, x_end, h):
    N = int(round((x_end - x0) / h))
    x_arr = [x0 + i * h for i in range(N + 1)]
    y_arr = [0.0] * (N + 1)
    z_arr = [0.0] * (N + 1)
    y_arr[0], z_arr[0] = y0, z0

    for k in range(3):
        y_arr[k + 1], z_arr[k + 1] = runge_kutta_step(x_arr[k], y_arr[k], z_arr[k], h)

    for k in range(3, N):
        f_y = lambda idx: z_arr[idx]
        f_z_val = lambda idx: f_z(x_arr[idx], y_arr[idx], z_arr[idx])

        y_arr[k + 1] = y_arr[k] + (h / 24) * (55 * f_y(k) - 59 * f_y(k - 1) + 37 * f_y(k - 2) - 9 * f_y(k - 3))
        z_arr[k + 1] = z_arr[k] + (h / 24) * (
                    55 * f_z_val(k) - 59 * f_z_val(k - 1) + 37 * f_z_val(k - 2) - 9 * f_z_val(k - 3))
    return x_arr, y_arr


x0, x_end = 1.0, 2.0
y0 = 2.0 + math.e
z0 = 1.0 + math.e
h = 0.1

x_pts, y_euler = solve_euler(x0, y0, z0, x_end, h)
_, y_rk4 = solve_rk4(x0, y0, z0, x_end, h)
_, y_adams = solve_adams(x0, y0, z0, x_end, h)

_, y_rk4_double = solve_rk4(x0, y0, z0, x_end, h * 2)

print(
    f"{'x':<6}{'Exact':<12}{'Euler':<12}{'RK4':<12}{'Adams':<12}{'Err Euler':<12}{'Err RK4':<12}{'Err Adams':<12}{'Runge-Romb':<12}")
print("-" * 102)

for i, x in enumerate(x_pts):
    exact = exact_sol(x)
    err_euler = abs(exact - y_euler[i])
    err_rk4 = abs(exact - y_rk4[i])
    err_adams = abs(exact - y_adams[i])

    r_romb = "-"
    if i % 2 == 0:
        idx_2h = i // 2
        r_romb = f"{abs(y_rk4[i] - y_rk4_double[idx_2h]) / 15:.2e}"

    print(f"{x:<6.1f}{exact:<12.6f}{y_euler[i]:<12.6f}{y_rk4[i]:<12.6f}{y_adams[i]:<12.6f}"
          f"{err_euler:<12.2e}{err_rk4:<12.2e}{err_adams:<12.2e}{r_romb:<12}")
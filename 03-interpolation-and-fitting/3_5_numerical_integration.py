def f(x):
    return 1 / (256 - x ** 4)


def rectangles_method(x_0, x_k, h):
    n = int((x_k - x_0) / h)
    integral = 0
    for i in range(n):
        mid_x = x_0 + i * h + h / 2
        integral += f(mid_x)
    return integral * h


def trapezoids_method(x_0, x_k, h):
    n = int((x_k - x_0) / h)
    integral = (f(x_0) + f(x_k)) / 2
    for i in range(1, n):
        integral += f(x_0 + i * h)
    return integral * h


def simpson_method(x_0, x_k, h):
    n = int((x_k - x_0) / h)
    integral = f(x_0) + f(x_k)

    for i in range(1, n):
        x_i = x_0 + i * h
        if i % 2 != 0:
            integral += 4 * f(x_i)
        else:
            integral += 2 * f(x_i)

    return (h / 3) * integral


def main():
    x_0, x_k = -2.0, 2.0
    h1, h2 = 1.0, 0.5

    print(f"Интегрирование на отрезке [{x_0}, {x_k}]")
    print("=" * 70)

    # Метод прямоугольников
    i_rect_h1 = rectangles_method(x_0, x_k, h1)
    i_rect_h2 = rectangles_method(x_0, x_k, h2)
    err_rect = abs(i_rect_h2 - i_rect_h1) / (2 ** 2 - 1)

    print(f"Метод прямоугольников:")
    print(f"  I (h1={h1}): {i_rect_h1:.6f}")
    print(f"  I (h2={h2}): {i_rect_h2:.6f}")
    print(f"  Погрешность Рунге-Ромберга: {err_rect:.6e}")
    print("-" * 70)

    # Метод трапеций
    i_trap_h1 = trapezoids_method(x_0, x_k, h1)
    i_trap_h2 = trapezoids_method(x_0, x_k, h2)
    err_trap = abs(i_trap_h2 - i_trap_h1) / (2 ** 2 - 1)

    print(f"Метод трапеций:")
    print(f"  I (h1={h1}): {i_trap_h1:.6f}")
    print(f"  I (h2={h2}): {i_trap_h2:.6f}")
    print(f"  Погрешность Рунге-Ромберга: {err_trap:.6e}")
    print("-" * 70)

    # Метод Симпсона
    i_simp_h1 = simpson_method(x_0, x_k, h1)
    i_simp_h2 = simpson_method(x_0, x_k, h2)
    err_simp = abs(i_simp_h2 - i_simp_h1) / (2 ** 4 - 1)

    print(f"Метод Симпсона:")
    print(f"  I (h1={h1}): {i_simp_h1:.6f}")
    print(f"  I (h2={h2}): {i_simp_h2:.6f}")
    print(f"  Погрешность Рунге-Ромберга: {err_simp:.6e}")
    print("=" * 70)


if __name__ == "__main__":
    main()
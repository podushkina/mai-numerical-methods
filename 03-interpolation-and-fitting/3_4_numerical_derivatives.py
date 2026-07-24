def main():
    x = [-0.2, 0.0, 0.2, 0.4, 0.6]
    y = [-0.40136, 0.0, 0.40136, 0.81152, 1.2435]
    X_star = 0.2

    try:
        idx = x.index(X_star)
    except ValueError:
        print(f"Ошибка: Точка X* = {X_star} не найдена в таблице.")
        return

    print(f"=== Численное дифференцирование для X* = {X_star} (индекс i = {idx}) ===")
    print("-" * 60)

    if idx > 0:
        y_prime_left = (y[idx] - y[idx - 1]) / (x[idx] - x[idx - 1])
        print(f"Левосторонняя производная y'({X_star}): {y_prime_left:.5f}")
    else:
        print("Левосторонняя производная: невозможно вычислить (нет левой точки)")

    if idx < len(x) - 1:
        y_prime_right = (y[idx + 1] - y[idx]) / (x[idx + 1] - x[idx])
        print(f"Правосторонняя производная y'({X_star}): {y_prime_right:.5f}")
    else:
        print("Правосторонняя производная: невозможно вычислить (нет правой точки)")

    print("-" * 60)

    if idx > 0 and idx < len(x) - 1:
        term1 = (y[idx] - y[idx - 1]) / (x[idx] - x[idx - 1])
        diff_quotient1 = (y[idx + 1] - y[idx]) / (x[idx + 1] - x[idx])
        diff_quotient2 = (y[idx] - y[idx - 1]) / (x[idx] - x[idx - 1])
        term2 = (diff_quotient1 - diff_quotient2) / (x[idx + 1] - x[idx - 1])
        multiplier = 2 * X_star - x[idx - 1] - x[idx + 1]

        y_prime_second_order = term1 + term2 * multiplier
        print(f"Первая производная (2-й порядок точности) y'({X_star}): {y_prime_second_order:.5f}")
    else:
        print("Второй порядок точности: недостаточно точек (нужны соседние слева и справа)")

    print("-" * 60)

    if idx > 0 and idx < len(x) - 1:
        diff_quotient1 = (y[idx + 1] - y[idx]) / (x[idx + 1] - x[idx])
        diff_quotient2 = (y[idx] - y[idx - 1]) / (x[idx] - x[idx - 1])

        y_double_prime = 2 * (diff_quotient1 - diff_quotient2) / (x[idx + 1] - x[idx - 1])
        print(f"Вторая производная y''({X_star}): {y_double_prime:.5f}")
    else:
        print("Вторая производная: невозможно вычислить (нужны соседние слева и справа)")
    print("-" * 60)


if __name__ == "__main__":
    main()
import numpy as np
import matplotlib.pyplot as plt


# Исходная функция
def f(x):
    return 5 * np.sin(x) - x + 1


# 1 производная
def f1(x):
    return 5 * np.cos(x) - 1


# 2 производная
def f2(x):
    return -5 * np.sin(x)


# 3 производная
def f3(x):
    return -5 * np.cos(x)


# 4 производная
def f4(x):
    return 5 * np.sin(x)


# Разложение Тейлора 1 степени
def p1_x(x, c):
    return f(c) + f1(c) * (x - c)


# Разложение Тейлора 2 степени
def p2_x(x, c):
    return p1_x(x, c) + (f2(c) * (x - c) ** 2) / 2


# Разложение Тейлора 3 степени
def p3_x(x, c):
    return p2_x(x, c) + (f3(c) * (x - c) ** 3) / 6


# Разложение Тейлора 4 степени
def p4_x(x, c):
    return p3_x(x, c) + (f(c) * (x - c) ** 4) / 24


# Ошибка 1 степени
def r1_x(x, c):
    return f(x) - p1_x(x, c)


# Ошибка 2 степени
def r2_x(x, c):
    return f(x) - p2_x(x, c)


# Ошибка 3 степени
def r3_x(x, c):
    return f(x) - p3_x(x, c)


# Ошибка 4 степени
def r4_x(x, c):
    return f(x) - p4_x(x, c)


# Нахождение ошибок аппроксимации
def approximation_mistakes(a, b, st, dpoint):
    m1 = np.zeros(round((b - a) / st))
    m2 = np.zeros(round((b - a) / st))
    m3 = np.zeros(round((b - a) / st))
    m4 = np.zeros(round((b - a) / st))
    i = 0
    j = a
    while j < b - st:
        m1[i] = r1_x(j, dpoint)
        m2[i] = r2_x(j, dpoint)
        m3[i] = r3_x(j, dpoint)
        m4[i] = r4_x(j, dpoint)
        j += st
        i += 1
    return m1, m2, m3, m4


# Графики ошибок аппроксимации
def draw_mistakes(x, m1, m2, m3, m4, dpoint):
    plt.figure(figsize=(10, 10))
    plt.title("Plots of approximation mistakes for the decomposition point x =" + str(dpoint))
    plt.plot(x, m1, label='Approximation mistake for a polynomial of degree 1')
    plt.plot(x, m2, label='Approximation mistake for a polynomial of degree 2')
    plt.plot(x, m3, label='Approximation mistake for a polynomial of degree 3')
    plt.plot(x, m4, label='Approximation mistake for a polynomial of degree 4')
    plt.axhline(y=0, color="black", linewidth=1.5)
    plt.axvline(x=0, color="black", linewidth=1.5)
    plt.grid(True)
    plt.legend()
    plt.show()


# интерполяционный полином Лагранжа
def lagrange_polynomial(x, new_p, func, n):
    s = 0
    for i in range(n):
        t = 1
        for j in range(n):
            if j != i:
                t *= (x - new_p[j]) / (new_p[i] - new_p[j])
        s += (func(new_p[i]) * t)
    return s


# графики интерполяционного полинома Лагранжа
def draw_lagrange_poly(x, func, lag_p, n):
    plt.figure(figsize=(10, 10))
    plt.title("The graphs of the original function and the Lagrange polynomial for n =" + str(n))
    plt.plot(x, lagrange_polynomial(x, lag_p, func, n), 'g--')
    plt.plot(x, func(x), color="red")
    plt.axhline(y=0, color="black", linewidth=1.5)
    plt.axvline(x=0, color="black", linewidth=1.5)
    plt.grid(True)
    plt.show()


def sliding_filling(a, func, length):
    points = []
    d0, d1, d2 = [], [], []
    for i in range(length + 1):
        points.append(a + 20 / length * i)

    for i in range(len(points) - 2):
        d0.append(func(points[i]))

        d1.append((func(points[i + 1]) - func(points[i])) / (points[i + 1] - points[i]))

        tc = (func(points[i + 1]) - func(points[i])) / (points[i + 1] - points[i])
        tmp = func(points[i + 2]) - func(points[i]) - tc * (points[i + 2] - points[i])
        d2.append(tmp / ((points[i + 2] - points[i]) * (points[i + 2] - points[i + 1])))

    return points, d0, d1, d2


# скользящий полином
def sliding_pol(b0, b1, b2, x0, x1, x):
    return b0 + b1 * (x - x0) + b2 * (x - x0) * (x - x1)


# графики скользящих полиномов
def draw_sliding_poly(points, b0, b1, b2):
    plt.title("Sliding polynomials for n = " + str(len(points) - 2))
    for i in range(len(points) - 2):
        xs = np.arange(points[i], points[i + 2], 0.001)
        plt.plot(xs, sliding_pol(b0[i], b1[i], b2[i], points[i], points[i + 1], xs))
    plt.axhline(y=0, color="black", linewidth=1.5)
    plt.axvline(x=0, color="black", linewidth=1.5)
    plt.grid(True)
    plt.show()


A = -7.0
B = 9.0
ST = 0.002
X = np.arange(A, B, ST)
dec_points = [-7, 9, 1]
lag_points1 = [-7, -4, 0, 3, 8]
lag_points2 = [-7, -5, -2, -1, 0, 2, 3, 5, 7, 8]

for i in range(len(dec_points)):
    app_m1, app_m2, app_m3, app_m4 = approximation_mistakes(A, B, ST, dec_points[i])
    draw_mistakes(X, app_m1, app_m2, app_m3, app_m4, dec_points[i])

draw_lagrange_poly(X, f, lag_points1, 5)
draw_lagrange_poly(X, f, lag_points2, 10)

for i in range(2, 16, 1):
    sliding_points, a0, a1, a2 = sliding_filling(A, f, i)
    draw_sliding_poly(sliding_points, a0, a1, a2)

print(r1_x(X, dec_points[0]))

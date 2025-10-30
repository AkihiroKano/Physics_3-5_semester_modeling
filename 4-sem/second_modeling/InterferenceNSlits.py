import numpy as np
import matplotlib.pyplot as plt


def wavelength_to_rgb(lambda_nm):
    """
    Преобразует длину волны (в нанометрах) в RGB-значения.
    Учитывает физиологические особенности восприятия цвета человеком.

    Параметры:
        lambda_nm (float): Длина волны в диапазоне 380-750 нм.

    Возвращает:
        tuple: (R, G, B) в диапазоне [0, 1].
    """
    lambda_nm = np.clip(lambda_nm, 380, 750)
    gamma = 0.8  # Гамма-коррекция для лучшего восприятия

    # Эмпирическая модель цветового восприятия
    if lambda_nm < 440:  # Фиолетовый-синий диапазон
        r = -(lambda_nm - 440) / (440 - 380)
        g, b = 0.0, 1.0
    elif lambda_nm < 490:  # Сине-голубой
        r, g, b = 0.0, (lambda_nm - 440) / (490 - 440), 1.0
    elif lambda_nm < 510:  # Голубой-зеленый
        r, g, b = 0.0, 1.0, -(lambda_nm - 510) / (510 - 490)
    elif lambda_nm < 580:  # Зеленый-желтый
        r, g, b = (lambda_nm - 510) / (580 - 510), 1.0, 0.0
    elif lambda_nm < 645:  # Желтый-красный
        r, g, b = 1.0, -(lambda_nm - 645) / (645 - 580), 0.0
    else:  # Красный
        r, g, b = 1.0, 0.0, 0.0

    # Коррекция для крайних участков спектра
    attenuation = 1.0
    if lambda_nm < 420:  # Фиолетовый (ослабление)
        attenuation = 0.3 + 0.7 * (lambda_nm - 380) / (420 - 380)
    elif lambda_nm > 700:  # Глубокий красный
        attenuation = 0.3 + 0.7 * (750 - lambda_nm) / (750 - 700)

    return (
        (r * attenuation) ** gamma,
        (g * attenuation) ** gamma,
        (b * attenuation) ** gamma
    )


def calculate_interference(x, N, b, d, L, lambda0, delta_lambda=0, K=100):
    """
    Рассчитывает распределение интенсивности и цвета для интерференции от N щелей.

    Параметры:
        x (np.array): Массив координат на экране (м).
        N (int): Число щелей (1-10).
        b (float): Ширина одной щели (м).
        d (float): Период решётки (расстояние между центрами щелей, м).
        L (float): Расстояние от решётки до экрана (м).
        lambda0 (float): Центральная длина волны (м).
        delta_lambda (float): Ширина спектра (м), 0 для монохроматического света.
        K (int): Число точек для интегрирования по спектру.

    Возвращает:
        tuple: (интенсивность, RGB-массив).
    """
    # Валидация входных данных
    if not (1 <= N <= 10):
        raise ValueError("Число щелей должно быть 1 ≤ N ≤ 10")
    if d < b:
        raise ValueError("Период решётки d не может быть меньше ширины щели b")
    if lambda0 <= 0 or delta_lambda < 0:
        raise ValueError("Некорректные параметры длины волны")

    # Расчет углов наблюдения (малые углы: θ ≈ x/L)
    theta = np.arctan(x / L)
    sin_theta = np.sin(theta)

    if delta_lambda > 0:  # Квазимонохроматический свет
        lambdas = np.linspace(lambda0 - delta_lambda / 2, lambda0 + delta_lambda / 2, K)
        total_intensity = np.zeros_like(x)
        total_rgb = np.zeros((x.size, 3))

        for l in lambdas:
            # Дифракция на одной щели
            beta = np.pi * b * sin_theta / l
            single_slit = (np.sinc(beta / np.pi)) ** 2

            # Интерференция N щелей
            alpha = np.pi * d * sin_theta / l
            multi_slit = (np.sin(N * alpha) / np.sin(alpha)) ** 2
            multi_slit = np.nan_to_num(multi_slit)  # Обработка alpha=0

            intensity = single_slit * multi_slit
            total_intensity += intensity

            # Преобразование в RGB
            r, g, b_rgb = wavelength_to_rgb(l * 1e9)
            total_rgb += intensity[:, np.newaxis] * [r, g, b_rgb]

        # Нормализация
        total_intensity /= K
        max_rgb = np.max(total_rgb)
        if max_rgb > 0:
            total_rgb /= max_rgb
    else:  # Монохроматический свет
        beta = np.pi * b * sin_theta / lambda0
        alpha = np.pi * d * sin_theta / lambda0

        single_slit = (np.sinc(beta / np.pi)) ** 2
        multi_slit = (np.sin(N * alpha) / np.sin(alpha)) ** 2
        multi_slit = np.nan_to_num(multi_slit)

        total_intensity = single_slit * multi_slit
        r, g, b = wavelength_to_rgb(lambda0 * 1e9)
        total_rgb = np.outer(total_intensity, [r, g, b])
        total_rgb /= np.max(total_rgb) if np.max(total_rgb) > 0 else 1.0

    return total_intensity, total_rgb


def plot_results(x, intensity, rgb):
    """
    Визуализирует результаты расчета.

    Параметры:
        x (np.array): Координаты на экране.
        intensity (np.array): Распределение интенсивности.
        rgb (np.array): RGB-массив размером (len(x), 3).
    """
    plt.figure(figsize=(12, 6))

    # График интенсивности
    plt.subplot(2, 1, 1)
    plt.plot(x, intensity, color='blue', linewidth=1.5)
    plt.title('Распределение интенсивности', fontsize=14)
    plt.xlabel('Координата на экране, м', fontsize=12)
    plt.ylabel('Относительная интенсивность', fontsize=12)
    plt.grid(True, alpha=0.3)

    # Цветовая карта
    plt.subplot(2, 1, 2)
    plt.imshow(rgb[np.newaxis, :, :],
               aspect='auto',
               extent=(x.min(), x.max(), 0, 1))
    plt.title('Цветное распределение', fontsize=14)
    plt.xlabel('Координата на экране, м', fontsize=12)
    plt.yticks([])
    plt.tight_layout()
    plt.show()


# Параметры запуска
params = {
    'N': 1,            # Число щелей (должно быть 1-10)
    'b': 1e-6,         # Ширина щели (1 мкм)
    'd': 2e-6,         # Период решётки (2 мкм, должен быть ≥ b)
    'L': 1.0,          # Расстояние до экрана (1 м)
    'lambda0': 500e-9, # Центральная длина волны (500 нм = зеленый свет)
    'delta_lambda': 0  # Ширина спектра (0 = монохроматический свет)
}

# Автоматический расчет области наблюдения
m_max = 3  # Число отображаемых порядков интерференции
x_max = (m_max * params['lambda0'] * params['L']) / params['d']
x = np.linspace(-x_max, x_max, 3000)  # Высокое разрешение

intensity, rgb = calculate_interference(x=x, **params)
plot_results(x, intensity, rgb)
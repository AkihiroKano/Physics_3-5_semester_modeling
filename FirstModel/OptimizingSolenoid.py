import numpy as np
import matplotlib.pyplot as plt
from math import pi, sqrt

def input_positive(prompt):
    """Функция для ввода положительных чисел"""
    while True:
        try:
            value = float(input(prompt))
            if value <= 0:
                raise ValueError
            return value
        except ValueError:
            print("Ошибка: введите положительное число!")

# Ввод параметров
print("Введите параметры катушки:")
L = input_positive("Длина провода, L (м): ")
D = input_positive("Диаметр каркаса, D (м): ")
d = input_positive("Диаметр провода, d (м): ")
I = input_positive("Ток в катушке, I (А): ")

# Расчет оптимальной длины
l_opt = (L * d) / (pi * D)
print(f"\nОптимальная длина катушки: {l_opt:.4f} м")

# Расчет индуктивности при оптимальной длине
N_opt = L / (pi * (D + d))  # Число витков с учетом диаметра провода
A = pi * (D/2)**2  # Площадь сечения
L_ind = (4e-7 * pi) * (N_opt**2) * A / l_opt  # Индуктивность

print(f"Индуктивность при l = {l_opt:.4f} м: {L_ind:.4e} Гн")

# Диапазон для построения графика
print("\nЗадайте диапазон длин катушки:")
l_min = input_positive("Минимальная длина (м): ")
l_max = input_positive("Максимальная длина (м): ")

# Расчет магнитной индукции
l_values = np.linspace(l_min, l_max, 1000)
B_values = (4e-7 * pi * I * L) / (2 * pi * D * np.sqrt(l_values**2 + D**2))

# Построение графика
plt.figure(figsize=(10, 6))
plt.plot(l_values, B_values, label='B(l)')
plt.axvline(l_opt, color='r', linestyle='--', label=f'Оптимум: {l_opt:.4f} м')
plt.scatter(l_opt, np.max(B_values), color='r')

plt.title(f'Зависимость B = f(l)\nL={L} м, D={D} м, d={d} м, I={I} А')
plt.xlabel('Длина катушки, l (м)')
plt.ylabel('Магнитная индукция, B (Тл)')
plt.grid(True, linestyle='--', alpha=0.7)
plt.legend()
plt.tight_layout()
plt.show()
# Импортируем генератор из твоего первого файла
from dataset import generate_data 
# Импортируем функции из файла модели
from model import gradient_descent, predict, mean_squared_error

# 1. Получаем данные
X, y = generate_data()

# 2. Инициализируем стартовые параметры (нули)
n_features = len(X[0])
initial_w = [0.0] * n_features
initial_b = 0.0

# 3. ПОЛИГОН ДЛЯ ТЕСТОВ (Меняй эти значения!)
alpha = 0.00000001     # Размер шага
num_epochs = 1000000  # Количество итераций

print("=== Старт обучения ===")
print(f"Шаг (alpha): {alpha} | Эпохи: {num_epochs}\n")

# 4. Запуск алгоритма
final_w, final_b = gradient_descent(X, y, initial_w, initial_b, alpha, num_epochs)

# 5. Финальный замер
final_predictions = predict(X, final_w, final_b)
final_cost = mean_squared_error(y, final_predictions)

print("\n=== Результаты ===")
print(f"Финальная ошибка (MSE): {final_cost:.4f}")
print(f"Веса (w): {final_w}")
print(f"Смещение (b): {final_b:.4f}")
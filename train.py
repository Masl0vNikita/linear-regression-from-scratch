
from dataset import generate_data 

from model import gradient_descent, predict, mean_squared_error, normalize_features


X, y = generate_data()
X = normalize_features(X)

n_features = len(X[0])
initial_w = [0.0] * n_features
initial_b = 0.0

alpha = 0.1    
num_epochs = 1000

print("=== Старт обучения ===")
print(f"Шаг (alpha): {alpha} | Эпохи: {num_epochs}\n")


final_w, final_b = gradient_descent(X, y, initial_w, initial_b, alpha, num_epochs)


final_predictions = predict(X, final_w, final_b)
final_cost = mean_squared_error(y, final_predictions)

print("\n=== Результаты ===")
print(f"Финальная ошибка (MSE): {final_cost:.4f}")
print(f"Веса (w): {final_w}")
print(f"Смещение (b): {final_b:.4f}")
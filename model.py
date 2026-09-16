def predict(X, w, b):

    y_preds = []
    for i in range(len(X)):
        y_pred = sum(w[j] * X[i][j] for j in range(len(w))) + b
        y_preds.append(y_pred)
    return y_preds

def mean_squared_error(y_true, y_pred):
    n = len(y_true)
    mse = sum((y_true[i] - y_pred[i]) ** 2 for i in range(n)) / n
    return mse

def compute_gradient(X, y, w, b):
    """
    dj_dw = (2/N) * sum((y_real - y_i) * x_ij)
    dj_db = (2/N) * sum(y_real - y_i)
    """
    N = len(X)
    n = len(w)
    
    y_real = predict(X, w, b)
    
    dj_dw = [0.0] * n
    dj_db = 0.0

    for i in range(N):
        error = y_real[i] - y[i]
        dj_db += error * 2/N

        for j in range(n):
            dj_dw[j] += error * X[i][j] * 2/N

    return dj_dw, dj_db

def gradient_descent(X, y, w, b, alpha, num_iters):
    n = len(w)
    
    for i in range(num_iters):

        dj_dw, dj_db = compute_gradient(X, y, w, b)
        
  
        b = b - alpha * dj_db
        
        for j in range(n):
            w[j] = w[j] - alpha * dj_dw[j]


        if i % 10000 == 0:
            y_pred = predict(X, w, b)
            cost = mean_squared_error(y, y_pred)
            print(f"Итерация {i:5d} | Ошибка (MSE): {cost:.4f}")
            
    return w, b
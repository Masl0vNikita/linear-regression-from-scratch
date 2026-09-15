import random

def generate_data(n=100):
    w_true = [100.0, 5.0, 400.0, 10000.0, 25000.0, 40.0, 15000.0]
    b_true = -750000.0 
    
    X = []
    y = []
    random.seed(42)
    
    for _ in range(n):
        x = [
            random.uniform(500, 4000),
            random.uniform(1500, 20000),
            random.randint(1900, 2026),
            random.randint(0, 4),
            random.randint(1, 10),
            random.uniform(0, 3000),
            random.randint(1, 4)
        ]
        X.append(x)
        
        price = sum(w * f for w, f in zip(w_true, x)) + b_true
        price += random.gauss(0, 15000.0)
        
        y.append(max(10000.0, price))
        
    return X, y

if __name__ == "__main__":
    X_train, y_train = generate_data(5)
    print([round(val, 1) for val in X_train[0]])
    print(f"${round(y_train[0], 2)}")
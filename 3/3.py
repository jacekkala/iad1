import numpy as np

X = np.array([
    [1, 0, 0, 0],
    [0, 1, 0, 0],
    [0, 0, 1, 0],
    [0, 0, 0, 1]
])
Z = X.copy()

def sigmoid(s):
    return 1.0 / (1.0 + np.exp(-s))

def sigmoid_deriv(y):
    return y * (1.0 - y)

np.random.seed(42)
W1 = np.random.uniform(-0.5, 0.5, (2, 4)) # warstwa ukryta (2 neurony, 4 wejścia)
W2 = np.random.uniform(-0.5, 0.5, (4, 3)) # warstwa wyjściowej (4 neurony, 2 wejścia + 1 bias)

eta = 0.5
epochs = 5000

for epoch in range(epochs):
    total_error = 0
    
    for p in range(4):
        x = X[p]
        z = Z[p]
        
        h_raw = np.dot(W1, x)
        
        # dodanie bias
        h = np.append(h_raw, 1.0)
        
        s = np.dot(W2, h)
        
        y = sigmoid(s)
        
        total_error += np.sum((z - y)**2)
        
        delta2 = sigmoid_deriv(y) * (z - y)
        
        # W2[:, 0:2] to wagi bez kolumny biasu - bias nie propaguje błędu wstecz
        delta1 = np.dot(W2[:, 0:2].T, delta2) 
        
        W1 += eta * np.dot(delta1.reshape(-1, 1), x.reshape(1, -1))
        W2 += eta * np.dot(delta2.reshape(-1, 1), h.reshape(1, -1))
        
    if (epoch + 1) % 1000 == 0:
        print(f"Epoka {epoch + 1:4d}, Błąd sumaryczny (SSE): {total_error:.6f}")

# test
print("\nWyniki po treningu:")
np.set_printoptions(formatter={'float': lambda x: "{0:0.4f}".format(x)})

for p in range(4):
    x = X[p]
    h = np.append(np.dot(W1, x), 1.0)
    y = sigmoid(np.dot(W2, h))
    print(f"Wejście: {x} -> Wyjście sieci: {y}")

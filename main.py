import matplotlib.pyplot as plt

from data import load_data
from train import train
from predict import predict

def accuracy(preds, Y):
    return (preds == Y).mean() * 100

X_train, Y_train, X_test, Y_test = load_data()

parameters, costs = train(
    X_train, Y_train,
    n_h=8,
    iterations=8000,
    learning_rate=0.01,
    print_cost=True
)

train_preds = predict(parameters, X_train)
test_preds = predict(parameters, X_test)

print("Train accuracy:", accuracy(train_preds, Y_train))
print("Test accuracy:", accuracy(test_preds, Y_test))

# Gradient descent cost graph
plt.plot(costs)
plt.ylabel("cost")
plt.xlabel("iterations (x100)")
plt.title("Cost reduction during gradient descent")
plt.show()

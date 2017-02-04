import numpy as np
import warnings
warnings.filterwarnings("error")

class LinearRegressor:
    def __init__(self, num_iteration, learning_rate, initial_w, initial_b):
        self.num_iteration = num_iteration
        self.learning_rate = learning_rate
        self.w = initial_w
        self.b = initial_b

    def train(self, data):
        for i in range(self.num_iteration):
            del_w, del_b = self.gradient_descent(data)
            self.w -= del_w
            self.b -= del_b

    def gradient_descent(self, data):
        total_gradient_w = 0
        total_gradient_b = 0

        try:
            for i in range(len(data)):
                x, y = data[i]
                total_gradient_w += 2 * x * self.error(x, y)
                total_gradient_b += 2 * self.error(x, y)
            total_gradient_w = total_gradient_w / float(len(data)) * self.learning_rate
            total_gradient_b = total_gradient_b / float(len(data)) * self.learning_rate
        except Warning:
            print("Current x : {0}, y : {1}, total_gradient_w : {2}".format(x, y, total_gradient_w))

        return total_gradient_w, total_gradient_b

    def predict(self, x):
        return self.w * x + self.b

    def error(self, x, y):
        return self.w * x + self.b - y

    def computeMSE(self, data):
        total_error = 0
        for i in range(len(data)):
            x, y = data[i]
            total_error += self.error(x, y) ** 2
        return 0.5 * total_error / float(len(data))


def run():
    data = np.genfromtxt('data.csv', delimiter=",")

    # define Hyper Parameter
    num_iteration = 1000
    learning_rate = 0.0001
    initial_w = 0
    imitial_b = 0

    model = LinearRegressor(num_iteration, learning_rate, initial_w, imitial_b)
    mse = model.computeMSE(data)
    print ("Before training, mean summery error is: {0}".format(mse))

    model.train(data)
    mse = model.computeMSE(data)
    print ("After training, mean summery error is: {0}".format(mse))
    print ("After training, W is: {0},  b is: {1}".format(model.w, model.b))


run()

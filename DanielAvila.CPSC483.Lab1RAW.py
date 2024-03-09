import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras

def generate_data(num_points=1000):
    xValues = np.linspace(-100, 100, num_points)
    yValues = xValues * np.sin(xValues ** 2 / 300)
    return xValues, yValues

#generate the data
x_train, y_train = generate_data()


def modelBuild_1():
    model = keras.Sequential([keras.layers.Dense(64, activation='relu', input_shape=(1,)),
                              keras.layers.Dense(64, activation='relu'),
                              keras.layers.Dense(1)
                             ])
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    return model

#splitting
x_train, y_train, x_test, y_test = train_test_split(x_train, y_train, test_size=0.6)

#training
model_1 = modelBuild_1()
history_1 = model_1.fit(x_train, y_train, epochs=100, validation_split=0.2, verbose=0)

def modelBuild_2():
    model = keras.Sequential([keras.layers.Dense(128, activation='relu', input_shape=(1,)),
                              keras.layers.Dense(64, activation='relu'),
                              keras.layers.Dense(32, activation='relu'),
                              keras.layers.Dense(1)
                             ])
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    return model

#splitting
x_train, y_train, x_test, y_test = train_test_split(x_train, y_train, test_size=0.6)

#training
model_2 = modelBuild_2()
history_2 = model_2.fit(x_train, y_train, epochs=100, validation_split=0.2, verbose=0)

def modelBuild_3():
    model = keras.Sequential([keras.layers.Dense(32, activation = 'relu', input_shape = (1,)),
                              keras.layers.Dense(64, activation = 'relu'),
                              keras.layers.Dense(32, activation = 'relu'),
                              keras.layers.Dense(1)
                             ])
    model.compile(optimizer = 'adam', loss = 'mse', metrics = ['mae'])
    return model

#splitting
x_train, y_train, x_test, y_test = train_test_split(x_train, y_train, test_size = 0.6)

#training
model_3 = modelBuild_3()
history_3 = model_3.fit(x_train, y_train, epochs = 100, validation_split = 0.2, verbose = 0)

def evaluateModel(model, x_test, y_test):
    loss, mae = model.evaluate(x_test, y_test, verbose = 0)
    print("Test Loss: ", loss)
    print("Test MAE: ", mae)

#evaluate models
print("Model 1: ")
evaluateModel(model_1, x_test, y_test)
print("\n")
print("Model 2: ")
evaluateModel(model_2, x_test, y_test)
print("\n")
print("Model 3: ")
evaluateModel(model_3, x_test, y_test)

#plotting predictions
def plotting(model, xValues, yValues, title):
    predictions = model.predict(xValues)
    plt.figure(figsize = (10,6))
    plt.scatter(xValues, yValues, label = 'Actual Data')
    plt.plot(xValues, predictions, color = 'red', label = 'Predictions')
    plt.title(title)
    plt.legend
    plt.show()

plotting(model_1, x_test, y_test, title = "Model 1 Predictions")
plotting(model_2, x_test, y_test, title = "Model 2 Predictions")
plotting(model_3, x_test, y_test, title = "Model 3 Predictions")

best_model = model_2
weights, biases = best_model.get_weights()
print("Weights Shape: ", weights.shape)
print("Biases Shape: ", biases.shape)

#5 data points
random_index = np.random.choice(len(x_train), 5)
x_sample = x_train[random_index]
output_manual = np.dot(x_sample, weights[0]) + biases[0]
output_manual = np.dot(output_manual, weights[2]) + biases[2]
print("Calculated Output: ", output_manual)

#compare
predictions = best_model.predict(x_sample)
print("Model Predictions: ", predictions)


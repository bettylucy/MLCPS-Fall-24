import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
# more imports
 
# Parse the file and return 2 numpy arrays
def load_data_set(filename):
    x=np.loadtxt(filename, usecols=(0,1), dtype='float')
    y=np.loadtxt(filename, usecols=2, dtype='float')
    return x, y 
 
def normal_equation(x, y):
    x_T=np.transpose(x)
    x_Tx=np.dot(x_T, x)
    x_T_y=np.dot(x_T, y)
    x_Tx_inv=np.linalg.inv(x_Tx)
    theta = np.dot(x_Tx_inv,x_T_y)
    return theta
 
# Given an array of x and theta predict y
def predict(x, theta):
    y_predict = np.dot(x, theta)
    return y_predict

# Given an array of y and y_predict return loss
def get_loss(y, y_predict):
    loss = np.mean((y - y_predict) ** 2)
    return loss
 
 
# Find thetas using stochiastic gradient descent
# Don't forget to shuffle
def stochiastic_gradient_descent(x, y, learning_rate, num_iterations):
    ind, lenx=x.shape
    thetas = np.random.randn(lenx)
    t_loss=[]
 
    for i in range (num_iterations):
        indices = np.random.permutation(ind)
        x_sh = x[indices]
        y_sh = y[indices]
        for j in range (ind):
            y_pred=predict(x_sh, thetas)
            error = y_pred - y_sh
            grad_thetas = np.dot(x_sh.T, error) / ind
            thetas-=learning_rate*(grad_thetas)
        t_loss.append(get_loss(y_sh, y_pred))
    return [thetas, t_loss]

# Find thetas using gradient descent
def gradient_descent(x, y, learning_rate, num_iterations):
    ind, lenx=x.shape
    thetas = np.random.randn(lenx)
    t_loss=[]
    
    for i in range (num_iterations):
        y_pred=predict(x, thetas)
        error = y_pred - y
        grad_thetas = np.dot(x.T, error) / ind
        thetas-=learning_rate*(grad_thetas)
        t_loss.append(get_loss(y, y_pred))
    return [thetas, t_loss]


# Find thetas using minibatch gradient descent
# Don't forget to shuffle
def minibatch_gradient_descent(x, y, learning_rate, num_iterations, batch_size):
    ind, lenx=x.shape
    thetas = np.random.randn(lenx)
    t_loss=[]
    
    for i in range (num_iterations):
        indices = np.random.permutation(ind)
        x_sh = x[indices]
        y_sh = y[indices]
        t_loss.append(0)
        for j in range (0, ind, batch_size):
            x_bt = x_sh[j:j+batch_size]
            y_bt = y_sh[j:j+batch_size]
            
            y_pred=predict(x_bt, thetas)
            error = y_pred - y_bt
            grad_thetas = np.dot(x_bt.T, error) / ind
            thetas-=learning_rate*(grad_thetas)
            
            t_loss[i]=get_loss(y_bt, y_pred)
        
    return [thetas, t_loss]

# Given a list of thetas one per epoch
# this creates a plot of epoch vs training error
def plot_training_errors(x, y, t_loss, title):
    epochs = []
    epoch = len(t_loss)
    losses = []
    epoch_num = 1
    for epoch_num in range (epoch):
        epochs.append(epoch_num)
        epoch_num += 1
    plt.plot(epochs, t_loss)
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.title(title)
    plt.show()

# Given x, y, y_predict and title,
# this creates a plot
def plot(x, y, theta, title):
    # plot
    y_predict = predict(x, theta)
    plt.scatter(x[:, 1], y)
    plt.plot(x[:, 1], y_predict)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title(title)
    plt.show()

if __name__ == "__main__":
    x, y = load_data_set('regression-data.txt')
    # plot
    plt.scatter(x[:, 1], y)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("Scatter Plot of Data")
    plt.show()

    theta = normal_equation(x, y)
    plot(x, y, theta, "Normal Equation Best Fit")

    # You should try multiple non-zero learning rates and  multiple different (non-zero) number of iterations
    thetas, t_loss = gradient_descent(x, y, 0.01, 100) 
    plot(x, y, thetas, "Gradient Descent Best Fit")
    plot_training_errors(x, y, t_loss, "Gradient Descent Mean Epoch vs Training Loss")

    # You should try multiple non-zero learning rates and  multiple different (non-zero) number of iterations
    thetas, t_loss = stochiastic_gradient_descent(x, y, 0.001, 100) # Try different learning rates and number of iterations
    plot(x, y, thetas, "Stochiastic Gradient Descent Best Fit")
    plot_training_errors(x, y, t_loss, "Stochiastic Gradient Descent Mean Epoch vs Training Loss")

    # You should try multiple non-zero learning rates and  multiple different (non-zero) number of iterations
    thetas, t_loss = minibatch_gradient_descent(x, y, 0.01, 100, 10)
    plot(x, y, thetas, "Minibatch Gradient Descent Best Fit")
    plot_training_errors(x, y, t_loss, "Minibatch Gradient Descent Mean Epoch vs Training Loss")



batch=[5,10,15,20,40,50]

# Code for plotting different interations in the same figure (best fit)
x, y = load_data_set('regression-data.txt')
i=0
for batches in (batch):
    thetas, t_loss = minibatch_gradient_descent(x, y, 0.3, 100, batches)
    y_predict = np.dot(x, thetas)
    plt.plot(x[:, 1], y_predict, label=str(batches))
    print(thetas)

plt.scatter(x[:, 1], y)
plt.xlabel("x")
plt.ylabel("y")
plt.title("Minibatch Gradient Descent Best Fit")
plt.legend(loc = 'best')
plt.show()

## Code for plotting different interations in the same figure (training loss)
x, y = load_data_set('regression-data.txt')
i=0
for batches in (batch):
    thetas, t_loss = minibatch_gradient_descent(x, y, 0.3, 100, batches)
    epochs = []
    epoch = len(t_loss)
    losses = []
    epoch_num = 1
    for epoch_num in range (epoch):
        epochs.append(epoch_num)
        epoch_num += 1
    plt.plot(epochs, t_loss, label=str(batches))
    print (round(t_loss[99],4))

plt.legend(loc = 'best')
plt.xlabel("epoch")
plt.ylabel("loss")
plt.title("Minibatch Gradient Descent Mean Epoch vs Training Loss")
plt.show()


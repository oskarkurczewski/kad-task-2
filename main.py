import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math

print("<-----\tTASK INFO\t----->\n\nTask: 2\nAuthor: Oskar Kurczewski\nIndex number: 229935")


# read csv files and add them to a pandas dataframe

data1 = pd.read_csv("data1.csv", names = ["X", "Y"])
data2 = pd.read_csv("data2.csv", names = ["X", "Y"])
data3 = pd.read_csv("data3.csv", names = ["X1", "X2", "Y"])
data4 = pd.read_csv("data4.csv", names = ["X1", "X2", "Y"])


# variance - dataframe version

def variance(dt):
    mean = dt.mean()
    var = 0

    for i in range(len(dt)):
        var += ((dt[i] - mean) * (dt[i] - mean))

    return var / len(dt)


# variance - list version

def variance(list):
    mean = sum(list)/len(list)
    var = 0

    for i in range(len(list)):
        var += ((list[i] - mean) * (list[i] - mean))

    return var / len(list)


# covariance

def covariance(dt):
    meanx = dt.X.mean()
    meany = dt.Y.mean()
    cov = 0

    for i in range(len(dt)):
        cov += ((dt.X[i] - meanx) * (dt.Y[i] - meany))

    return cov / len(dt)


# error

def error(dt, y):
    err = []
    for i in range(len(dt)):
        err.append(dt.Y[i] - y[i])
    return err


# mse

def meansquareerror(expected, measured):
    mse = 0
    for i in range(len(expected)):
        mse += ((expected[i] - measured[i])**2)
    return mse / len(expected)


# R**2

def rsquared(err, y):
    fuv = variance(err) / variance(y)
    mse = 0
    for i in range(len(y)):
        mse += (1 - fuv)
    return mse / len(y)


# maximum deviation

def maxdeviation(expected, measured):
    deviations = []
    for i in range(len(expected)):
        deviations.append(math.fabs((expected[i] - measured[i])))
    return max(deviations)


# model 1: f(x) = a * X

def model1figure(dt):
    meanx = dt.X.mean()
    meany = dt.Y.mean()
    a = meany / meanx
    print('Model 1: f(x) = ' + str(round(a, 3)) + " * X")


# returns an array of expected values

def model1function(dt):
    meanx = dt.X.mean()
    meany = dt.Y.mean()
    a = meany / meanx
    y = []
    for i in range(len(dt)):
        y.append(a*dt.X[i])
    return y


# model 2: f(x) = a * X + b

def model2figure(dt):
    meanx = dt.X.mean()
    meany = dt.Y.mean()
    a = covariance(dt) / variance(dt.X)
    b = meany - a * meanx
    print('Model 2: f(x) = ' + str(round(a, 3)) + " * X + " + str(round(b, 3)))

def model2function(dt):
    meanx = dt.X.mean()
    meany = dt.Y.mean()
    a = covariance(dt) / variance(dt.X)
    b = meany - a * meanx
    y = []
    for i in range(len(dt)):
        y.append(a * dt.X[i] + b)
    return y


# model 3: f(x) = a * X**2 + b * sin(X) + c

def model3figure(dt):
    x = np.array([np.ones(100), dt.X**2, np.sin(dt.X)]).reshape(3, len(dt.X)).T
    y = np.array([dt.Y]).T
    inverted = np.linalg.pinv(np.matmul(x.T, x))
    xty = np.matmul(x.T, y)
    z = np.matmul(inverted, xty).T
    a = z[0, 1]
    b = z[0, 2]
    c = z[0, 0]
    print('Model 3: f(x) = ' + str(round(a, 3)) + " * X**2 + " + str(round(b, 3)) + " * sin(X) + ", str(round(c, 3)))

def model3function(dt):
    x = np.array([np.ones(100), dt.X**2, np.sin(dt.X)]).reshape(3, len(dt.X)).T
    y = np.array([dt.Y]).T
    inverted = np.linalg.pinv(np.matmul(x.T, x))
    xty = np.matmul(x.T, y)
    z = np.matmul(inverted, xty).T
    a = z[0, 1]
    b = z[0, 2]
    c = z[0, 0]
    result = a * (dt.X**2) + b * (np.sin(dt.X)) + c
    return result


# model 4: f(X1, X2) = a * X1 + b * X2 + c

def model4figure(dt):
    x = np.array([np.ones(100), dt.X1, np.sin(dt.X2)]).reshape(3, len(dt.X1)).T
    y = np.array([dt.Y]).T
    inverted = np.linalg.pinv(np.matmul(x.T, x))
    xty = np.matmul(x.T, y)
    z = np.matmul(inverted, xty).T
    a = z[0, 1]
    b = z[0, 2]
    c = z[0, 0]
    print('Model 3: f(x) = ' + str(round(a, 3)) + " * X1 + " + str(round(b, 3)) + " * X2 + ", str(round(c, 3)))

def model4function(dt):
    x = np.array([np.ones(100), dt.X1, np.sin(dt.X2)]).reshape(3, len(dt.X1)).T
    y = np.array([dt.Y]).T
    inverted = np.linalg.pinv(np.matmul(x.T, x))
    xty = np.matmul(x.T, y)
    z = np.matmul(inverted, xty).T
    a = z[0, 1]
    b = z[0, 2]
    c = z[0, 0]
    result = a * dt.X1 + b * dt.X2 + c
    return result


# model 5: f(X1, X2) = a * X1**2 + b * X1*X2 + c * X2**2 + d * X1 + e * X2 + f

def model5figure(dt):
    x = np.array([np.ones(100), dt.X1**2, dt.X1*dt.X2, dt.X2**2, dt.X1, dt.X2]).reshape(6, len(dt.X1)).T
    y = np.array([dt.Y]).T
    inverted = np.linalg.pinv(np.matmul(x.T, x))
    xty = np.matmul(x.T, y)
    z = np.matmul(inverted, xty).T
    a = z[0, 1]
    b = z[0, 2]
    c = z[0, 3]
    d = z[0, 4]
    e = z[0, 5]
    f = z[0, 0]
    print('Model 3: f(x) = ' + str(round(a, 3)) + " * X1**2 + " + str(round(b, 3)) + " * X1*X2 + ", str(round(c, 3)), " * X2**2 + ", str(round(d, 3)), " * X1 + ", str(round(e, 3)), " * X2 + ", str(round(f, 3)))

def model5function(dt):
    x = np.array([np.ones(100), dt.X1**2, dt.X1*dt.X2, dt.X2**2, dt.X1, dt.X2]).reshape(6, len(dt.X1)).T
    y = np.array([dt.Y]).T
    inverted = np.linalg.pinv(np.matmul(x.T, x))
    xty = np.matmul(x.T, y)
    z = np.matmul(inverted, xty).T
    a = z[0, 1]
    b = z[0, 2]
    c = z[0, 3]
    d = z[0, 4]
    e = z[0, 5]
    f = z[0, 0]
    result = a * dt.X1**2 + b * dt.X1*dt.X2 + c * dt.X2**2 + d * dt.X1 + e * dt.X2 + f
    return result

def printall():
    print("\n<-----\tDATA SET 1\t----->\n")

    model1figure(data1)
    print("Mean square error - model 1: " + str(round(meansquareerror(model1function(data1), data1.Y), 3)))
    print("Maximum deviation - model 1: " + str(round(maxdeviation(model1function(data1), data1.Y), 3)))
    print("R**2 - model 1: " + str(round(rsquared(error(data1, model1function(data1)), model1function(data1)), 3)) + "\n")

    model2figure(data1)
    print("Mean square error - model 2: " + str(round(meansquareerror(model2function(data1), data1.Y), 3)))
    print("Maximum deviation - model 2: " + str(round(maxdeviation(model2function(data1), data1.Y), 3)))
    print("R**2 - model 2: " + str(round(rsquared(error(data1, model2function(data1)), model2function(data1)), 3)) + "\n")

    model3figure(data1)
    print("Mean square error - model 3: " + str(round(meansquareerror(model3function(data1), data1.Y), 3)))
    print("Maximum deviation - model 3: " + str(round(maxdeviation(model3function(data1), data1.Y), 3)))
    print("R**2 - model 3: " + str(round(rsquared(error(data1, model3function(data1)), model3function(data1)), 3)) + "\n")

    print("<-----\tDATA SET 2\t----->\n")

    model1figure(data2)
    print("Mean square error - model 1: " + str(round(meansquareerror(model1function(data2), data2.Y), 3)))
    print("Maximum deviation - model 1: " + str(round(maxdeviation(model1function(data2), data2.Y), 3)))
    print("R**2 - model 1: " + str(round(rsquared(error(data2, model1function(data2)), model1function(data2)), 3)) + "\n")

    model2figure(data2)
    print("Mean square error - model 2: " + str(round(meansquareerror(model2function(data2), data2.Y), 3)))
    print("Maximum deviation - model 2: " + str(round(maxdeviation(model2function(data2), data2.Y), 3)))
    print("R**2 - model 2: " + str(round(rsquared(error(data2, model2function(data2)), model2function(data2)), 3)) + "\n")

    model3figure(data2)
    print("Mean square error - model 3: " + str(round(meansquareerror(model3function(data2), data2.Y), 3)))
    print("Maximum deviation - model 3: " + str(round(maxdeviation(model3function(data2), data2.Y), 3)))
    print("R**2 - model 3: " + str(round(rsquared(error(data2, model3function(data2)), model3function(data2)), 3)) + "\n")

    print("<-----\tDATA SET 3\t----->\n")

    model4figure(data3)
    print("Mean square error - model 4: " + str(round(meansquareerror(model4function(data3), data3.Y), 3)))
    print("Maximum deviation - model 4: " + str(round(maxdeviation(model4function(data3), data3.Y), 3)))
    print("R**2 - model 4: " + str(round(rsquared(error(data3, model4function(data3)), model4function(data3)), 3)) + "\n")

    model5figure(data3)
    print("Mean square error - model 4: " + str(round(meansquareerror(model5function(data3), data3.Y), 3)))
    print("Maximum deviation - model 4: " + str(round(maxdeviation(model5function(data3), data3.Y), 3)))
    print("R**2 - model 4: " + str(round(rsquared(error(data3, model5function(data3)), model5function(data3)), 3)) + "\n")

    print("<-----\tDATA SET 4\t----->\n")

    model4figure(data4)
    print("Mean square error - model 4: " + str(round(meansquareerror(model4function(data4), data4.Y), 3)))
    print("Maximum deviation - model 4: " + str(round(maxdeviation(model4function(data4), data4.Y), 3)))
    print("R**2 - model 4: " + str(round(rsquared(error(data4, model4function(data4)), model4function(data4)), 3)) + "\n")

    model5figure(data4)
    print("Mean square error - model 4: " + str(round(meansquareerror(model5function(data4), data4.Y), 3)))
    print("Maximum deviation - model 4: " + str(round(maxdeviation(model5function(data4), data4.Y), 3)))
    print("R**2 - model 4: " + str(round(rsquared(error(data4, model5function(data4)), model5function(data4)), 3)) + "\n")

printall()

# function plotting a dataframe

# def twoDimensionGraph(dt, n):
#     plt.figure(n)
#     dt["X"] = pd.Series(list(range(len(dt))))
#     dt.plot(x = "X", y = "Y", color = 'red', marker = 'o', linewidth = '0', markersize = '2')
#     plt.savefig(fname = n.__str__())
#     plt.close(n)
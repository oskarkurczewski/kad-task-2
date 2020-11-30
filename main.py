import pandas as pd
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

# returns an array of expected values

def model2function(dt):
    meanx = dt.X.mean()
    meany = dt.Y.mean()
    a = covariance(dt) / variance(dt.X)
    b = meany - a * meanx
    y = []
    for i in range(len(dt)):
        y.append(a * dt.X[i] + b)
    return y


print("\n<-----\tDATA SET 1\t----->\n")

model1figure(data1)
print("Mean square error - model 1: " + str(round(meansquareerror(model1function(data1), data1.Y), 3)))
print("Maximum deviation - model 1: " + str(round(maxdeviation(model1function(data1), data1.Y), 3)))
print("R**2 - model 1: " + str(round(rsquared(error(data1, model1function(data1)), model1function(data1)), 3)) + "\n")

model2figure(data1)
print("Mean square error - model 2: " + str(round(meansquareerror(model2function(data1), data1.Y), 3)))
print("Maximum deviation - model 2: " + str(round(maxdeviation(model2function(data1), data1.Y), 3)))
print("R**2 - model 2: " + str(round(rsquared(error(data1, model2function(data1)), model2function(data1)), 3)) + "\n")


print("<-----\tDATA SET 2\t----->\n")

model1figure(data2)
print("Mean square error - model 1: " + str(round(meansquareerror(model1function(data2), data2.Y), 3)))
print("Maximum deviation - model 1: " + str(round(maxdeviation(model1function(data2), data2.Y), 3)))
print("R**2 - model 1: " + str(round(rsquared(error(data2, model1function(data2)), model1function(data2)), 3)) + "\n")

model2figure(data2)
print("Mean square error - model 2: " + str(round(meansquareerror(model2function(data2), data2.Y), 3)))
print("Maximum deviation - model 2: " + str(round(maxdeviation(model2function(data2), data2.Y), 3)))
print("R**2 - model 2: " + str(round(rsquared(error(data2, model2function(data2)), model2function(data2)), 3)) + "\n")


# function plotting a dataframe

# def twoDimensionGraph(dt, n):
#     plt.figure(n)
#     dt["X"] = pd.Series(list(range(len(dt))))
#     dt.plot(x = "X", y = "Y", color = 'red', marker = 'o', linewidth = '0', markersize = '2')
#     plt.savefig(fname = n.__str__())
#     plt.close(n)
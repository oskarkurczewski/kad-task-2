import pandas as pd
import matplotlib.pyplot as plt

print("Task 2!\n")

# read csv files and add them to a pandas dataframe

data1 = pd.read_csv("data1.csv", names = ["X", "Y"])
data2 = pd.read_csv("data2.csv", names = ["X", "Y"])
data3 = pd.read_csv("data3.csv", names = ["X1", "X2", "Y"])
data4 = pd.read_csv("data4.csv", names = ["X1", "X2", "Y"])

# variance

def variance(dt):
    meanx = dt.X.mean()
    var = 0

    for x in range(100):
        var += ((dt.X - meanx) * (dt.X - meanx))

    return var.mean()

# covariance

def covariance(dt):
    meanx = dt.X.mean()
    meany = dt.Y.mean()
    cov = 0

    for x in range(100):
        cov += ((dt.X - meanx) * (dt.Y - meany))

    return cov.mean()

# model 1: f(x) = a * X

def model1(dt):
    meanx = dt.X.mean()
    meany = dt.Y.mean()
    a = meany / meanx
    print('f(x) = ' + str(round(a, 2)) + " * X")

# model 2: f(x) = a * X + b

def model2(dt):
    meanx = dt.X.mean()
    meany = dt.Y.mean()
    a = covariance(dt) / variance(dt)
    b = meany - a * meanx
    print('f(x) = ' + str(round(a, 2)) + " * X + " + str(round(b, 2)))

print("\n--- ZESTAW DANYCH 1 ---")

model1(data1)
model2(data1)

print("\n--- ZESTAW DANYCH 2 ---")

model1(data2)
model2(data2)


# function plotting a dataframe

def twoDimensionGraph(dt, n):
    plt.figure(n)
    dt["X"] = pd.Series(list(range(len(dt))))
    dt.plot(x = "X", y = "Y", color = 'red', marker = 'o', linewidth = '0', markersize = '2')
    plt.savefig(fname = n.__str__())
    plt.close(n)

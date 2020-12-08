import numpy as np
import matplotlib.pyplot as plt
import math

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
        mse += ((measured[i] - expected[i])**2)
    return mse / len(expected)

# R**2

def rsquared(err, y):
    fuv = variance(err) / variance(y)
    r = 0
    for i in range(len(y)):
        r += (1 - fuv)
    return r / len(y)

# maximum deviation

def maxdeviation(expected, measured):
    deviations = []
    for i in range(len(expected)):
        deviations.append(math.fabs((measured[i] - expected[i])))
    return max(deviations)


# model 1: f(x) = a * X

def model1figure(dt):
    meanx = dt.X.mean()
    meany = dt.Y.mean()
    a = meany / meanx
    print('Model 1: f(x) =', str(round(a, 3)), "* X")

# returns an array of expected values

def model1function(dt):
    meanx = dt.X.mean()
    meany = dt.Y.mean()
    a = meany / meanx
    y = []
    for i in range(len(dt)):
        y.append(a*dt.X[i])
    return y

# plots expected and measured values

def model1plot(dt):
    plt.scatter(dt.X, dt.Y)
    plt.title("Model 2: f(X) = a \u22c5 X")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.plot(dt.X, model1function(dt), color= "orange")

# plots a histogram of deviations

def model1histogram(dt):
    plt.hist(error(dt, model1function(dt)), color = "orange", bins = 20)
    plt.title("Model 2: f(X) = a \u22c5 X")
    plt.savefig(fname = dt.name + "_model_1_histogram.png", dpi = 300)

# model 2: f(x) = a * X + b

def model2figure(dt):
    meanx = dt.X.mean()
    meany = dt.Y.mean()
    a = covariance(dt) / variance(dt.X)
    b = meany - a * meanx
    print('Model 2: f(x) =', str(round(a, 3)),
                    "* X +", str(round(b, 3)))

def model2function(dt):
    meanx = dt.X.mean()
    meany = dt.Y.mean()
    a = covariance(dt) / variance(dt.X)
    b = meany - a * meanx
    y = []
    for i in range(len(dt)):
        y.append(a * dt.X[i] + b)
    return y

def model2plot(dt):
    plt.scatter(dt.X, dt.Y)
    plt.title("Model 2: f(X) = a \u22c5 X + b")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.plot(dt.X, model2function(dt), color= "blue")

def model2histogram(dt):
    plt.hist(error(dt, model2function(dt)), color = "blue", bins = 20)
    plt.title("Model 2: f(X) = a \u22c5 X + b")
    plt.savefig(fname = dt.name + "_model_2_histogram.png", dpi = 300)

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
    print('Model 3: f(x) =', str(round(a, 3)),
                  "* X**2 +", str(round(b, 3)),
               "* sin(X) +", str(round(c, 3)))

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

def model3plot(dt):
    plt.scatter(dt.X, dt.Y)
    plt.plot(dt.X, model3function(dt), color= "red")
    plt.title("Model 3: f(X) = a \u22c5 X\u00b2 + b \u22c5 sin(X) + c")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.savefig(fname = dt.name + "_chart.png", dpi = 300)

def model3histogram(dt):
    plt.hist(error(dt, model3function(dt)), color = "red", bins = 20)
    plt.title("Model 3: f(X) = a \u22c5 X\u00b2 + b \u22c5 sin(X) + c")
    plt.savefig(fname = dt.name + "_model_3_histogram.png", dpi = 300)

# model 4: f(X1, X2) = a * X1 + b * X2 + c

def model4figure(dt):
    x = np.array([np.ones(100), dt.X1, dt.X2]).reshape(3, len(dt.X1)).T
    y = np.array([dt.Y]).T
    inverted = np.linalg.pinv(np.matmul(x.T, x))
    xty = np.matmul(x.T, y)
    z = np.matmul(inverted, xty).T
    a = z[0, 1]
    b = z[0, 2]
    c = z[0, 0]
    print('Model 4: f(x) =', str(round(a, 3)),
                   "* X1 +", str(round(b, 3)),
                   "* X2 +", str(round(c, 3)))

def model4function(dt):
    x = np.array([np.ones(100), dt.X1, dt.X2]).reshape(3, len(dt.X1)).T
    y = np.array([dt.Y]).T
    inverted = np.linalg.pinv(np.matmul(x.T, x))
    xty = np.matmul(x.T, y)
    z = np.matmul(inverted, xty).T
    a = z[0, 1]
    b = z[0, 2]
    c = z[0, 0]
    result = a * dt.X1 + b * dt.X2 + c
    return result

def model4plot(dt):
    x = np.array([np.ones(100), dt.X1, dt.X2]).reshape(3, len(dt.X1)).T
    y = np.array([dt.Y]).T
    inverted = np.linalg.pinv(np.matmul(x.T, x))
    xty = np.matmul(x.T, y)
    z = np.matmul(inverted, xty).T
    a = z[0, 1]
    b = z[0, 2]
    c = z[0, 0]

    fig = plt.figure()
    ax = fig.add_subplot(111, projection = "3d")
    ax.scatter(dt.X2, dt.Y, dt.X1, c = "red")
    zz = np.array([])
    xx, yy = np.meshgrid(np.arange(0, 7, 0.7), np.arange(0, 7, 0.7))
    for i, xs in enumerate(xx):
        for j, ys in enumerate(yy):
            zz = np.append(zz, a * xs[i] + b * ys[j] + c)
    zz = zz.reshape(np.size(xx, 0), np.size(yy, 0))
    ax.plot_wireframe(xx, zz, yy)
    ax.set_xlabel('X1', fontsize = 20)
    ax.set_ylabel('X2', fontsize = 20)
    ax.set_zlabel('f(X1, X2)', fontsize = 20)
    plt.title("Model 4: f(X\u2081, X\u2082) = a \u22c5 X\u2081 + b \u22c5 X\u2082 + c")
    plt.savefig(fname = dt.name + "_model_4_plot.png", dpi = 300)

def model4histogram(dt):
    plt.hist(error(dt, model4function(dt)), color = "red", bins = 20)
    plt.title("Model 4: f(X\u2081, X\u2082) = a \u22c5 X\u2081 + b \u22c5 X\u2082 + c")
    plt.savefig(fname = dt.name + "_model_4_histogram.png", dpi = 300)

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
    print('Model 5: f(x) =', str(round(a, 3)),
                 "* X1^2 +", str(round(b, 3)),
                "* X1*X2 +", str(round(c, 3)),
                 "* X2^2 +", str(round(d, 3)),
                   "* X1 +", str(round(e, 3)),
                   "* X2 +", str(round(f, 3)))

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

def model5plot(dt):
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

    fig = plt.figure()
    ax = fig.add_subplot(111, projection = "3d")
    ax.scatter(dt.X2, dt.Y, dt.X1, c = "purple")
    zz = np.array([])
    xx, yy = np.meshgrid(np.arange(0, 7, 0.5), np.arange(0, 7, 0.5))
    for i, xs in enumerate(xx):
        for j, ys in enumerate(yy):
            zz = np.append(zz, a * xs[i]**2 + b * xs[i]*ys[j] + c * ys[j]**2 + d * xs[i] + e * ys[j] + f)
    zz = zz.reshape(np.size(xx, 0), np.size(yy, 0))
    ax.plot_wireframe(xx, zz, yy)
    ax.set_xlabel('X1', fontsize = 20)
    ax.set_ylabel('X2', fontsize = 20)
    ax.set_zlabel('f(X1, X2)', fontsize = 20)
    plt.title("Model 5: f(X\u2081, X\u2082) = a \u22c5 X\u2081\u00b2 + b \u22c5 X\u2081\u22c5X\u2082 + c \u22c5 X\u2081\u00b2 + d \u22c5 X\u2081 + e \u22c5 X\u2082 + f")
    plt.savefig(fname = dt.name + "_model_5_plot.png", dpi = 300)

def model5histogram(dt):
    plt.hist(error(dt, model5function(dt)), color = "purple", bins = 20)
    plt.title("Model 5: f(X\u2081, X\u2082) = a \u22c5 X\u2081\u00b2 + b \u22c5 X\u2081\u22c5X\u2082 + c \u22c5 X\u2081\u00b2 + d \u22c5 X\u2081 + e \u22c5 X\u2082 + f")
    plt.savefig(fname = dt.name + "_model_5_histogram.png", dpi = 300)

# plotting all histograms and graphs

def plotall(data1, data2, data3, data4):

    # data 1

    model1plot(data1)
    model2plot(data1)
    model3plot(data1)
    plt.close()
    model1histogram(data1)
    plt.close()
    model2histogram(data1)
    plt.close()
    model3histogram(data1)
    plt.close()

    # data 2

    model1plot(data2)
    model2plot(data2)
    model3plot(data2)
    plt.close()
    model1histogram(data2)
    plt.close()
    model2histogram(data2)
    plt.close()
    model3histogram(data2)
    plt.close()

    # data 3

    model4plot(data3)
    plt.close()
    model5plot(data3)
    plt.close()
    model4histogram(data3)
    plt.close()
    model5histogram(data3)
    plt.close()

    # data 4

    model4plot(data4)
    plt.close()
    model5plot(data4)
    plt.close()
    model4histogram(data4)
    plt.close()
    model5histogram(data4)
    plt.close()

# printing all needed parameters and values

def printall(data1, data2, data3, data4):
    print("\n<-----\tDATA SET 1\t----->\n")

    model1figure(data1)
    print("Mean square error - model 1: ",
          str(round(meansquareerror(model1function(data1), data1.Y), 3)))
    print("Maximum deviation - model 1: ",
          str(round(maxdeviation(model1function(data1), data1.Y), 3)))
    print("R**2 - model 1: ",
          str(round(rsquared(error(data1, model1function(data1)), model1function(data1)), 3)) + "\n")

    model2figure(data1)
    print("Mean square error - model 2: ",
          str(round(meansquareerror(model2function(data1), data1.Y), 3)))
    print("Maximum deviation - model 2: ",
          str(round(maxdeviation(model2function(data1), data1.Y), 3)))
    print("R**2 - model 2: ",
          str(round(rsquared(error(data1, model2function(data1)), model2function(data1)), 3)) + "\n")

    model3figure(data1)
    print("Mean square error - model 3: ",
          str(round(meansquareerror(model3function(data1), data1.Y), 3)))
    print("Maximum deviation - model 3: ",
          str(round(maxdeviation(model3function(data1), data1.Y), 3)))
    print("R**2 - model 3: ",
          str(round(rsquared(error(data1, model3function(data1)), model3function(data1)), 3)) + "\n")

    print("<-----\tDATA SET 2\t----->\n")

    model1figure(data2)
    print("Mean square error - model 1: ",
          str(round(meansquareerror(model1function(data2), data2.Y), 3)))
    print("Maximum deviation - model 1: ",
          str(round(maxdeviation(model1function(data2), data2.Y), 3)))
    print("R**2 - model 1: ",
          str(round(rsquared(error(data2, model1function(data2)), model1function(data2)), 3)) + "\n")

    model2figure(data2)
    print("Mean square error - model 2: ",
          str(round(meansquareerror(model2function(data2), data2.Y), 3)))
    print("Maximum deviation - model 2: ",
          str(round(maxdeviation(model2function(data2), data2.Y), 3)))
    print("R**2 - model 2: ",
          str(round(rsquared(error(data2, model2function(data2)), model2function(data2)), 3)) + "\n")

    model3figure(data2)
    print("Mean square error - model 3: ",
          str(round(meansquareerror(model3function(data2), data2.Y), 3)))
    print("Maximum deviation - model 3: ",
          str(round(maxdeviation(model3function(data2), data2.Y), 3)))
    print("R**2 - model 3: ",
          str(round(rsquared(error(data2, model3function(data2)), model3function(data2)), 3)) + "\n")

    print("<-----\tDATA SET 3\t----->\n")

    model4figure(data3)
    print("Mean square error - model 4: ",
          str(round(meansquareerror(model4function(data3), data3.Y), 3)))
    print("Maximum deviation - model 4: ",
          str(round(maxdeviation(model4function(data3), data3.Y), 3)))
    print("R**2 - model 4: " + str(round(rsquared(error(data3, model4function(data3)), model4function(data3)), 3)) + "\n")

    model5figure(data3)
    print("Mean square error - model 5: ",
          str(round(meansquareerror(model5function(data3), data3.Y), 3)))
    print("Maximum deviation - model 5: ",
          str(round(maxdeviation(model5function(data3), data3.Y), 3)))
    print("R**2 - model 5: ",
          str(round(rsquared(error(data3, model5function(data3)), model5function(data3)), 3)) + "\n")

    print("<-----\tDATA SET 4\t----->\n")

    model4figure(data4)
    print("Mean square error - model 4: ",
          str(round(meansquareerror(model4function(data4), data4.Y), 3)))
    print("Maximum deviation - model 4: ",
          str(round(maxdeviation(model4function(data4), data4.Y), 3)))
    print("R**2 - model 4: ",
          str(round(rsquared(error(data4, model4function(data4)), model4function(data4)), 3)) + "\n")

    model5figure(data4)
    print("Mean square error - model 5: ",
          str(round(meansquareerror(model5function(data4), data4.Y), 3)))
    print("Maximum deviation - model 5: ",
          str(round(maxdeviation(model5function(data4), data4.Y), 3)))
    print("R**2 - model 5: ",
          str(round(rsquared(error(data4, model5function(data4)), model5function(data4)), 3)) + "\n")
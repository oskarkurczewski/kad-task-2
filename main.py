import pandas as pd
import matplotlib.pyplot as plt

print("Task 2!\n")

# read csv files and add them to a pandas dataframe

data1 = pd.read_csv("data1.csv", names = ["X", "Y"])
data2 = pd.read_csv("data2.csv", names = ["X", "Y"])
data3 = pd.read_csv("data3.csv", names = ["X1", "X2", "Y"])
data4 = pd.read_csv("data4.csv", names = ["X1", "X2", "Y"])

# function plotting a dataframe

def twoDimensionGraph(dt, n):
    plt.figure(n)
    dt["X"] = pd.Series(list(range(len(dt))))
    dt.plot(x = "X", y = "Y", color = 'red', marker = 'o', linewidth = '0', markersize = '2')
    plt.savefig(fname = n.__str__())
    plt.close(n)

# calling a plotting function for dataframes 1 and 2

twoDimensionGraph(data1, 1)
twoDimensionGraph(data2, 2)

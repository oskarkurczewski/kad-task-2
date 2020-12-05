import pandas as pd
import functions as f

print("<-----\tTASK INFO\t----->\n\nTask: 2\nAuthor: Oskar Kurczewski\nIndex number: 229935")

# read csv files and add them to a pandas dataframe

data1 = pd.read_csv("data1.csv", names = ["X", "Y"])
data1.name = "data1"
data2 = pd.read_csv("data2.csv", names = ["X", "Y"])
data2.name = "data2"
data3 = pd.read_csv("data3.csv", names = ["X1", "X2", "Y"])
data3.name = "data3"
data4 = pd.read_csv("data4.csv", names = ["X1", "X2", "Y"])
data4.name = "data4"

# call a function that prints for each model:
# - parameters values
# - mean square errors
# - greatest deviation values
# - R**2 parameter values

f.printall(data1, data2, data3, data4)

# call a function that exports plots for each model:

f.plotall(data1, data2, data3, data4)
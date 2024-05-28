import numpy as np
import pandas as pd

data = pd.read_csv("data.csv")
print(data)

suppliers = data["Supplier"].values
print(suppliers, "\n")

attributes = data.columns[1:].to_numpy()
print(attributes, "\n")

raw_data = np.genfromtxt("data.csv", delimiter=",", skip_header=1, usecols=range(1, 6))
print(raw_data, "\n")

# (["Cost", "Quality", "Delivery Time", "Reliability", "Environmental Impact"])
weights = np.array([0.3, 0.25, 0.2, 0.15, 0.1])
print(weights)

import numpy as np
import pandas as pd
import topsis

data = pd.read_csv("data.csv")
print(data)

suppliers = data.iloc[:, 0].values
# print(suppliers, "\n")

attributes = data.columns[1:].to_numpy()
# print(attributes, "\n")

raw_data = np.genfromtxt("data.csv", delimiter=",", skip_header=1, usecols=range(1, len(attributes)+1))
# print(raw_data, "\n")

# (["Cost", "Quality", "Delivery Time", "Reliability", "Environmental Impact"]) -> [C, B, C, B, C]
criterias = ["C", "B", "C", "B", "C"] # Cost x Benefit
weights = np.array([0.3, 0.25, 0.2, 0.15, 0.1])

score = topsis.topsis(raw_data, criterias, weights)

ranking = topsis.generate_ranking(score, suppliers)

topsis.print_result(ranking)

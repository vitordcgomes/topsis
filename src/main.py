import topsis

input_path = "../input/data.csv"
data = topsis.read_data(input_path)
print(data)

suppliers = topsis.get_objects_from_data(data)
attributes = topsis.get_attributes_from_data(data)

ncols = data.shape[1]
raw_data = topsis.get_raw_data(input_path, ncols)

# (["Cost", "Quality", "Delivery Time", "Reliability", "Environmental Impact"]) -> [C, B, C, B, C] Cost x Benefit
json = topsis.read_json_data("../input/data.json") 

criterias = json["criterias"]
weights = json["weights"]

score = topsis.topsis(raw_data, criterias, weights)

ranking = topsis.generate_ranking(score, suppliers)

topsis.print_result(ranking)

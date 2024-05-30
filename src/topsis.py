import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import json
import os

def read_data(path):
    if os.path.exists(path):
        data = pd.read_csv(path)
        return data
    
    else:
        print("Data Path does not exist!")
        exit(0)
    

def get_objects_from_data(data):
    return data.iloc[:, 0].values


def get_attributes_from_data(data):
    return data.columns[1:].to_numpy()


def get_raw_data(path, ncols):
    return np.genfromtxt(path, delimiter=",", skip_header=1, usecols=range(1, ncols))
    

def read_json_data(path):
    if os.path.exists(path):
        with open(path, 'r') as file:
            return json.load(file)
        
    else:
        print("JSON Path does not exist!")
        exit(0)
    
    
def normalize_data(data):
    normalized_data = np.zeros_like(data)
    
    cols_sum = np.sum(data ** 2, axis=0)
    
    normalized_data = data / np.sqrt(cols_sum)
    
    return normalized_data


def apply_weights(normalized_data, weights):
    return normalized_data * weights


def find_ideal_positive_solution(p_matrix, criterias):
    index = 0
    solution = np.array([])
    for criteria in criterias:
        if (criteria == "C"):
            solution = np.append(solution, np.min(p_matrix[:, index]))
            
        elif (criteria == "B"):
            solution = np.append(solution, np.max(p_matrix[:, index]))
        
        index += 1
    
    return solution
            
    

def find_ideal_negative_solution(p_matrix, criterias):
    index = 0
    solution = np.array([])
    for criteria in criterias:
        if (criteria == "C"):
            solution = np.append(solution, np.max(p_matrix[:, index]))
            
        elif (criteria == "B"):
            solution = np.append(solution, np.min(p_matrix[:, index]))
        
        index += 1
    
    return solution


def find_positive_distances_matrix(p_matrix, positive_sol):
    d_matrix = np.linalg.norm(p_matrix - positive_sol, axis=1)
    
    return d_matrix

def find_negative_distances_matrix(p_matrix, negative_sol):
    d_matrix = np.linalg.norm(p_matrix - negative_sol, axis=1)
    
    return d_matrix
    

def topsis(raw_data, criterias, weights):
    normalized_data = normalize_data(raw_data)
    p_matrix = apply_weights(normalized_data, np.array(weights))
    
    positive_sol = find_ideal_positive_solution(p_matrix, criterias)
    negative_sol = find_ideal_negative_solution(p_matrix, criterias)
    
    pos_d_matrix = find_positive_distances_matrix(p_matrix, positive_sol)
    neg_d_matrix = find_negative_distances_matrix(p_matrix, negative_sol)
    
    result = neg_d_matrix / (pos_d_matrix + neg_d_matrix)
    
    return result

def generate_ranking(score, objects):
    associate_elems = tuple(zip(objects, score))
    ranking = sorted(associate_elems, key=lambda x: x[1], reverse=True)
   
    return ranking    
    

def print_result(ranking):
    
    output_directory = "../output"
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)
    
    with open(output_directory + "/rank.txt", "w") as file:
        print("Ranking:", file=file)
        for i in range(len(ranking)):
            print(f"{i+1}. {ranking[i][0]} -> Score: {ranking[i][1]:.4f}", file=file)
    
    suppliers, score = zip(*ranking)
    
    plt.bar(suppliers, score, color="skyblue")
    
    plt.xlabel("Objects")
    plt.ylabel("Performance Score")
    plt.title("Performance Ranking of Objects - TOPSIS")

    plt.savefig(output_directory + "/rank.png")

    
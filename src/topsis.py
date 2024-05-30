import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import json
import os

def read_data(path):
    """
    Reads a CSV file from the given path and returns a DataFrame.

    Parameters:
    path (str): The file path to the CSV file.

    Returns:
    pd.DataFrame: The data from the CSV file.

    Raises:
    SystemExit: If the file does not exist.
    """
    if os.path.exists(path):
        data = pd.read_csv(path)
        return data
    
    else:
        print("Data Path does not exist.")
        exit(0)
    

def get_objects_from_data(data):
    """
    Extracts the objects (first column) from the data.

    Parameters:
    data (pd.DataFrame): The input data.

    Returns:
    np.ndarray: The objects from the first column of the data.
    """
    return data.iloc[:, 0].values


def get_attributes_from_data(data):
    """
    Extracts the attribute names (column headers excluding the first column) from the data.

    Parameters:
    data (pd.DataFrame): The input data.

    Returns:
    np.ndarray: The attribute names.
    """
    return data.columns[1:].to_numpy()


def get_raw_data(path, ncols):
    """
    Reads raw data from a CSV file, excluding the first column and header.

    Parameters:
    path (str): The file path to the CSV file.
    ncols (int): The number of columns to read.

    Returns:
    np.ndarray: The raw data excluding the first column.
    """
    return np.genfromtxt(path, delimiter=",", skip_header=1, usecols=range(1, ncols))
    

def read_json_data(path):
    """
    Reads a JSON file from the given path and returns its content.

    Parameters:
    path (str): The file path to the JSON file.

    Returns:
    dict: The content of the JSON file.

    Raises:
    SystemExit: If the file does not exist.
    """
    if os.path.exists(path):
        with open(path, 'r') as file:
            return json.load(file)
        
    else:
        print("JSON Path does not exist.")
        exit(0)
    
    
def normalize_data(data):
    """
    Normalizes the data using vector normalization.

    Parameters:
    data (np.ndarray): The input data to be normalized.

    Returns:
    np.ndarray: The normalized data.
    """
    cols_sum = np.sum(data ** 2, axis=0)
    
    normalized_data = data / np.sqrt(cols_sum)
    
    return normalized_data


def apply_weights(normalized_data, weights):
    """
    Applies weights to the normalized data.

    Parameters:
    normalized_data (np.ndarray): The normalized data.
    weights (np.ndarray): The weights to apply.

    Returns:
    np.ndarray: The weighted data.
    """
    return normalized_data * weights


def find_ideal_positive_solution(p_matrix, criterias):
    """
    Finds the ideal positive solution based on the criteria.

    Parameters:
    p_matrix (np.ndarray): The weighted normalized data.
    criterias (list): List of criteria types ('C' for cost, 'B' for benefit).

    Returns:
    np.ndarray: The ideal positive solution.

    Raises:
    SystemExit: If any criteria is not 'C' (Cost) or 'B' (Benefit).
    """
    index = 0
    solution = np.array([])
    for criteria in criterias:
        if (criteria == "C"):
            solution = np.append(solution, np.min(p_matrix[:, index]))
            
        elif (criteria == "B"):
            solution = np.append(solution, np.max(p_matrix[:, index]))
        
        else:
            print("All criteria must be either 'B'(Benefit) or 'C'(Cost).")
            exit(0)
        
        index += 1
    
    return solution
    

def find_ideal_negative_solution(p_matrix, criterias):
    """
    Finds the ideal negative solution based on the criteria.

    Parameters:
    p_matrix (np.ndarray): The weighted normalized data.
    criterias (list): List of criteria types ('C' for cost, 'B' for benefit).

    Returns:
    np.ndarray: The ideal negative solution.

    Raises:
    SystemExit: If any criteria is not 'C' (Cost) or 'B' (Benefit).
    """
    index = 0
    solution = np.array([])
    for criteria in criterias:
        if (criteria == "C"):
            solution = np.append(solution, np.max(p_matrix[:, index]))
            
        elif (criteria == "B"):
            solution = np.append(solution, np.min(p_matrix[:, index]))
        
        else:
            print("All criteria must be either 'B'(Benefit) or 'C'(Cost).")
            exit(0)
        
        index += 1
    
    return solution


def find_positive_distances_matrix(p_matrix, positive_sol):
    """
    Calculates the distances of each point to the ideal positive solution.

    Parameters:
    p_matrix (np.ndarray): The weighted normalized data.
    positive_sol (np.ndarray): The ideal positive solution.

    Returns:
    np.ndarray: The distances to the ideal positive solution.
    """
    d_matrix = np.linalg.norm(p_matrix - positive_sol, axis=1)
    
    return d_matrix


def find_negative_distances_matrix(p_matrix, negative_sol):
    """
    Calculates the distances of each point to the ideal negative solution.

    Parameters:
    p_matrix (np.ndarray): The weighted normalized data.
    negative_sol (np.ndarray): The ideal negative solution.

    Returns:
    np.ndarray: The distances to the ideal negative solution.
    """
    d_matrix = np.linalg.norm(p_matrix - negative_sol, axis=1)
    
    return d_matrix


def generate_ranking(score, objects):
    """
    Generates a ranking based on the TOPSIS scores.

    Parameters:
    score (np.ndarray): The TOPSIS scores.
    objects (np.ndarray): The objects to be ranked.

    Returns:
    list: A sorted list of tuples containing objects and their scores in descending order.
    """
    associate_elems = tuple(zip(objects, score))
    ranking = sorted(associate_elems, key=lambda x: x[1], reverse=True)
   
    return ranking    

def topsis(raw_data, criterias, weights, objects):
    """
    Performs the TOPSIS method on the given data.

    Parameters:
    raw_data (np.ndarray): The raw data.
    criterias (list): List of criteria types ('C' for cost, 'B' for benefit).
    weights (list): List of weights for each criterion.
    objects (list): List of objects.

    Returns:
    list: The ranked list of tuples containing objects and their scores.

    Raises:
    SystemExit: If the length of criterias and weights does not match the number of columns in raw_data.
    """
    if len(criterias) != len(weights) or len(criterias) != raw_data.shape[1]:
        print("Make sure criterias and weights array have the same lenght as the number of columns in data input.")
        exit(0)
    
    # Step 0: normalize data and apply weights
    normalized_data = normalize_data(raw_data)
    p_matrix = apply_weights(normalized_data, np.array(weights))
    
    # Step 1: find ideal and reverse-ideal solutions
    positive_sol = find_ideal_positive_solution(p_matrix, criterias)
    negative_sol = find_ideal_negative_solution(p_matrix, criterias)
    
    # Step 2: compute the euclidian distance from each object to the ideal and rever-ideal solutions
    pos_d_matrix = find_positive_distances_matrix(p_matrix, positive_sol)
    neg_d_matrix = find_negative_distances_matrix(p_matrix, negative_sol)
    
    # Step 3: compute relative proximity to each object to generate individual score
    score = neg_d_matrix / (pos_d_matrix + neg_d_matrix)
    
    # Step 4: rank the objects based on the score
    rank = generate_ranking(score, objects)
    
    return rank
    

def print_result(rank):
    """
    Prints the ranking results to a file and saves a bar plot.

    Parameters:
    rank (list): The sorted list of tuples containing objects and their scores.

    Outputs:
    A text file with the ranking results and a bar plot saved as an image.
    """
    output_directory = "../output"
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)
    
    # Print on 'rank.txt' output file
    with open(output_directory + "/rank.txt", "w") as file:
        print("Ranking:", file=file)
        for i in range(len(rank)):
            print(f"{i+1}. {rank[i][0]} -> Score: {rank[i][1]:.4f}", file=file)
    
    
    suppliers, score = zip(*rank)
    
    # Plot 'rank.png' bar graphic
    plt.bar(suppliers, score, color="skyblue")
    plt.xlabel("Objects")
    plt.ylabel("Performance Score")
    plt.title("Performance Ranking of Objects - TOPSIS")
    plt.savefig(output_directory + "/rank.png")

    
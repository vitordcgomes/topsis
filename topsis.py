import numpy as np
import matplotlib.pyplot as plt

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
    p_matrix = apply_weights(normalized_data, weights)
    
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
    print("\nRanking:")
    
    for i in range(len(ranking)):
        print(f"{i+1}. {ranking[i][0]} -> Score: {ranking[i][1]:.3f}")
    print("\n")
    
    suppliers, score = zip(*ranking)
    
    plt.bar(suppliers, score, color="skyblue")
    
    plt.xlabel("Objects")
    plt.ylabel("Performance Score")
    plt.title("Performance Ranking of Objects - TOPSIS")

    plt.savefig("rank.png")

    
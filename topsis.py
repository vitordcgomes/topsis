import numpy as np

def normalize_data(data):
    normalized_data = np.zeros_like(data)
    
    cols_sum = np.sum(data ** 2, axis=0)
    
    normalized_data = data / np.sqrt(cols_sum)
    print(normalized_data)
    
    return normalized_data


def apply_weights(normalized_data, weights):
    print("\n",normalized_data * weights,"\n")
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
    
    # print("\n", solution)
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
    
    # print("\n", solution)
    return solution


def topsis(raw_data, criterias, weights):
    normalized_data = normalize_data(raw_data)
    p_matrix = apply_weights(normalized_data, weights)
    
    positive_sol = find_ideal_positive_solution(p_matrix, criterias)
    negative_sol = find_ideal_negative_solution(p_matrix, criterias)
    
    print(positive_sol, "\n")
    print(negative_sol, "\n")
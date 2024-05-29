import numpy as np

def normalizing_data(data):
    normalized_data = np.zeros_like(data)
    
    col_sum = np.sum(data ** 2, axis=0)
    normalized_data = data / np.sqrt(col_sum)
        
    print(normalized_data)
    
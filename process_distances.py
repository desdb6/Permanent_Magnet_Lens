"""
Author: Des De Borger

Script to process distances on images.
"""

import numpy as np

def analyze_distances(file_path, three_gaps=False, five_gaps=False):
    """
    Reads distance data from a text file and calculates the 
    average and standard error.
    """
    
    data = np.loadtxt(file_path)

    avg = np.mean(data)
        
    # Standard Error of the Mean (SEM)
    # Formula: Standard Deviation / sqrt(n)
    std_dev = np.std(data, ddof=1) # Using sample standard deviation
    std_error = std_dev / np.sqrt(len(data))
    
    if three_gaps==True:
        avg = avg*204/(3*204+2*50)
        std_error = std_error*204/(3*204+2*50)
    if five_gaps==True:
        avg = avg*204/(5*204+4*50)
        std_error = std_error*204/(5*204+4*50)
    print(f"Average mesh distance:{avg} microns, standard error:{std_error} microns")
    return avg, std_error

if __name__ == "__main__":
    analyze_distances(r'01042026_images\np\Resolution_np_shadow\270mm_mesh_measured_dist.txt', three_gaps=False, five_gaps=True)
    
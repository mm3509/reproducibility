import numpy as np
import sys

filepath = sys.argv[1]

my_data = np.genfromtxt(filepath, delimiter = ',')
#print(my_data)
print(np.std(my_data, axis = 0))

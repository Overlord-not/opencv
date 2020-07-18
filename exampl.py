import numpy as np

n=np.array([1,2,3],ndmin=2)


targets = np.zeros(10) + 0.01
targets[int(all_values[0])] = 0.99
print(targets)
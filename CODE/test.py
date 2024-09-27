import numpy as np

l = np.array([[0,1,2],
             [0,1,2]])
s = np.array([[3,4,5,5,5],
                [3,4,5,5,5]])
m = np.hstack([l,s])
print(f"l = {l}")
print(f"s = {s}")
print(f"m = {m}")
print(f" m shape = {np.array(m).shape}")

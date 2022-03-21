from Matrix import *
import numpy as np
import sys

Avals = [[1,2,3],
         [3,2,1],
         [1,0,-1]]
A = Matrix(Avals) 
npA = np.array(A.vals) 
eVals, eVecs = np.linalg.eig(npA) 

array = np.array([1,2,3,3])
for num in array: 
     print(num)



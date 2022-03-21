################################################
# Matrix.py 
# Matrix class
# Name: Tianze Shou 
# Andrew ID: tshou
################################################

import copy, math, sys
from fractions import Fraction 
import numpy as np

# taken from 15-112 website 
# https://www.cs.cmu.edu/~112/index.html
def almostEqual(d1, d2, epsilon=10**-7):
    # note: use math.isclose() outside 15-112 with Python version 3.5 or later
    return (abs(d2 - d1) < epsilon)

# taken from 15-112 website 
# https://www.cs.cmu.edu/~112/index.html 
def make2dList(rows, cols):
    return [ ([0] * cols) for row in range(rows) ]

def zeroOnly(lst): 
    length = len(lst) 
    return lst == [0] * length 

def isInt(num): 
    return almostEqual(num, math.floor(num)) or almostEqual(num, math.ceil(num))  

# this function find the element with smallest abs val, except for zero 
# takes in an iterable object 
def findAbsMinExcZero(array): 
    minNum = sys.maxsize 
    for n in array: 
        if abs(n) < minNum and n != 0: 
            minNum = abs(n) 

    return minNum 


class Matrix: 
    # the constructor takes in a 2D list as values of the matrix
    def __init__(self, vals):
        for i in range(len(vals)): 
            if len(vals[i]) != len(vals[0]): 
                raise Exception('Matrix value illegal')
        self.vals = vals 
        if vals == []: 
            self.rows = 0
            self.cols = 0
        else: 
            self.rows = len(vals) 
            self.cols = len(vals[0]) 

        # handle string entry 
        for i in range(self.rows): 
            for j in range(self.cols): 
                if isinstance(self.vals[i][j], str): 
                    if self.vals[i][j].count('.') == 1: 
                        self.vals[i][j] = float(self.vals[i][j])
                    elif self.vals[i][j].count('/') == 1: 
                        self.vals[i][j] = Fraction(self.vals[i][j])
                    else: 
                        try: 
                            self.vals[i][j] = int(self.vals[i][j])
                        except: 
                            raise Exception('Matrix value illegal')

        # # handle approximation error 
        # for i in range(self.rows): 
        #     for j in range(self.cols): 
        #         if almostEqual(self.vals[i][j], int(self.vals[i][j])): 
        #             self.vals[i][j] = int(self.vals[i][j])
        #         elif (isinstance(self.vals[i][j], Fraction) and 
        #               len(str(self.vals[i][j])) > 7): 
        #             self.vals[i][j] = float(self.vals[i][j]) 

    # return the repr in a rather neat manner 
    # round to two decimal numbers 
    def __repr__(self): 
        maxLength = 7
        result = '\n' 
        for i in range(self.rows): 
            result += '{'
            for j in range(self.cols): 
                if isinstance(self.vals[i][j], float): 
                    num = '{:.2f}'.format(self.vals[i][j])
                else: 
                    num = str(self.vals[i][j])
                result += num.rjust(maxLength)
            result += '}\n' 

        return result 

    def __eq__(self, other): 
        # type unmatched 
        if not isinstance(other, Matrix): 
            return False 

        # Dimension unmatched 
        if not self.getDimension() == other.getDimension(): 
            return False 

        # Values unmatched 
        m, n = self.getDimension() 
        for i in range(m): 
            for j in range(n): 
                if not almostEqual(self.getVal(i,j), other.getVal(i,j)): 
                    return False 

        return True 

    def getDimension(self): 
        return self.rows, self.cols 

    def getVal(self, row, col): 
        return self.vals[row][col]

    def modifyVal(self, i, j, newVal): 
        self.vals[i][j] = newVal

    # non-destructively remove the ith row, starting from 0
    def removeRow(self, i):  
        newVals = copy.deepcopy(self.vals)
        newVals.pop(i) 
        return Matrix(newVals)

    def removeCol(self, i): 
        newVals = copy.deepcopy(self.vals)
        for row in newVals: 
            row.pop(i) 
        return Matrix(newVals)

    # have a matrix to multiply with another matrix, int, or float
    # non-destructive, returns a new matrix 
    def multiply(self, other): 
        # scalar multiplication 
        if (isinstance(other, int) or 
            isinstance(other, float) or 
            isinstance(other, Fraction)):
            newVals = copy.deepcopy(self.vals)
            for i in range(self.rows): 
                for j in range(self.cols): 
                    newVals[i][j] *= other
            
            return Matrix(newVals) 

        # matrices multiplication 
        elif isinstance(other, Matrix): 
            # the dimensions of matrices must match in order to multiply 
            if self.getDimension()[1] != other.getDimension()[0]: 
                raise Exception('Matrices dimension unmatched')

            newRows = self.getDimension()[0] 
            newCols = other.getDimension()[1]
            middleDim = self.getDimension()[1]
            newVals= make2dList(newRows, newCols) 
            for i in range(newRows): 
                for j in range(newCols): 
                    for n in range(middleDim): 
                        newVals[i][j] += self.vals[i][n] * other.vals[n][j] 
            
            return Matrix(newVals) 

        # A matrix cannot multiply with things other than a number or a matrix 
        # Exception will be raised 
        else:
            raise Exception(f'Cannot multiply Matrix with {type(other)}')
    
    # enables Matrix*int
    def __mul__(self, other): 
        return self.multiply(other)

    # enables int*Matrix
    def __rmul__(self, other): 
        return self.multiply(other)

    def __add__(self, other): 
        if not isinstance(other, Matrix): 
            raise Exception('Can only add Matrix with Matrix') 

        if self.getDimension() != other.getDimension(): 
            raise Exception('Matrix dimension unmatched') 

        m, n = self.getDimension() 
        newVals = make2dList(m, n) 
        for i in range(m): 
            for j in range(n): 
                newVals[i][j] = self.vals[i][j] + other.vals[i][j] 

        return Matrix(newVals) 

    def __sub__(self, other): 
        if not isinstance(other, Matrix): 
            raise Exception('Can only add Matrix with Matrix') 

        if self.getDimension() != other.getDimension(): 
            raise Exception('Matrix dimension unmatched') 

        m, n = self.getDimension() 
        newVals = make2dList(m, n) 
        for i in range(m): 
            for j in range(n): 
                newVals[i][j] = self.vals[i][j] - other.vals[i][j] 

        return Matrix(newVals) 

    def __pow__(self, other): 
        if other != math.floor(other): 
            raise Exception('Matrix can only be raised to integer power')
        if not self.isSquare(): 
            raise Exception('Only sqaure matrix ca be raised to a power') 
        
        if other >= 0: 
            newMatrix = Matrix.makeI(self.getDimension()[0])
            for i in range(other): 
                newMatrix *= self

            return newMatrix
        elif other == -1: 
            return self.findInverse() 
        else: 
            raise Exception(f'Matrix cannot be raised to {other} power') 
    
    # return if the matrix is square 
    def isSquare(self): 
        m, n = self.getDimension()
        return m == n
    
    # find the determinant of the matrix 
    # uses Leplace expansion theorem 
    # this is a recursive method 
    def det(self): 
        # Non-sqaure matrix does not have determinant
        if not self.isSquare(): 
            raise Exception('Non-sqaure matrix does not have determinant')
        
        # base case
        # return the only value if the matrix is one-by-one
        if self.getDimension() == (1, 1):
            return self.getVal(0, 0) 
        # recursion case 
        # use LaPlace expansion theorem and expand along the first (0th) row
        else:  
            determinant = 0 
            for i in range(self.getDimension()[1]): 
                entry = self.getVal(0, i)
                coFactor = self.getCoFactor(0, i) 
                determinant += entry * coFactor 

            if isInt(determinant): 
                return int(determinant)
            else: 
                return determinant

    # get the cofactor of entry at (i, j) 
    def getCoFactor(self, i, j): 
        sign = math.pow(-1, i+j) 
        minorMatrix = self.removeRow(i).removeCol(j)
        minor = minorMatrix.det() 

        return sign*minor  

    # return the inverse matrix of self 
    def findInverse(self): 
        # Non-sqaure matrix has no inverse
        if not self.isSquare(): 
            raise Exception('Non-sqaure matrix has no inverse') 
        # if the matrix is not invertible 
        if self.det() == 0: 
            raise Exception('Matrix not invertible')

        m, n = self.getDimension()  
        # we need to find the cofactor matrix first 
        coFactorMatrixVals = make2dList(m, n) 
        for i in range(m): 
            for j in range(n): 
                coFactorMatrixVals[i][j] = self.getCoFactor(i, j) 
        
        return (Fraction(1/self.det()) * 
                Matrix(coFactorMatrixVals).findTranspose())

    def findTranspose(self): 
        m0, n0 = self.getDimension() 
        m1, n1 = n0, m0 # dimensions of the resulting matrix 
        newVals = make2dList(m1, n1) 

        for i in range(m0): 
            for j in range(n0): 
                newVals[j][i] = self.vals[i][j] 

        return Matrix(newVals)

    # switch the ith and jth row
    # destructive method 
    def rowSwitch(self, i, j): 
        self.vals[i], self.vals[j] = self.vals[j], self.vals[i]


    # reduce the matrix to row-echelon form 
    def RREF(self): 
        newVals = copy.deepcopy(self.vals) 
        m, n = self.getDimension() 
        operations = list()

        # loop through the row in matrix 
        for i in range(min(m, n)):
            # if rows need to be switched 
            if newVals[i][i] == 0: 
                # loop through the rows again to see which to switch 
                for k in range(i+1, m):
                    if newVals[k][i] != 0: 
                        newVals[i], newVals[k] = newVals[k], newVals[i]
                        operations.append(f'R{i}<-->R{k}')
                        break
            
            # do row operations on the rows after Ri
            for k in range(i+1, m): 
                if newVals[i][i] != 0: 
                    f = -Fraction(newVals[k][i], newVals[i][i])
                    # do row operations on Rk
                    if f != 0:
                        operations.append(f'R{k}+=<{f}>R{i}')
                        for j in range(n): 
                            newVals[k][j] += newVals[i][j] * f

        return Matrix(newVals), operations

    @staticmethod 
    # returns an identity matrix 
    def makeI(size): 
        IVals = make2dList(size, size) 
        for i in range(size): 
            for j in range(size): 
                if i == j: IVals[i][j] = 1
        
        return Matrix(IVals) 

    # performs LU or PTLU factorization 
    def LUFac(self): 
        m, n = self.getDimension()
        U, operations = self.RREF() 
        P = Matrix.makeI(m) 
        L = Matrix.makeI(m)

        # get the correct P
        for opt in operations: 
            if opt.find('<-->') != -1: # the operation incolves a switch 
                # get the row number of row-switch 
                i, j = int(opt[1]), int(opt[-1]) 
                P.rowSwitch(i, j) 

        # get the correct L
        for opt in operations:
            if opt.find('+=') != -1: # the operation involves a row addition 
                start = opt.find('<') 
                end = opt.find('>')
                num = -Fraction(opt[start+1:end])
                i, j = int(opt[1]), int(opt[-1])
                L.modifyVal(i, j, num) 

        return (P.findTranspose(), L, U)

    # create a copy of self
    def copy(self): 
        return Matrix(copy.deepcopy(self.vals))

    # solve the linear system for x where Ax = b
    @staticmethod
    def solveLinearSystem(A, b): 
        agmtedVal = []
        for i in range(A.rows): 
            agmtedVal.append(copy.deepcopy(A.vals[i])) 
            agmtedVal[i].extend(copy.deepcopy(b.vals[i]))

        agmtedMatrix = Matrix(agmtedVal)
        # get the reduced row-echelon form of the augmented matrix 
        RREFM, operations = agmtedMatrix.RREF()

        # see if the system has no solution 
        for i in range(RREFM.rows): 
            rowAllZero = True
            for j in range(RREFM.cols-1):
                if RREFM.vals[i][j] != 0: rowAllZero = False 
            # the system has no solution 
            if rowAllZero and RREFM.vals[i][RREFM.cols-1] != 0: return None

        # see if the system has unique solution 
        if A.isSquare() and A.det() != 0: 
            return A.findInverse() * b 

        # finally, the system has infinite solutions 
        result = '\n' 
        for row in RREFM.vals: 
            if zeroOnly(row): continue 
            isFirst = True
            for i in range(len(row)): 
                if isFirst and row[i] != 0: 
                    result += f'({row[i]})x{i}'
                    isFirst = False 
                elif i == len(row)-1: 
                    result += f'={row[i]}\n'
                elif row[i] != 0: 
                    result += f'+({row[i]})x{i}'

        return result

    def makeAugmentedMatrx(self, b): 
        if not (self.rows == b.rows and b.cols == 1):
            raise Exception('Cannot make augmented matrix due to dimension mismatch') 

        agmtedVal = []
        for i in range(self.rows): 
            agmtedVal.append(copy.deepcopy(self.vals[i])) 
            agmtedVal[i].extend(copy.deepcopy(b.vals[i]))

        agmtedMatrix = Matrix(agmtedVal)
        return agmtedMatrix

    # draw() method, used in joint with CMU graphics 
    # needs a canvas, and (x,y) is the upper-left cornor of the matrix 
    def draw(self, canvas, x, y): 
        x0, y0 = x, y
        size = 40 # the grid size 
        m, n = self.getDimension()
        for i in range(m): 
            for j in range(n): 
                if isinstance(self.getVal(i, j), float): 
                    if isInt(self.getVal(i, j)): 
                        num = int(self.getVal(i, j))
                    else: 
                        num = '{:.2f}'.format(self.getVal(i, j)) 
                elif isinstance(self.getVal(i, j), Fraction): 
                    # handle the case where Fraction is too long 
                    if len(str(self.getVal(i, j))) > 6: 
                        num = '{:.2f}'.format(float(self.getVal(i, j))) 
                    else: 
                        num = self.getVal(i, j) 
                else: 
                    num = self.getVal(i, j)
                canvas.create_text((2*x+size)/2, (2*y+size)/2, 
                                   text=str(num))
                x += size 
            y += size
            x = x0

        # draw the parenthesis surrounding the matrix 
        xn, yn = x0 + n*size, y0 + m*size 
        size = 10 # size of the bend 
        # draw the left parenthesis 
        canvas.create_line(x0, y0, x0, yn) # straight line 
        canvas.create_line(x0, y0, x0+size, y0-size) # upper bend 
        canvas.create_line(x0, yn, x0+size, yn+size) # lower bend 
        # draw the right parenthesis 
        canvas.create_line(xn, y0, xn, yn) # straight line
        canvas.create_line(xn, y0, xn-size, y0-size) # upper bend 
        canvas.create_line(xn, yn, xn-size, yn+size) # lower bend 

    # do row operations 
    # e.g.: R1+=<-1>R0, R0<-->R1 
    # takes in a string, return a new Matrix 
    def rowOp(self, op): 
        if op.find('<-->') != -1: # it is a row switch
            r1 = int(op[1])
            r2 = int(op[-1])
            M = self.copy()
            M.rowSwitch(r1, r2) 
            return M
        elif op.find('+=') != -1: # it is a row addition/subtraction 
            r1 = int(op[1]) # the receiving row
            r2 = int(op[-1])
            M = self.copy() 
            start = op.find('<') 
            end = op.find('>') 
            try: 
                f = float(op[start+1:end]) 
            except: 
                # for cases like float('-3/2')
                # error would occur otherwise 
                f = float(Fraction(op[start+1:end]))
            for j in range(M.cols): 
                M.vals[r1][j] += f*M.vals[r2][j] 

            return M 
        else: 
            raise Exception('Invalid row operation command')

    # find the eigen values and eigen vectors of Matrix A
    def eigen(A): 
        result = list()
        npA = np.array(A.vals) 
        eVals, eVecs = np.linalg.eig(npA) 
        for i in range(len(eVals)): 
            eVal = eVals[i] 
            eVec = eVecs[:,i] 
            divisor = findAbsMinExcZero(eVec) 
            eVec = eVec / divisor # try to intergerize e-vector 
            # convert eVec to Matrix Class 
            eVecVals = []
            for val in eVec: 
                eVecVals.append([val]) 

            eVecMatrix = Matrix(eVecVals) 
            result.append((eVal, eVecMatrix)) # append a tuple to the result 

        return result 


'''
A = Matrix([[1,-3,11],
            [2,-6,16],
            [1,-3,7]]) 
print(A.eigen())


b = Matrix([[1],
            [2], 
            [3]])
print(A.makeAugmentedMatrx(b))

print(A.rowOp('R0+=<-1>R2')) 

print(A*A**-1)

bVals = [[0],
         [0],
         [0],
         [0]]
b = Matrix(bVals)

I = Matrix.makeI(4)

print(Matrix.solveLinearSystem(A, b))

# TESTS
# A is 3*4
Avals = [[1,2,-2,8],
         [2,3,-4,5],
         [3,4,5,6]]
A = Matrix(Avals) 
# B is 4*2
BVals = [[1,4],
         [5,-6],
         [7,8]]
B = Matrix(BVals)
# C is 3*3
CVals = [[1,2,3], 
         [4,5,6], 
         [7,8,9]]
C = Matrix(CVals)
# identity matrix 
I = Matrix([[1,0,0], 
            [0,1,0], 
            [0,0,1]])
anotherI = Matrix([[1,0,0], 
                   [0,1,0], 
                   [0,0,1]])
# D is 4*4 
D = Matrix([[13,22,4,7], 
            [5,10,6,0], 
            [3,5,-5,2], 
            [-33,19,5,6]])

EVals = [[1,1,-2,8],
         [2,3,-4,5],
         [3,4,5,6]]
E = Matrix(EVals)

GVals = [[0,-1,1,3], 
         [-1,1,1,2],
         [0,1,-1,1],
         [0,0,1,1]] 
G = Matrix(GVals) 

'''







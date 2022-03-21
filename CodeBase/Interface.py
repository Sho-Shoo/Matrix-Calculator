from cmu_112_graphics import *
from TextField import *
from fractions import Fraction 
from Matrix import *
from tkinter import *

# taken from 15-112 website 
# https://www.cs.cmu.edu/~112/index.html 
def make2dList(rows, cols):
    return [ ([0] * cols) for row in range(rows) ]

def det(M): 
    if not isinstance(M, Matrix):  
        raise Exception(f'{type(M)} has no determinant') 
    else: 
        return M.det() 

def RREF(M): 
    if not isinstance(M, Matrix): 
        raise Exception(f' Cannot perform RREF on {type(M)}')
    else: 
        result, operations = M.RREF() 
        return result 

def RREFOperations(M): 
    if not isinstance(M, Matrix): 
        raise Exception(f' Cannot perform RREF on {type(M)}')
    else: 
        result, operations = M.RREF() 
        return operations 

# find transpose 
def T(M): 
    if not isinstance(M, Matrix): 
        raise Exception(f'Cannot perform transpose operation on {type(M)}')
    else: 
        return M.findTranspose() 

def LU(M): 
    if not isinstance(M, Matrix): 
        raise Exception(f'Cannot perform LU factorization on {type(M)}')
    else: 
        return M.LUFac()  

# find the eigen values and eigen vectors of a matrix 
def eigen(M): 
    if not isinstance(M, Matrix): 
        raise Exception(f'Cannot find eigen-values or eigen-vectors of {type(M)}') 
    else: 
        return M.eigen() 

# divid row operations from RREF into substeps 
def dividIntoSteps(ops): 
     steps = [[]]
     previousOp = None 
     for op in ops: 
          if op.find('<-->') != -1: 
               # the operation is a row switch, it has its own step
               steps.append([]) 
               steps[-1].append(op) 
               previousOp = None
          elif previousOp == None or previousOp[-1] == op[-1]:
               # previous operation has the same giving row as the current 
               # append op to the current step
               steps[-1].append(op) 
               previousOp = op
          else:
               # previous op and current differs, start a new step
               steps.append([op]) 
               previousOp = op
     
     # make sure there is no empty step
     for i in range(len(steps)): 
          if steps[i] == []: steps.pop(i) 

     return steps

class MyApp(App):

    def appStarted(app):
        app.matrixMargin = 450
        app.d = 5 # matrix field dimension 
        app.Ax, app.Ay = 20, 100
        app.Bx, app.By = app.Ax + app.matrixMargin, app.Ay 
        app.Cx, app.Cy = app.Bx + app.matrixMargin, app.By 
        app.exprX, app.exprY = 280, 350
        app.exprToBeDrawn = '' 
        app.answerFieldX, app.answerFieldY = app.exprX, app.exprY+100
        app.mouseX, app.mouseY = None, None

        # 2d lists continaing 5*5 test entry fields 
        app.AFields = TextField.make2dFields(app.d, app.d, app.Ax, app.Ay)
        app.BFields = TextField.make2dFields(app.d, app.d, app.Bx, app.By)
        app.CFields = TextField.make2dFields(app.d, app.d, app.Cx, app.Cy)
        app.A, app.B, app.C, app.result = None, None, None, None 

        # 1d list containing aliases of all fields 
        # for clearity purpose 
        app.allFields = []
        for M in [app.AFields, app.BFields, app.CFields]: 
            for row in M: 
                for field in row: 
                    app.allFields.append(field) 

        app.exprField = TextField(app.exprX, app.exprY, 600, 55, 'w', 'Arial 20')
        app.allFields.append(app.exprField) 

        app.selectedField = None 

        app.canDrawRREFProcess = False 
        app.canDrawLinearSolution = False 
        app.canDrawEigenResult = False 

        # logo design by Zhejiang Sci-Tech Univeristy, Apparel Design Student, Jing Tian
        # Much gratitude toward her artistic assistance 
        app.logo = app.loadImage('TP LOGO.JPG') # load logo image
        app.logo = app.scaleImage(app.logo, 1/3) # scale the logo 

    def mousePressed(app, event): 
        pressedOnField = False 
        for field in app.allFields: 
            if field.pressed(event.x, event.y): 
                if app.selectedField != None: 
                    app.selectedField.selectionOff()
                app.selectedField = field 
                app.selectedField.selectionOn()
                pressedOnField = True
                break 

        if not pressedOnField and app.selectedField != None: 
            app.selectedField.selectionOff() 
            app.selectedField = None

    def mouseMoved(app, event): 
        app.mouseX, app.mouseY = event.x, event.y 

    def keyPressed(app, event):

        if event.key == 'Enter': # Enter is pressed, start calculation 
            # transcribe number in fields to variables 
            AVals = []
            for row in app.AFields: 
                entries = []
                for field in row:
                    entries.append(field.getMessage().strip()) 
                while len(entries) > 0 and entries[-1] == '': 
                    entries.pop() 
                if entries != []: AVals.append(entries)

            BVals = []
            for row in app.BFields: 
                entries = []
                for field in row:
                    entries.append(field.getMessage().strip()) 
                while len(entries) > 0 and entries[-1] == '': 
                    entries.pop() 
                if entries != []: BVals.append(entries)

            CVals = []
            for row in app.CFields: 
                entries = []
                for field in row:
                    entries.append(field.getMessage().strip()) 
                while len(entries) > 0 and entries[-1] == '': 
                    entries.pop() 
                if entries != []: CVals.append(entries)

            # compose the matrices and do calculations
            try: 
                app.A = Matrix(AVals) 
                app.B = Matrix(BVals) 
                app.C = Matrix(CVals) 
            except Exception as error: 
                app.showMessage(str(error))
                return 

            expression = app.exprField.getMessage().replace('A', 'app.A').replace('B', 'app.B').replace('C', 'app.C')
            # do not pass in any app.result if it is a linear system process 
            # or a eigen process 
            if (expression != '' and expression.find('=') == -1 and 
                expression.find('eigen') == -1):  
                try: 
                    app.result = eval(expression)
                except Exception as error: 
                    # for error capture purpose 
                    app.showMessage(str(error))
                    return 
            else: 
                # if the expression entered is empty string or an equation to solve
                # we do not calculate the result 
                app.result = None

            app.exprToBeDrawn = app.exprField.getMessage()
            app.canDrawRREFProcess = (app.exprField.getMessage() == 'RREF(A)' or 
                                      app.exprField.getMessage() == 'RREF(B)' or 
                                      app.exprField.getMessage() =='RREF(C)')
            app.canDrawLinearSolution = (app.exprField.getMessage().find('=') != -1)
            app.canDrawEigenResult = (app.exprField.getMessage().find('eigen') != -1) 


            '''
            IMPORTANT!!!!
            below is the final version of this part 
            used for Exception capture  
            '''
            # try: 
            #     app.A = Matrix(AVals) 
            #     app.B = Matrix(BVals) 
            #     app.C = Matrix(CVals) 
            #     app.result = eval(app.exprField.getMessage(), {'A': app.A,
            #                                                    'B': app.B,
            #                                                    'C': app.C})
            # except Exception as error: 
            #     app.showMessage(str(error))
            #     return 

        elif app.selectedField != None: 
            if len(event.key) == 1: # the user pressing number or letter key 
                app.selectedField.addMessage(event.key)
            elif event.key == 'Space': 
                app.selectedField.addMessage(' ')
            elif event.key == 'Delete': 
                app.selectedField.dropLastLetter() 


    def redrawAll(app, canvas):
        # draw logo 
        canvas.create_image(130, app.height-120, 
                            image=ImageTk.PhotoImage(app.logo))

        # draw mouse coordinates 
        # for graphics purposes 
        # will not be included in the final version 
        # canvas.create_text(app.width-70, app.height-30, 
        #                    text=f'X = {app.mouseX}', anchor='w') 
        # canvas.create_text(app.width-70, app.height-15, 
        #                    text=f'Y = {app.mouseY}', anchor='w') 

        # draw out matrix name
        margin = 40
        canvas.create_text(app.Ax + margin, app.Ay - margin, 
                           text='A', font='Arial 40 bold')
        canvas.create_text(app.Bx + margin, app.By - margin, 
                           text='B', font='Arial 40 bold')
        canvas.create_text(app.Cx + margin, app.Cy - margin, 
                           text='C', font='Arial 40 bold')
        canvas.create_text(20, 350, 
                           text='Enter your expression here\nPress ENTER to calculate',
                           font='Ariel 20 bold', anchor='nw')

        # draw out all the fields 
        for field in app.allFields: 
            field.draw(canvas) 

        # draw the expression/process before result 
        # indent is the length of th expression 
        if app.canDrawRREFProcess: 
            indent = drawRREFProcess(app, canvas)
        elif app.canDrawLinearSolution:
            indent = drawLinearSolution(app, canvas) 
        elif app.canDrawEigenResult: 
            indent = drawEigenResult(app, canvas) 
        else: 
            indent = drawExpression(app, canvas) 

        # draw app.result 
        if app.result == None: 
            pass
        elif (isinstance(app.result, int) or isinstance(app.result, float) or 
              isinstance(app.result, Fraction)):
            # if app.result is a number, just draw the text 
            canvas.create_text(app.answerFieldX+indent, app.answerFieldY, 
                               text=str(app.result), anchor='nw')
        elif isinstance(app.result, Matrix): 
            # if app.result is a Matrix, call Matrix.draw()
            app.result.draw(canvas, app.answerFieldX+indent, app.answerFieldY)
        elif isinstance(app.result, tuple): 
            # if app.result contains a tuple of matrices 
            for M in app.result: 
                unitSize = 40 # in coordinance with matrix.draw() 
                margin = 10
                M.draw(canvas, app.answerFieldX+indent, app.answerFieldY) 
                indent += M.cols*unitSize + margin 
        else: 
            raise Exception(f'app.result current does not support {type(app.result)}')

# returns the x-length of the expression 
def drawExpression(app, canvas): 
    indent = 0 # reset expression length to 0
    expression = app.exprToBeDrawn
    unitSize = 40 # in coordinance with matrix.draw() 
    margin = 10
    for char in expression: 
        # draw Matrices A, B, C if they are in the expression 
        if char == 'A': 
            app.A.draw(canvas, app.answerFieldX+indent, app.answerFieldY)
            m, n = app.A.getDimension() 
            indent += n*unitSize + margin
        elif char == 'B': 
            app.B.draw(canvas, app.answerFieldX+indent, app.answerFieldY)
            m, n = app.B.getDimension() 
            indent += n*unitSize + margin
        elif char == 'C': 
            app.C.draw(canvas, app.answerFieldX+indent, app.answerFieldY)
            m, n = app.C.getDimension() 
            indent += n*unitSize + margin 
        else: # char is just a normal letter 
            canvas.create_text(app.answerFieldX+indent, app.answerFieldY, 
                               text=char, anchor='n') 
            indent += margin 

    # draw the last '=' sign 
    canvas.create_text(app.answerFieldX+indent, app.answerFieldY, text='=', 
                       anchor='n') 
    indent += margin 

    return indent 

def drawRREFProcess(app, canvas): 
    indent = 0
    yIndent = 0
    margin = 10
    textMargin = margin + 5
    unitSize = 40
    arrowLen = 100
    userExpr = app.exprToBeDrawn.strip()
    Mstring = userExpr[-2] 
    if Mstring == 'A': M = app.A
    elif Mstring == 'B': M = app.B
    elif Mstring == 'C': M = app.C
    else: raise Exception('Unknown matrix name')
    operations = RREFOperations(M) 

    # draw out the original matrix 
    M.draw(canvas, app.answerFieldX, app.answerFieldY)
    indent += M.cols*unitSize + margin 
    # divid the operations into steps 
    steps = dividIntoSteps(operations)

    for step in steps: 
        # draw the arrow 
        canvas.create_line(app.answerFieldX+indent, app.answerFieldY, 
                           app.answerFieldX+indent+arrowLen, app.answerFieldY) 
        canvas.create_line(app.answerFieldX+indent+arrowLen, app.answerFieldY, 
                           app.answerFieldX+indent+arrowLen-margin, app.answerFieldY-margin) 
        canvas.create_line(app.answerFieldX+indent+arrowLen, app.answerFieldY, 
                           app.answerFieldX+indent+arrowLen-margin, app.answerFieldY+margin) 
        yIndent += margin
        # draw the text of operations 
        for op in step: 
            canvas.create_text(app.answerFieldX+indent, app.answerFieldY+yIndent,
                               text=op, anchor='nw')
            yIndent += textMargin 
            # formulate the matrix after this step
            M = M.rowOp(op) 
        
        indent +=arrowLen + margin
        yIndent = 0
        # the the intermediate matrix, not the final result
        if M != app.result: 
            M.draw(canvas, app.answerFieldX+indent, app.answerFieldY) 
            indent += M.cols*unitSize + margin  
        
    return indent

def drawLinearSolution(app, canvas): 
    indent = 0
    yIndent = 0
    margin = 10
    textMargin = margin + 5
    unitSize = 40
    arrowLen = 100
    userExpr = app.exprToBeDrawn.strip() 

    # get Matrix A in Ax = b
    if userExpr[0] == 'A': A = app.A
    elif userExpr[0] == 'B': A = app.B 
    elif userExpr[0] == 'C': A = app.C
    else: raise Exception('Unknown variable entered') 
    # get Matrix b in Ax = b
    if userExpr[-1] == 'A': b = app.A
    elif userExpr[-1] == 'B': b = app.B 
    elif userExpr[-1] == 'C': b = app.C
    else: raise Exception('Unknown variable entered') 

    # make the augmented Matrix M 
    M = A.makeAugmentedMatrx(b) 

    # CONDUCT RREF ON THE AUGMENTED MATRIX
    operations = RREFOperations(M) 
    # draw out the original augmented matrix 
    M.draw(canvas, app.answerFieldX, app.answerFieldY)
    indent += M.cols*unitSize + margin 
    # divid the operations into steps 
    steps = dividIntoSteps(operations)

    for step in steps: 
        # draw the arrow 
        canvas.create_line(app.answerFieldX+indent, app.answerFieldY, 
                           app.answerFieldX+indent+arrowLen, app.answerFieldY) 
        canvas.create_line(app.answerFieldX+indent+arrowLen, app.answerFieldY, 
                           app.answerFieldX+indent+arrowLen-margin, app.answerFieldY-margin) 
        canvas.create_line(app.answerFieldX+indent+arrowLen, app.answerFieldY, 
                           app.answerFieldX+indent+arrowLen-margin, app.answerFieldY+margin) 
        yIndent += margin
        # draw the text of operations 
        for op in step: 
            canvas.create_text(app.answerFieldX+indent, app.answerFieldY+yIndent,
                               text=op, anchor='nw')
            yIndent += textMargin 
            # formulate the matrix after this step
            M = M.rowOp(op) 
        
        indent +=arrowLen + margin
        yIndent = 0
        # the the intermediate matrix, not the final result
        if M != app.result: 
            M.draw(canvas, app.answerFieldX+indent, app.answerFieldY) 
            indent += M.cols*unitSize + margin  

    # done with the RREF process on augmented matrix 
    # present the result now 
    indent = 0
    yIndent = A.rows*unitSize + 5*margin # go to the second line 
    result = Matrix.solveLinearSystem(A, b) 
    if isinstance(result, Matrix): # the system has unique solution 
        canvas.create_text(app.answerFieldX+indent, app.answerFieldY+yIndent, 
                            text='The system has unique solution, x =', anchor='nw')
        indent += 250
        result.draw(canvas, app.answerFieldX+indent, app.answerFieldY+yIndent)
    elif isinstance(result, str): # the system has infinite solutions 
        canvas.create_text(app.answerFieldX+indent, app.answerFieldY+yIndent, 
                            text='The system has infinite solution, proceed with the simplified system:',
                            anchor='nw')
        yIndent += 2*textMargin
        canvas.create_text(app.answerFieldX+indent, app.answerFieldY+yIndent, 
                            text=result,
                            anchor='nw')
    elif result == None: 
        canvas.create_text(app.answerFieldX+indent, app.answerFieldY+yIndent, 
                            text='The system has no solution', anchor='nw')
        
    return 0 # since the nothing is drawn after, we return indent as 0

def drawEigenResult(app, canvas): 
    indent = 0
    rowSpace = 50
    # draw headings for e-values and e-vectors
    canvas.create_text(app.answerFieldX+indent, app.answerFieldY, 
                       text='Eigen Value(s):', anchor='nw')
    canvas.create_text(app.answerFieldX+indent, app.answerFieldY+rowSpace, 
                       text='Eigen Vector(s):', anchor='nw') 
    indent += 150

    # expr will be in forms like eigen(A) or eigen(2*A+B)...
    expr = app.exprToBeDrawn 
    expr = expr.replace('A', 'app.A').replace('B', 'app.B').replace('C', 'app.C')
    for eVal, eVec in eval(expr): 
        # draw e-value
        if isInt(eVal): 
            eVal = int(eVal) 
        elif len(str(eVal)) > 8: 
            eVal = '{:.2f}'.format(eVal) # just keep two digits after the decimal 
        canvas.create_text(app.answerFieldX+indent, app.answerFieldY, 
                           text=str(eVal), anchor='nw') 
        # draw e-vector 
        eVec.draw(canvas, app.answerFieldX+indent, app.answerFieldY+rowSpace) 
        indent += 150



    return 0 # no app.result draw, return 0 just for convention 




MyApp(width=1350, height=750) 
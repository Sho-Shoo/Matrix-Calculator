################################################
# TextField.py 
# TextField class
# Name: Tianze Shou 
# Andrew ID: tshou
################################################

# taken from 15-112 website 
# https://www.cs.cmu.edu/~112/index.html 
def make2dList(rows, cols):
    return [ ([0] * cols) for row in range(rows) ]

class TextField: 

    def __init__(self, x, y, width=70, height=25, textAnchor='e', font='Ariel'): 
        self.x, self.y = x, y
        self.message = ''
        self.selected = False 
        self.width = width
        self.height = height
        self.textMargin = 5
        self.fieldMargin = 15
        self.textAnchor = textAnchor 
        self.font = font

    def addMessage(self, s): 
        self.message += s

    def dropLastLetter(self): 
        self.message = self.message[0:len(self.message)-1]

    def selectionOn(self): 
        self.selected = True

    def selectionOff(self): 
        self.selected = False 

    def getMessage(self): 
        return self.message 

    def draw(self, canvas): 
        x, y = self.x, self.y
        # draw the box 
        if self.selected == True: 
            canvas.create_rectangle(x, y, x+self.width, y+self.height, 
                                    outline='blue', width=5) 
        else: 
            canvas.create_rectangle(x, y, x+self.width, y+self.height, 
                                    outline='black', width=5) 

        # draw the message 
        if self.textAnchor == 'e': 
            canvas.create_text(x+self.width-self.textMargin, (2*y+self.height)/2, 
                               text=self.message, anchor=self.textAnchor, 
                               font=self.font) 
        elif self.textAnchor == 'w': 
            canvas.create_text(x+self.textMargin, (2*y+self.height)/2, 
                               text=self.message, anchor=self.textAnchor,
                               font=self.font) 

    # return if the field is being pressed upon 
    def pressed(self, x, y): 
        return (self.x < x < self.x + self.width and 
                self.y < y < self.y + self.height)

    @staticmethod
    def make2dFields(rows, cols, x, y): 
        startX = x
        lst = make2dList(rows, cols)
        for row in range(rows): 
            for col in range(cols): 
                lst[row][col] = TextField(x, y) 
                x += lst[row][col].width + lst[row][col].fieldMargin
            y += lst[row][col].height + lst[row][col].fieldMargin
            x = startX

        return lst 


        

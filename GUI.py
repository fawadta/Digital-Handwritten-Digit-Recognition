import io
from tkinter import *
import tkinter as tk
from tkinter.ttk import *
import PIL.ImageGrab
from PIL import Image, ImageTk
from keras.models import load_model
import numpy as np
import PIL.Image



model = load_model('HANDWRITTEN.h5')

# create global variables
accurate = "Accuracy: "
operator = "Predicted Number: "
operator2 = ""


# create function to clear canvas and text
def Clear():
    cv.delete("all")
    global operator2
    text_input.set(operator2)
    accuracy.set(operator2)


# create function to predict and display predicted number
def Predict():
    file = 'image.jpg'
    if file:
        # save the canvas in jpg format
        x = root.winfo_rootx() + cv.winfo_x()
        y = root.winfo_rooty() + cv.winfo_y()
        x1 = x + cv.winfo_width()
        y1 = y + cv.winfo_height()
        PIL.ImageGrab.grab().crop((x, y, x1, y1)).save(file)

        # convert to greyscale
        img = Image.open(file).convert("L")

        # resize image
        img = img.resize((28, 28))

        # convert image to array
        im2arr = np.array(img)

        # reshape array
        im2arr = im2arr.reshape(1, 28, 28, 1)
        im2arr = im2arr/255.0

        # predict class
        y_pred2 = model.predict(im2arr)

        # covert class to scalar
        x = np.argmax(y_pred2[0])
        acc = max(y_pred2[0])
        acc = acc*100
        # display predicted number
        global operator, accurate
        operator = operator + str(x)
        text_input.set(operator)
        operator = operator = "Predicted Number: "
        accurate = accurate + str(acc)
        accuracy.set(accurate)
        accurate = accurate = "Accuracy: "


# create function to draw on canvas
def paint(event):
    old_x = event.x
    old_y = event.y

    cv.create_line(old_x, old_y, event.x, event.y,
                   width=20, fill="white",
                   capstyle=ROUND, smooth=TRUE, splinesteps=36)


# all interface elements must be between Tk() and mainloop()
root = Tk()

root.title("Handwritten Digit Recognition")
root.geometry('850x480')
# create string variable
text_input = StringVar()
accuracy = StringVar()

# create field to display text
textdisplay = Entry(root,
                    textvariable=text_input,
                    justify='center', width=25)

textdisplay1 = Entry(root,
                     textvariable=accuracy,
                     justify='center', width=25)


# create predict and clear buttons
btn1 = Button(root, text="Predict", command=lambda: Predict())
btn2 = Button(root, text="Clear", command=lambda: Clear())

# create canvas to draw on
cv = Canvas(root, width=415, height=415, bg="black", )

# using left mouse button to draw
cv.bind('<B1-Motion>', paint)

# organise the elements
cv.grid(row=0, column=0)
textdisplay.grid(row=0, column=1)
textdisplay1.grid(row=0, column=2, pady=10, padx=10)
btn1.grid(row=1, column=0)
btn2.grid(row=1, column=1)

lab = tk.Label(root, text="DRAW DIGIT ON CANVAS", width=25, height=1, fg="white",bg="midnightblue",
                font=('times', 14, ' bold '))
lab.place(x=520, y=80)

lab1 = tk.Label(root, text="HANDWRITTEN DIGIT RECOGNITION", width=35, height=1, fg="black",bg="white",
                font=('times', 16, ' bold '))
lab1.place(x=420, y=40)

lab2 = tk.Label(root, text="PREDICTION", width=15, height=1, fg="black",bg="white",
                font=('times', 12, ' bold '))
lab2.place(x=450, y=190)

lab3 = tk.Label(root, text="ACCURACY", width=15, height=1, fg="black",bg="white",
                font=('times', 12, ' bold '))
lab3.place(x=670, y=190)

lab4 = tk.Label(root, text="DEVELOPED BY:", width=15, height=1, fg="black",bg="white",
                font=('times', 12, ' bold '))
lab4.place(x=420, y=300)

lab5 = tk.Label(root, text="MUHAMMAD IRFAN", width=20, height=1, fg="black",bg="white",
                font=('times', 12, ' bold '))
lab5.place(x=540, y=330)
lab5 = tk.Label(root, text="FAWAD AHMAD", width=18, height=1, fg="black",bg="white",
                font=('times', 12, ' bold '))
lab5.place(x=540, y=360)

# this 2 lines for expand the interface
root.rowconfigure(0, weight=2)
root.columnconfigure(1, weight=2)

root.mainloop()

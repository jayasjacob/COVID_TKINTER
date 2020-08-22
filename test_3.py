from tkinter import *
import pyttsx3
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import (FigureCanvasTkAgg,
                                               NavigationToolbar2Tk)
import seaborn as sns
import tensorflow.keras
from tkinter import filedialog
from PIL import ImageTk, ImageOps,Image
import numpy as np

engine = pyttsx3.init()
window = Tk()
window.title('Covid Checker')


def predict_ct():
    model = tensorflow.keras.models.load_model('MODELS/keras_model.h5')
    data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
    filename = filedialog.askopenfilename(title="Open Image")
    test_image = Image.open(filename)
    image_show = ImageTk.PhotoImage(test_image)
    image_label = Label(window, image=image_show)
    image_label.image = image_show
    image_label.pack()
    np.set_printoptions(suppress=True)
    size = (224, 224)
    test_image = ImageOps.fit(test_image, size, Image.ANTIALIAS)
    image_array = np.asarray(test_image)
    normalized_image_array = (image_array.astype(np.float32) / 127.0) - 1
    data[0] = normalized_image_array
    prediction = model.predict(data)
    img = Image.open('image_400.jpg')
    im_show= ImageTk.PhotoImage(img)
    image_label_2 = Label(window, image=im_show,)
    image_label_2.image = im_show
    image_label_2.pack()
    lael = Label(window, text=prediction).pack()
    f, ax = plt.subplots(figsize=(11, 9))
    x_axis_label = "+-"
    sns.heatmap(prediction, Label="Covid", xticklabels=x_axis_label)
    canvas = FigureCanvasTkAgg(f,master=window)
    canvas.draw()
    canvas.get_tk_widget().pack()
    toolbar = NavigationToolbar2Tk(canvas,window)
    toolbar.update()
    canvas.get_tk_widget().pack()


window.geometry("500x500")
test_button = Button(master=window, command=predict_ct, height=2,
                     width=10,
                     text="Test",fg="black",bg='skyblue',font='Times 20 bold')

button_quit = Button(master=window,
                     command=window.destroy,
                     height=2,
                     width=10,
                     text="Quit",fg="black",bg='red',font='Times 20 bold')

test_button.pack(padx = 40,pady =30 , side = LEFT)
button_quit.pack(padx = 50,pady = 40,side = LEFT)

window.mainloop()
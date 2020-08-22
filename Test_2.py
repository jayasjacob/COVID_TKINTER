from tkinter import *
import pyttsx3
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import (FigureCanvasTkAgg,
                                               NavigationToolbar2Tk)
import seaborn as sns
import tensorflow.keras
from tkinter import filedialog
from PIL import ImageTk, Image, ImageOps
import numpy as np

# engine = pyttsx3.init()
window = Tk()
window.title('Covid Checker')


def predict(file):  # predict function
    model = tensorflow.keras.models.load_model('MODELS/keras_model.h5')  # loading the model
    data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
    test_image = Image.open(file)
    image_show = ImageTk.PhotoImage(test_image)
    image_label = Label(window,image = image_show)
    image_label.image = image_show
    image_label.pack()
    np.set_printoptions(suppress=True)
    size = (224, 224)
    test_image = ImageOps.fit(test_image, size, Image.ANTIALIAS)
    image_array = np.asarray(test_image)
    normalized_image_array = (image_array.astype(np.float32) / 127.0) - 1
    data[0] = normalized_image_array
    prediction = model.predict(data)
    print(prediction)
    lael = Label(window,text = prediction).pack()
    return prediction


def plot(prediction):
    f, ax = plt.subplots(figsize=(11, 9))
    x_axis_label = "+-"
    sns.heatmap(prediction, Label="Covid", xticklabels=x_axis_label)

    canvas = FigureCanvasTkAgg(f,
                               master=window)
    canvas.draw()

    canvas.get_tk_widget().pack()

    toolbar = NavigationToolbar2Tk(canvas,
                                   window)
    toolbar.update()

    canvas.get_tk_widget().pack()
def run():
    filename = filedialog.askopenfilename(title = "Open Image")
    predict(filename)
    plot(prediction)

window.geometry("500x500")
test_button = Button(master=window,command = run,height=2,
                     width=10,
                     text="Test")


# plot_button = Button(master=window,
#                      command=plot,
#                      height=2,
#                      width=10,
#                      text="Plot")




test_button.pack()
# plot_button.pack()


window.mainloop()

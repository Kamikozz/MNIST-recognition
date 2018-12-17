<<<<<<< HEAD
#
from tkinter import *
from tkinter import messagebox
from neuralnetwork import neuralnetwork
from skimage import data,io,filters
import numpy as np


class Paint(Frame):

    def __init__(self, parent):
        Frame.__init__(self, parent)


        self.parent = parent
        self.color = "black"
        self.brush_size = 7
        self.setUI()

    def set_color(self, new_color):
        self.color = new_color

    def set_brush_size(self, new_size):
        self.brush_size = new_size

    def draw(self, event):
        self.canv.create_oval(event.x - self.brush_size,
                              event.y - self.brush_size,
                              event.x + self.brush_size,
                              event.y + self.brush_size,
                              fill=self.color, outline=self.color)

    def show_message(self, event):
        # save canvas to .eps (postscript) file
        # image will be grayscaled
        self.scale = 28/self.canv.winfo_width()
        self.canv.scale(ALL,0,0,self.scale,self.scale)
        self.canv.postscript(file="tmp_canvas.eps",
                          colormode="gray",
                          width=int(self.canv.winfo_width()*self.scale),
                          height=int(self.canv.winfo_height()*self.scale),
                          pagewidth=int(self.canv.winfo_width()*self.scale) - 1,
                          pageheight=int(self.canv.winfo_height()*self.scale) - 1)
        self.canv.scale(ALL, 0, 0, 1/self.scale,1/self.scale)

        # read the postscript data
        # [row][column][R G B]
        # data = numpy ndarray (n-dimensional array). Final picture = 404x404 (from original canvas = 400x400)
        data = io.imread('tmp_canvas.eps')
        data=data.reshape(784,3)
        d1=np.zeros(len(data))
        for i in range(len(data)):
            d1[i]=255-data[i][0]
        data=d1.astype('float32')
        data/=255
        


        result =neuralnetwork(data)

        messagebox.showinfo("GUI Python", result)

    def setUI(self):
        self.parent.title("GUI for CNN")  # Устанавливаем название окна
        self.pack(fill=BOTH, expand=1)  # Размещаем активные элементы на родительском окне

        self.columnconfigure(6, weight=1) # Даем седьмому столбцу возможность растягиваться, благодаря чему кнопки не будут разъезжаться при ресайзе
        self.rowconfigure(3, weight=1) # То же самое для третьего ряда

        self.canv = Canvas(self, bg="white", width=400, height=400)  # Создаем поле для рисования, устанавливаем белый фон
        self.canv.grid(row=3, column=0, columnspan=7, sticky=N+W)  # Прикрепляем канвас методом grid. Он будет находится в 3м ряду, первой колонке, и будет занимать 7 колонок, задаем отступы по X и Y в 5 пикселей, и заставляем растягиваться при растягивании всего окна
        self.canv.bind("<B1-Motion>", self.draw) # Привязываем обработчик к канвасу. <B1-Motion> означает "при движении зажатой левой кнопки мыши" вызывать функцию draw


        color_lab = Label(self, text="Color: ") # Создаем метку для кнопок изменения цвета кисти
        color_lab.grid(row=0, column=0,pady=5, sticky=N) # Устанавливаем созданную метку в первый ряд и первую колонку



        red_btn = Button(self, text="Red", width=10, command=lambda: self.set_color("red"))
        red_btn.grid(row=0, column=1) # Устанавливаем кнопку

        green_btn = Button(self, text="Green", width=10, command=lambda: self.set_color("green"))
        green_btn.grid(row=0, column=2)

        blue_btn = Button(self, text="Blue", width=10, command=lambda: self.set_color("blue"))
        blue_btn.grid(row=0, column=3)

        black_btn = Button(self, text="Black", width=10, command=lambda: self.set_color("black"))
        black_btn.grid(row=0, column=4)

        white_btn = Button(self, text="White", width=10, command=lambda: self.set_color("white"))
        white_btn.grid(row=0, column=5)

        clear_btn = Button(self, text="Clear all", width=10, command=lambda: self.canv.delete("all"))
        clear_btn.grid(row=0, column=6,sticky=E)



        size_lab = Label(self, text="Brush size: ")
        size_lab.grid(row=1, column=0,pady=5)
        # one_btn = Button(self, text="Two", width=10,
        #                  command=lambda: self.set_brush_size(2))
        # one_btn.grid(row=1, column=1)
        #
        # two_btn = Button(self, text="Five", width=10,
        #                  command=lambda: self.set_brush_size(5))
        # two_btn.grid(row=1, column=2)

        five_btn = Button(self, text="Seven", width=10, command=lambda: self.set_brush_size(7))
        five_btn.grid(row=1, column=1)

        seven_btn = Button(self, text="Ten", width=10, command=lambda: self.set_brush_size(10))
        seven_btn.grid(row=1, column=2)

        ten_btn = Button(self, text="Twenty", width=10, command=lambda: self.set_brush_size(20))
        ten_btn.grid(row=1, column=3)

        twenty_btn = Button(self, text="Fifty", width=10, command=lambda: self.set_brush_size(50))
        twenty_btn.grid(row=1, column=4)

        # def show_message(self):
        #     result = CNN.calc(message.get(), canvas=self.canv)
        #     messagebox.showinfo("GUI Python", result)
        self.message = StringVar()
        message_entry = Entry(self, textvariable=self.message)
        message_entry.grid(row=2, columnspan=7)

        self.canv.bind("<ButtonRelease-1>",
                       self.show_message)  # Привязываем обработчик к канвасу. <B1-Motion> означает "при движении зажатой левой кнопки мыши" вызывать функцию draw

        # # save canvas to .eps (postscript) file
        # self.canv.postscript(file="tmp_canvas.eps",
        #                   colormode="color",
        #                   width=self.canv.winfo_width(),
        #                   height=self.canv.winfo_height(),
        #                   pagewidth=self.canv.winfo_width() - 1,
        #                   pageheight=self.canv.winfo_height() - 1)

        # # read the postscript data
        # data = io.imread("tmp_canvas.eps")
        #
        # # write a rasterized png file
        # io.imsave("canvas_image.png", data)

        #message_entry.place(relx=.5,  anchor="c")


        #message_entry = Entry(textvariable=message)
        #message_entry.grid(row=2, column=0)
        #message_entry.place(relx=.5, rely=.1, anchor="c")

        #message_button = Button(self, text="Click Me", command=show_message(message))
        #message_button.place(relx=.5, rely=.5, anchor="c")






def main():
    root = Tk()
    root.geometry("600x500+500+300")
    app = Paint(root)
    root.mainloop()


if __name__ == '__main__':
    main()


=======
import numpy as np
>>>>>>> e71ac41c8816297de2a5ea2768f4fea42e7fb447

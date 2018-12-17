#
from tkinter import *
from tkinter import messagebox
from main import CNN
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
        # write a rasterized png file
        io.imsave("canvas_image.png", data)
        data = data.reshape(28*28,3)
        res = []
        for i in data:
            res.append(255-i[0])
        res = np.array(res)
        print(res)

        #data = [i[0] for i in data]
        # print(data)
        # print(data.ndim)
        # print(data[185][24])
        # print(data[186][24])
        # print(data[185][25])
        # print(data[186][25])
        # print(data[188][25])
        # print(data[187][24])
        # print(data[25][186])
        # print(data[25][187])
        # print(data[25][216])
        # print(data[25][217])
        # print(data[0])
        #k = 0
        # with open('somefile.txt', 'w') as the_file:
        #     for i in data:
        #         the_file.write(str(k))
        #         for j in i:
        #             the_file.write(str(j))
        #             the_file.write('\n')
        #         k+=1

        # print(data[403].__len__())

        # #write a rasterized png file
        # io.imsave("canvas_image.png", data)

        result = CNN.calc(self.message.get())

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





# #!/usr/bin/env python
#
# __author__ = "Dmitriy Krasota aka g0t0wasd"
#
# # An example of Quadratic Calc using Tkinter.
# # More at http://pythonicway.com/index.php/python-examples/python-gui-examples/14-python-tkinter-quadratic-equations
#
#
# from tkinter import *
# from math import sqrt
#
# def solver(a,b,c):
#     """ Solves quadratic equation and returns the result in formatted string """
#     D = b*b - 4*a*c
#     if D >= 0:
#         x1 = (-b + sqrt(D)) / (2*a)
#         x2 = (-b - sqrt(D)) / (2*a)
#         text = "The discriminant is: %s \n X1 is: %s \n X2 is: %s \n" % (D, x1, x2)
#     else:
#         text = "The discriminant is: %s \n This equation has no solutions" % D
#     return text
#
# def inserter(value):
#     """ Inserts specified value into text widget """
#     output.delete("0.0","end")
#     output.insert("0.0",value)
#
# def clear(event):
#     """ Clears entry form """
#     caller = event.widget
#     caller.delete("0", "end")
#
# def handler():
#     """ Get the content of entries and passes result to the text """
#     try:
#         # make sure that we entered correct values
#         a_val = float(a.get())
#         b_val = float(b.get())
#         c_val = float(c.get())
#         inserter(solver(a_val, b_val, c_val))
#     except ValueError:
#         inserter("Make sure you entered 3 numbers")
#
# root = Tk()
# root.title("Quadratic calculator")
# root.minsize(325,230)
# root.resizable(width=False, height=False)
#
#
# frame = Frame(root)
# frame.grid()
#
# a = Entry(frame, width=3)
# a.grid(row=1,column=1,padx=(10,0))
# a.bind("<FocusIn>", clear)
# a_lab = Label(frame, text="x**2+").grid(row=1,column=2)
#
# b = Entry(frame, width=3)
# b.bind("<FocusIn>", clear)
# b.grid(row=1,column=3)
# b_lab = Label(frame, text="x+").grid(row=1, column=4)
#
# c = Entry(frame, width=3)
# c.bind("<FocusIn>", clear)
# c.grid(row=1, column=5)
# c_lab = Label(frame, text="= 0").grid(row=1, column=6)
#
# but = Button(frame, text="Solve", command=handler).grid(row=1, column=7, padx=(10,0))
#
# output = Text(frame, bg="lightblue", font="Arial 12", width=35, height=10)
# output.grid(row=2, columnspan=8)
#
# root.mainloop()
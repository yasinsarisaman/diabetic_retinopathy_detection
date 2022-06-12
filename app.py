# -*- coding: utf-8 -*-
"""
Created on Wed Mar 23 21:22:12 2022

@author: yasin
"""

import tkinter as tk
from PIL import Image, ImageTk
from tkinter.filedialog import askopenfile
#import predict
import pred_TT


app = tk.Tk()
canvas = tk.Canvas(app, width=1200, height=600)
canvas.grid(columnspan=5, rowspan=10)

def create_logo(logoImage,cl,rw,rwsp,clsp):
    logo = Image.open(logoImage)
    logo = ImageTk.PhotoImage(logo)
    logo_label = tk.Label(image=logo)
    logo_label.image = logo
    logo_label.grid(column=cl,row=rw,rowspan=rwsp,columnspan=clsp)
    
def print_prediction(pred_res):
    if pred_res<0.15:
        prediction = "NO DIABETIC RETINOPATHY"
    else:
        prediction = "DIABETIC RETINOPATHY"
    return prediction;
    
create_logo("logo.png",2,0,1,1)

#Footer text 
instructions = tk.Label(app, text="Hastalık tahmini yapılacak olan fotoğrafı (.png, .jpg) bilgisayarınızdan seçin")
instructions.grid(column=2,row=1)

#Choosing Image
def open_file():
    browse_text.set("Yükleniyor...")
    DRfile = askopenfile(parent=app, mode="rb", title="Bir fotoğraf seçin", filetype=[("Png Dosyası", "*.png"), ("Jpeg Dosyası", "*.jpeg"), ("Jpg Dosyası", "*.jpg")]) 
    if DRfile:
        choosenImagePath = tk.Label(app, text="Seçilen dosya: " + DRfile.name)
        choosenImagePath.grid(column=2,row=3)
        browse_text.set("Bir fotoğraf seçin")
        create_logo(DRfile.name,2,4,1,1)
        resultF = pred_TT.predict_single_img(DRfile.name)
        print(resultF)
        resultS = print_prediction(resultF)
        result_label = tk.Label(app, text="Tahmin: " + resultS)
        result_label.grid(column=2, row=5)
        
#browse button
browse_text = tk.StringVar()
browse_button = tk.Button(app, command=lambda:open_file(), textvariable=browse_text, font="Raleway", bg="#0e0872", fg="white", width=15, height=1)
browse_text.set("Bir fotoğraf seç")
browse_button.grid(column=2,row=2)


#another canvas for padding
canvas = tk.Canvas(app, width=600, height=200)
canvas.grid(columnspan=3)

app.mainloop()
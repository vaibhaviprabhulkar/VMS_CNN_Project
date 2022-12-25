import firebase_admin
from firebase_admin import credentials
from firebase_admin import db
import cv2
import numpy as np
import face_recognition
MODEL = 'cnn'
# TOLERANCE = 0.6

from tkinter import messagebox

from tkinter import *
import tkinter.ttk
from PIL import ImageTk, Image
import os

import pyrebase

firebaseConfig = {
    "apiKey": "AIzaSyBbZAJE3n9-ye0GLlwNrRNdW0uxYnC5uc8",
    "authDomain": "vizigo-8d625.firebaseapp.com",
    "databaseURL": "https://vizigo-8d625.firebaseio.com",
    "projectId": "vizigo-8d625",
    "storageBucket": "vizigo-8d625.appspot.com",
    "ServiceAccount": "firebase_SDK.json"
  };

root = Tk()
root.title("ViziGo: Face Recognition Based Visitor Management System")
root.geometry("800x780+300+400")
root.iconbitmap("go.ico")
root.configure(bg="#f4e8fc")

img = Image.open("wow.png")
img = img.resize((570, 450))
my_img = ImageTk.PhotoImage(img)
my_label = Label(image=my_img)
my_label.pack(pady=15)


def button_command():
    text = entry1.get()
    print(text)
    firebase_storage = pyrebase.initialize_app(firebaseConfig)
    storage = firebase_storage.storage()

    # storage.child('Images/test_download.jpg').put('test_download.jpg')
    luck = str(text)
    path = 'Images/' + luck + '.jpg'
    print(path)
    storage.child(path).download('finally.jpg')
    entry1.delete(0,END)
    return NONE


l1 =Label(root, text="Enter your Contact Number: ", font=("Helvetica", 14, "bold"))
l1.pack()
entry1 = Entry(root, width=22, font=("Times", 20))
entry1.pack()

Button(root, text= "Submit",fg = "black", bg="#5fb6f5",activebackground="#5fb6f5",
                                activeforeground="#c5def0",  font=("Helvetica", 18, "bold"),width=20, relief= "raised",command=button_command).pack()


def onClick1():
    myLabel=Label(root, text="Image Saved Successfully!")
    myLabel.pack()
    os.system('python webcam-capture.py')

btn_capture_your_photo = Button(root, text="Capture Your Photo", fg = "black", bg="#5fb6f5", activebackground="#5fb6f5",
                                activeforeground="#c5def0", font=("Helvetica", 18, "bold"), width=34, relief= "raised",
                                command=onClick1)
btn_capture_your_photo.pack(pady=10)


def popup():
    if results == [TRUE]:
    # if results:
        messagebox.showinfo("info", "Face Verified Successfully!")
    else:
        messagebox.showinfo("info", "Face NOT Verified! FAILURE Exists!")

def onClick2():
    global results
    print("Please wait processing your images...")

    sampleImage = face_recognition.load_image_file('finally.jpg')
    sampleImage = cv2.cvtColor(sampleImage, cv2.COLOR_BGR2RGB)

    testImage = face_recognition.load_image_file(
        'saved_img.jpg')

    testImage = cv2.cvtColor(testImage, cv2.COLOR_RGB2BGR)

    sampleLocations = face_recognition.face_locations(sampleImage, model=MODEL)[0]  # face location coordinates of image1

    print("\nImage 1 Face Locations :", sampleLocations)
    print("\nEncoding vector of 128 values of Image1.....\n")

    sampleEncodings = face_recognition.face_encodings(sampleImage)[0]  # "face_encodings" function to encode the image
    print(sampleEncodings)

    cv2.rectangle(sampleImage, (sampleLocations[3], sampleLocations[0]), (sampleLocations[1], sampleLocations[2]),
                  (255, 0, 255), 2)
    testLocations = face_recognition.face_locations(testImage, model=MODEL)[0]

    print("\nImage 2 Face Locations", testLocations)  # face location coordinates of image1
    print("\nEncoding vector of 128 values of Image 2.....\n")
    testEncodings = face_recognition.face_encodings(testImage)[0]  # "face_encodings" function to encode the image
    print(testEncodings)

    cv2.rectangle(testImage, (testLocations[3], testLocations[0]), (testLocations[1], testLocations[2]), (255, 0, 255),
                  2)
    cv2.imshow('neha1', sampleImage)
    cv2.imshow('image2', testImage)
    cv2.waitKey(12000)
    cv2.destroyAllWindows()

    print("\n\nFetching the result...")

    results = face_recognition.compare_faces([sampleEncodings],
                                             testEncodings)  # compares the encoding of both the image
    print("The result of Features Extraction :  ", results)

    # if results:
    #     myLabel = Label(root, text="Face Verified Successfully!")
    #     myLabel.pack()
    #
    # else:
    #     myLabel = Label(root, text="Face NOT Verified! FAILURE Exists!")
    #     myLabel.pack()
    #     my_label.destroy()

btn_recognize_me = Button(root, text="Recognize Me", fg = "black", bg="#5fb6f5", activebackground="#5fb6f5",
                          activeforeground="#c5def0", font=("Helvetica", 18, "bold"), width=34, relief= "raised",
                          command=lambda:[onClick2(), popup()])
btn_recognize_me.pack(pady=10)
root.mainloop()
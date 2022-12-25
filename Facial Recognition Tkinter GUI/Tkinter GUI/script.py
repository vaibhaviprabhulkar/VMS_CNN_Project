import firebase_admin
from firebase_admin import credentials
from firebase_admin import db
import cv2
import numpy as np
import face_recognition
MODEL = 'cnn'
# TOLERANCE = 0.6





print("Please wait processing your images...")


# in do lines ki jagah pe hame firebase se get karna hai
# haa
sampleImage = face_recognition.load_image_file('testing.jpg')
sampleImage = cv2.cvtColor(sampleImage, cv2.COLOR_BGR2RGB)




testImage = face_recognition.load_image_file('saved_img.jpg') # yeh jaise saved img kiya hai waise hi msain file mien hi get krke save kar de toh?



# samajhlo re

testImage = cv2.cvtColor(testImage, cv2.COLOR_RGB2BGR)

sampleLocations = face_recognition.face_locations(sampleImage, model=MODEL)[0]   # face location coordinates of image1

print("\nImage 1 Face Locations :", sampleLocations)
print("\nEncoding vector of 128 values of Image1.....\n")

sampleEncodings = face_recognition.face_encodings(sampleImage)[0]  # "face_encodings" function to encode the image
print(sampleEncodings)

cv2.rectangle(sampleImage, (sampleLocations[3], sampleLocations[0]), (sampleLocations[1], sampleLocations[2]), (255, 0, 255), 2)
testLocations = face_recognition.face_locations(testImage, model=MODEL)[0]

print("\nImage 2 Face Locations", testLocations)    # face location coordinates of image1
print("\nEncoding vector of 128 values of Image 2.....\n")
testEncodings = face_recognition.face_encodings(testImage)[0]    # "face_encodings" function to encode the image
print(testEncodings)

cv2.rectangle(testImage, (testLocations[3], testLocations[0]), (testLocations[1], testLocations[2]), (255, 0, 255), 2)
cv2.imshow('neha1', sampleImage)
cv2.imshow('image2', testImage)
cv2.waitKey(12000)
cv2.destroyAllWindows()

print("\n\nFetching the result...")

results = face_recognition.compare_faces([sampleEncodings], testEncodings)  # compares the encoding of both the image
print("The result of Features Extraction :  ", results)

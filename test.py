# Importing KNeighborsClassifier from scikit-learn for implementing K-Nearest Neighbors algorithm
from sklearn.neighbors import KNeighborsClassifier

# OpenCV for computer vision tasks such as image processing, video capture, and face detection
import cv2

# Pickle for serializing and deserializing Python objects (saving and loading data)
import pickle

# NumPy for numerical operations on arrays and matrices
import numpy as np

# OS module for interacting with the operating system, used here for file operations
import os

# CSV module for reading and writing CSV files
import csv

# Time module for handling time-related functions
import time

# Datetime module for manipulating dates and times
from datetime import datetime

# Dispatch module from win32com.client for integrating with Windows SAPI for speech synthesis
from win32com.client import Dispatch


# Function to speak text using Windows SAPI
def speak(text):
    speak = Dispatch("SAPI.SpVoice")
    speak.Speak(text)

# Capture video from webcam
video = cv2.VideoCapture(0)

# Load pre-trained Haar Cascade classifier for face detection
facedetect = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Load pre-trained labels (names) and face data from pickle files
with open('data/names.pkl', 'rb') as w:
    LABELS = pickle.load(w)
with open('data/faces_data.pkl', 'rb') as f:
    FACES = pickle.load(f)

# Initialize and fit K-Nearest Neighbors classifier
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(FACES, LABELS)

# Load background image for display
imgBackground = cv2.imread("newbackground.png")

# Define column names for attendance CSV file
COL_NAMES = ['NAME', 'TIME']

# Variable to store recognized name
recognized_name = None

# Main loop to capture frames from webcam
while True:
    ret, frame = video.read()  # Read frame from webcam
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # Convert frame to grayscale
    faces = facedetect.detectMultiScale(gray, 1.3, 5)  # Detect faces in the frame
    recognized_name = None  # Reset recognized_name for each frame
    
    # Iterate through detected faces
    for (x, y, w, h) in faces:
        crop_img = frame[y:y+h, x:x+w, :]  # Crop detected face region
        resized_img = cv2.resize(crop_img, (50, 50)).flatten().reshape(1, -1)  # Resize and flatten for prediction
        output = knn.predict(resized_img)  # Predict label (name) for the face
        probability = knn.predict_proba(resized_img)  # Predict probabilities
        
        # Check confidence level and perform actions accordingly
        if max(probability[0]) < 0.7:  # Adjust confidence threshold as needed
            speak("It is not present in database")
        else:
            recognized_name = str(output[0])  # Store recognized name
            ts = time.time()
            date = datetime.fromtimestamp(ts).strftime("%d-%m-%Y")
            timestamp = datetime.fromtimestamp(ts).strftime("%H:%M:%S")
            exist = os.path.isfile("Attendance/Attendance_" + date + ".csv")  # Check if attendance file exists
            
            # Draw rectangle and text on frame for recognized face
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 1)
            cv2.rectangle(frame, (x, y), (x+w, y+h), (50, 50, 255), 2)
            cv2.rectangle(frame, (x, y-40), (x+w, y), (50, 50, 255), -1)
            cv2.putText(frame, recognized_name, (x, y-15), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 1)
            cv2.rectangle(frame, (x, y), (x+w, y+h), (50, 50, 255), 1)
            
            attendance = [recognized_name, timestamp]  # Prepare attendance record
    
    # Display annotated frame on background image
    imgBackground[162:162 + 480, 55:55 + 640] = frame
    cv2.imshow("Frame", imgBackground)  # Show the frame with annotations
    k = cv2.waitKey(1)  # Wait for user input
    
    # Handle 'o' key press to take attendance
    if k == ord('o'):
        if recognized_name is not None:
            speak(f"Attendance taken for {recognized_name}")  # Speak attendance confirmation
            time.sleep(5)  # Wait for 5 seconds
            if exist:
                with open("Attendance/Attendance_" + date + ".csv", "+a") as csvfile:
                    writer = csv.writer(csvfile)
                    writer.writerow(attendance)  # Write attendance record
                csvfile.close()
            else:
                with open("Attendance/Attendance_" + date + ".csv", "+a") as csvfile:
                    writer = csv.writer(csvfile)
                    writer.writerow(COL_NAMES)  # Write column names if file doesn't exist
                    writer.writerow(attendance)  # Write attendance record
                csvfile.close()
        else:
            speak("Face is not clear")  # Speak if face recognition confidence is low
    
    # Handle 'q' key press to quit the program
    if k == ord('q'):
        break

video.release()  # Release the webcam
cv2.destroyAllWindows()  # Close all OpenCV windows

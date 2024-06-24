# OpenCV for image processing and computer vision tasks
import cv2

# Pickle for serialization and deserialization of Python objects
import pickle

# NumPy for numerical operations on arrays and matrices
import numpy as np

# OS module for interacting with the operating system (e.g., file operations)
import os

# Function to ensure the data directory exists
def ensure_data_dir_exists(directory='data'):
    if not os.path.exists(directory):
        os.makedirs(directory)

# Function to save face data and names into pickle files
def save_faces_and_names(name, faces_data, data_dir='data'):
    faces_data = np.asarray(faces_data)  # Convert list to numpy array
    faces_data = faces_data.reshape(len(faces_data), -1)  # Reshape data to 2D array

    names_file = os.path.join(data_dir, 'names.pkl')  # Path for names pickle file
    faces_file = os.path.join(data_dir, 'faces_data.pkl')  # Path for faces pickle file

    if not os.path.exists(names_file):  # If names file doesn't exist, create it
        names = [name] * len(faces_data)  # Create list of names
    else:  # If names file exists, load it and append new names
        with open(names_file, 'rb') as f:
            names = pickle.load(f)
        names = names + [name] * len(faces_data)

    # Save the updated names list to the pickle file
    with open(names_file, 'wb') as f:
        pickle.dump(names, f)

    if not os.path.exists(faces_file):  # If faces file doesn't exist, create it
        faces = faces_data
    else:  # If faces file exists, load it and append new faces data
        with open(faces_file, 'rb') as f:
            faces = pickle.load(f)
        faces = np.append(faces, faces_data, axis=0)

    # Save the updated faces data to the pickle file
    with open(faces_file, 'wb') as f:
        pickle.dump(faces, f)

# Main function to capture face data
def main():
    ensure_data_dir_exists()  # Ensure the data directory exists

    video = cv2.VideoCapture(0)  # Start video capture from webcam
    facedetect = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')  # Load face detection model
    faces_data = []  # List to store face data
    i = 0

    name = input("Enter Your Name: ")  # Prompt user to enter their name

    while True:
        ret, frame = video.read()  # Read frame from webcam
        if not ret:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # Convert frame to grayscale
        faces = facedetect.detectMultiScale(gray, 1.3, 5)  # Detect faces in the frame

        for (x, y, w, h) in faces:  # For each detected face
            crop_img = frame[y:y+h, x:x+w, :]  # Crop the face from the frame
            resized_img = cv2.resize(crop_img, (50, 50))  # Resize the face image
            if len(faces_data) < 100 and i % 10 == 0:  # If less than 100 face images collected and current frame is every 10th frame
                faces_data.append(resized_img)  # Add resized face to list
            i += 1
            cv2.putText(frame, str(len(faces_data)), (50, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (50, 50, 255), 1)  # Display the number of collected faces
            cv2.rectangle(frame, (x, y), (x+w, y+h), (50, 50, 255), 1)  # Draw rectangle around the face

        cv2.imshow("Frame", frame)  # Display the frame
        k = cv2.waitKey(1)  # Wait for a key press
        if k == ord('q') or len(faces_data) == 100:  # Exit loop if 'q' is pressed or 100 faces are collected
            break

    video.release()  # Release the webcam
    cv2.destroyAllWindows()  # Close all OpenCV windows

    save_faces_and_names(name, faces_data)  # Save the collected faces and name

if __name__ == '__main__':
    main()  # Run the main function

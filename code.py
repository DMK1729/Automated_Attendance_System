import os
import face_recognition
import cv2
import numpy as np
import pandas as pd
from datetime import datetime
import pickle

# Load the trained model
print('loaded')
def load_trained_model(model_path):
    with open(model_path, 'rb') as f:
        known_face_encodings, known_face_names = pickle.load(f)
    return known_face_encodings, known_face_names

# Mark attendance in the CSV file
def mark_attendance(name):
    now = datetime.now()
    dt_string = now.strftime('%Y-%m-%d %H:%M:%S')
    
    # Check if the file exists
    if not os.path.isfile('attendance.csv'):
        # Create the file with the header if it doesn't exist
        attendance_data = pd.DataFrame(columns=['Name', 'DateTime'])
        attendance_data.to_csv('attendance.csv', index=False)
    
    # Read the existing CSV file
    attendance_data = pd.read_csv('attendance.csv')
    
    # Add the new attendance record
    new_record = pd.DataFrame([[name, dt_string]], columns=['Name', 'DateTime'])
    attendance_data = pd.concat([attendance_data, new_record], ignore_index=True)
    
    # Save the updated attendance data back to the CSV file
    attendance_data.to_csv('attendance.csv', index=False)
# Train the model
def train_model(dataset_dir):
    known_face_encodings = []
    known_face_names = []
    for person_name in os.listdir(dataset_dir):
        print(person_name)
        person_dir = os.path.join(dataset_dir, person_name)
        for image_name in os.listdir(person_dir):
            image_path = os.path.join(person_dir, image_name)
            image = cv2.imread(image_path)
            if image is not None:
                # Convert the image to RGB
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                # Ensure the image is in the correct format
                if len(image.shape) == 3 and image.shape[2] == 3:
                    image = np.array(image, dtype=np.uint8)
                    face_encodings = face_recognition.face_encodings(image)
                    if len(face_encodings) > 0:
                        face_encoding = face_encodings[0]
                        known_face_encodings.append(face_encoding)
                        known_face_names.append(person_name)
                else:
                    print(f"Error: Image {image_path} is not a 3-channel RGB image.")
            else:
                print(f"Error reading image: {image_path}")
    with open('trained_model.pkl', 'wb') as f:
        pickle.dump((known_face_encodings, known_face_names), f)

# Recognize faces from video
def recognize_faces_from_video(model_path):
    known_face_encodings, known_face_names = load_trained_model(model_path)
    video_capture = cv2.VideoCapture(0)
    
    while True:
        ret, frame = video_capture.read()
        if not ret:
            break
        
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        small_frame = cv2.resize(frame_rgb, (0, 0), fx=0.25, fy=0.25)
        
        face_locations = face_recognition.face_locations(small_frame)
        face_encodings = face_recognition.face_encodings(small_frame, face_locations)
        
        face_names = []
        for face_encoding in face_encodings:
            matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
            name = "Unknown"
            face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
            best_match_index = np.argmin(face_distances)
            if matches[best_match_index]:
                name = known_face_names[best_match_index]
            face_names.append(name)
            if name != "Unknown":
                mark_attendance(name)
        
        # Display the results
        for (top, right, bottom, left), name in zip(face_locations, face_names):
            top *= 4
            right *= 4
            bottom *= 4
            left *= 4
            
            # Draw a box around the face
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
            
            # Draw a label with a name below the face
            cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
            font = cv2.FONT_HERSHEY_DUPLEX
            cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)
        
        # Display the resulting image
        cv2.imshow('Video', frame)
        
        # Hit 'q' on the keyboard to quit!
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    video_capture.release()
    cv2.destroyAllWindows()
# Main function
if __name__ == '__main__':
    dataset_dir = './dataset'
    train_model(dataset_dir)
    recognize_faces_from_video('trained_model.pkl')
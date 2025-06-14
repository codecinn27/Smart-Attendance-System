
import cv2
import os
from facenet_pytorch import MTCNN
import torch
import sqlite3

# Initialize MTCNN for face detection
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
mtcnn = MTCNN(
    keep_all=True,       # Detect all faces in the frame
    min_face_size=60,    # Adjust based on face distance from camera
    thresholds=[0.6, 0.7, 0.7],  # Detection thresholds (lower = more sensitive)
    device=device
)

# Initialize webcam
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

# Database function (same as before)
def insertOrUpdate(Id, Name, age):
    conn = sqlite3.connect("sqlite.db")
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM students WHERE Id=?", (Id,))
    isRecordExist = cursor.fetchone() is not None
    
    if isRecordExist:
        cursor.execute("UPDATE students SET Name=?, age=? WHERE Id=?", (Name, age, Id))
    else:
        cursor.execute("INSERT INTO students (Id, Name, age) VALUES (?, ?, ?)", (Id, Name, age))
    
    conn.commit()
    conn.close()

# Get user input
Id = input("Enter your ID: ")
Name = input("Enter your name: ")
age = input("Enter your age: ")

# Insert into database
insertOrUpdate(Id, Name, age)

# Create user directory if it doesn't exist
user_dir = os.path.join('dataset', Name)
os.makedirs(user_dir, exist_ok=True)

# Capture face samples
sample_num = 0
while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Detect faces using MTCNN
    boxes, _ = mtcnn.detect(frame)

    if boxes is not None:
        for box in boxes:
            x1, y1, x2, y2 = map(int, box)
            
            # Extract face and save
            face_img = frame[y1:y2, x1:x2]
            if face_img.size == 0:
                continue  # Skip if face is too small
            
            # Save face image
            sample_num += 1
            save_path = os.path.join(user_dir, f"{Id}_{sample_num}.jpg")
            cv2.imwrite(save_path, face_img)
            
            # Draw bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f"Samples: {sample_num}", (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    # Show the frame
    cv2.imshow("Face Data Collection (MTCNN)", frame)
    
    # Press 'q' to quit or collect enough samples
    if cv2.waitKey(1) & 0xFF == ord('q') or sample_num >= 30:
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
print(f"Successfully collected {sample_num} samples for {Name}!")
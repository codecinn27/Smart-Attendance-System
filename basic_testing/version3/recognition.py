import cv2
import torch
import torch.nn.functional as F
from facenet_pytorch import InceptionResnetV1, MTCNN
import pickle
import numpy as np

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Load precomputed embeddings
try:
    with open('embeddings.pkl', 'rb') as f:
        embeddings_db = pickle.load(f)
    # Move all embeddings to the current device
    embeddings_db = {name: emb.to(device) for name, emb in embeddings_db.items()}
    print(f"Loaded embeddings for {len(embeddings_db)} identities")
except FileNotFoundError:
    print("Error: embeddings.pkl not found. Please run trainer.py first.")
    exit()

# Initialize models with proper settings
mtcnn = MTCNN(
    image_size=160,  # FaceNet requires 160x160 input
    margin=20,      # Add margin to prevent too-small crops
    keep_all=True,
    min_face_size=40,  # Increase minimum face size
    device=device,
    post_process=False
)

resnet = InceptionResnetV1(pretrained='vggface2').eval().to(device)

def preprocess_face(face_img, target_size=160):
    """Ensure face image is properly sized and normalized"""
    # Resize to target size
    face_img = cv2.resize(face_img, (target_size, target_size))
    # Convert to tensor and normalize
    face_tensor = torch.from_numpy(face_img).permute(2, 0, 1).float()
    face_tensor = (face_tensor - 127.5) / 128.0  # Normalize to [-1, 1]
    return face_tensor.unsqueeze(0).to(device)  # Add batch dimension

def recognize_faces(frame, threshold=0.7):
    # Convert to RGB
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Detect faces
    boxes, probs, landmarks = mtcnn.detect(frame_rgb, landmarks=True)
    
    if boxes is not None:
        for i, box in enumerate(boxes):
            x1, y1, x2, y2 = [int(coord) for coord in box]
            
            # Skip if the box is invalid
            if x2 <= x1 or y2 <= y1:
                continue
                
            # Extract face with some padding
            pad = 20  # Additional padding
            h, w = frame.shape[:2]
            y1 = max(0, y1 - pad)
            y2 = min(h, y2 + pad)
            x1 = max(0, x1 - pad)
            x2 = min(w, x2 + pad)
            
            face = frame_rgb[y1:y2, x1:x2]
            
            # Skip if face is too small
            if face.size == 0 or min(face.shape[:2]) < 40:
                continue
                
            try:
                # Preprocess face
                face_tensor = preprocess_face(face)
                
                # Get embedding
                with torch.no_grad():
                    query_embedding = resnet(face_tensor)
                    query_embedding = F.normalize(query_embedding, p=2, dim=1)
                    
                    # Compare with database
                    best_match = None
                    best_score = threshold
                    
                    for name, db_embedding in embeddings_db.items():
                        score = F.cosine_similarity(
                            query_embedding, 
                            db_embedding.unsqueeze(0)
                        ).item()
                        
                        if score > best_score:
                            best_score = score
                            best_match = name
                    
                    # Draw results
                    label = best_match if best_match is not None else "Unknown"
                    color = (0, 255, 0) if best_match is not None else (0, 0, 255)
                    
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(frame, f"{label} ({best_score:.2f})", 
                              (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 
                              0.9, color, 2)
            
            except Exception as e:
                print(f"Error processing face: {e}")
                continue
    
    return frame

# Main loop
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    recognized_frame = recognize_faces(frame)
    cv2.imshow('Face Recognition', recognized_frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
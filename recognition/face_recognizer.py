import cv2
import torch
import torch.nn.functional as F
import pickle
import numpy as np
from facenet_pytorch import MTCNN, InceptionResnetV1
import os

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"[FaceRecognition] Using device: {device}")

# Initialize models
mtcnn = MTCNN(
    image_size=160,
    margin=20,
    keep_all=True,
    min_face_size=40,
    device=device,
    post_process=False
)
resnet = InceptionResnetV1(pretrained='vggface2').eval().to(device)

def preprocess_face(face_img, target_size=160):
    face_img = cv2.resize(face_img, (target_size, target_size))
    face_tensor = torch.from_numpy(face_img).permute(2, 0, 1).float()
    face_tensor = (face_tensor - 127.5) / 128.0
    return face_tensor.unsqueeze(0).to(device)

def recognize_faces(frame, threshold=0.7):
    
    # Load latest embeddings every time the function runs
    try:
        with open('embeddings.pkl', 'rb') as f:
            raw_embeddings = pickle.load(f)
            embeddings_db = {name: emb.to(device) for name, emb in raw_embeddings.items()}
    except Exception as e:
        print(f"[FaceRecognition] Failed to load embeddings: {e}")
        return frame
    
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    boxes, probs, landmarks = mtcnn.detect(frame_rgb, landmarks=True)

    if boxes is not None:
        for i, box in enumerate(boxes):
            x1, y1, x2, y2 = [int(coord) for coord in box]
            if x2 <= x1 or y2 <= y1:
                continue

            pad = 20
            h, w = frame.shape[:2]
            y1 = max(0, y1 - pad)
            y2 = min(h, y2 + pad)
            x1 = max(0, x1 - pad)
            x2 = min(w, x2 + pad)

            face = frame_rgb[y1:y2, x1:x2]

            if face.size == 0 or min(face.shape[:2]) < 40:
                continue

            try:
                face_tensor = preprocess_face(face)
                with torch.no_grad():
                    query_embedding = resnet(face_tensor)
                    query_embedding = F.normalize(query_embedding, p=2, dim=1)

                    best_match = None
                    best_score = threshold

                    for name, db_embedding in embeddings_db.items():
                        score = F.cosine_similarity(query_embedding, db_embedding.unsqueeze(0)).item()
                        if score > best_score:
                            best_score = score
                            best_match = name

                    label = best_match if best_match else "Unknown"
                    color = (0, 255, 0) if best_match else (0, 0, 255)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(frame, f"{label} ({best_score:.2f})", (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
            except Exception as e:
                print(f"[FaceRecognition] Error processing face: {e}")
                continue
    return frame

def generate_embeddings(dataset_path='./static/dataset', output_path='embeddings.pkl'):
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Initialize models
    resnet = InceptionResnetV1(pretrained='vggface2').eval().to(device)
    mtcnn = MTCNN(
        image_size=160,
        margin=0,
        keep_all=False,
        min_face_size=20,
        device=device
    )

    def get_embedding(face_img):
        # Convert BGR to RGB
        face_img_rgb = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
        
        # Preprocess with MTCNN and get embedding
        face_tensor = mtcnn(face_img_rgb)
        
        if face_tensor is not None:
            face_tensor = face_tensor.to(device)
            embedding = resnet(face_tensor.unsqueeze(0))
            return F.normalize(embedding, p=2, dim=1)[0].detach().cpu()
        
        return None

    embeddings_db = {}

    for person_name in os.listdir(dataset_path):
        person_path = os.path.join(dataset_path, person_name)
        if not os.path.isdir(person_path):
            continue

        embeddings = []
        processed_count = 0

        for img_file in sorted(os.listdir(person_path))[:30]:  # Only take first 30 images
            if img_file.lower().endswith(('.jpg', '.jpeg', '.png')):
                img_path = os.path.join(person_path, img_file)
                img = cv2.imread(img_path)
                
                if img is not None:
                    embedding = get_embedding(img)
                    if embedding is not None:
                        embeddings.append(embedding)
                        processed_count += 1

        if embeddings:
            avg_embedding = torch.stack(embeddings).mean(dim=0)
            embeddings_db[person_name] = avg_embedding
            print(f"✅ Processed {processed_count} images for {person_name}")
        else:
            print(f"⚠️ No valid faces found for {person_name}")

    with open(output_path, 'wb') as f:
        pickle.dump(embeddings_db, f)

    print(f"\n🎉 Embeddings generation completed! Saved {len(embeddings_db)} identities to '{output_path}'")

    
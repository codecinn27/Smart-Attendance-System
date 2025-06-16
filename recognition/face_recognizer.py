import cv2
import torch
import torch.nn.functional as F
import pickle
import numpy as np
from facenet_pytorch import MTCNN, InceptionResnetV1
import os
from datetime import datetime
from database.helper_function import save_attendance_to_db, get_student_id_by_name,  has_attendance_today
from pathlib import Path

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"[FaceRecognition] Using device: {device}")
attendance_displayed_today = set()  # Set of tuples like (student_id, class_id, date)

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

# Load face embedding global, load only when changed
embedding_file = Path("embeddings.pkl")
last_loaded_time = 0
embeddings_db = {}

def preprocess_face(face_img, target_size=160):
    face_img = cv2.resize(face_img, (target_size, target_size))
    face_tensor = torch.from_numpy(face_img).permute(2, 0, 1).float()
    face_tensor = (face_tensor - 127.5) / 128.0
    return face_tensor.unsqueeze(0).to(device)

def recognize_faces(frame, threshold=0.7, class_id=None, class_name=None):
    global embeddings_db, last_loaded_time

    # Check if embeddings.pkl has changed, reload the embedding only when changed
    current_mtime = embedding_file.stat().st_mtime
    if current_mtime > last_loaded_time:
        try:
            with open(embedding_file, 'rb') as f:
                raw_embeddings = pickle.load(f)
                embeddings_db = {name: emb.to(device) for name, emb in raw_embeddings.items()}
                last_loaded_time = current_mtime
                print("[FaceRecognition] Reloaded embeddings from updated file.")
        except Exception as e:
            print(f"[FaceRecognition] Failed to load embeddings: {e}")
            return frame, None
        
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    boxes, probs, landmarks = mtcnn.detect(frame_rgb, landmarks=True)
    attendance_message = None

    if boxes is not None:
        for i, box in enumerate(boxes):
            x1, y1, x2, y2 = map(int, box)
            if x2 <= x1 or y2 <= y1:
                continue

            pad = 20
            h, w = frame.shape[:2]
            y1, y2 = max(0, y1 - pad), min(h, y2 + pad)
            x1, x2 = max(0, x1 - pad), min(w, x2 + pad)

            face = frame_rgb[y1:y2, x1:x2]
            if face.size == 0 or min(face.shape[:2]) < 40:
                continue

            try:
                face_tensor = preprocess_face(face)
                with torch.no_grad():
                    query_embedding = F.normalize(resnet(face_tensor), p=2, dim=1)
                    best_score = threshold
                    best_match = None

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

                    if class_id and best_match:
                        student_id = get_student_id_by_name(best_match)
                        today_str = datetime.now().strftime("%Y-%m-%d")
                        if student_id and not has_attendance_today(student_id, class_id, today_str):
                            saved = save_attendance_to_db(student_id, class_id)
                            if saved:
                                attendance_message = {
                                    "student_name": best_match,
                                    "class_name": class_name,
                                    "timestamp": datetime.now().strftime("%H:%M:%S")
                                }
            except Exception as e:
                print(f"[FaceRecognition] Recognition error: {e}")
                continue

    return frame, attendance_message


def generate_embeddings_async(session_id, training_sessions):
    import io, sys
    import torch.nn.functional as F
    from facenet_pytorch import MTCNN, InceptionResnetV1
    import torch, os, cv2, pickle, shutil

    log_stream = io.StringIO()
    sys_stdout = sys.stdout
    sys.stdout = log_stream

    # Device and models
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Train-{session_id}] Using device: {device}")
    resnet = InceptionResnetV1(pretrained='vggface2').eval().to(device)
    mtcnn = MTCNN(image_size=160, keep_all=False, device=device)

    def get_embedding(face_img):
        face_img_rgb = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
        face_tensor = mtcnn(face_img_rgb)
        if face_tensor is not None:
            if len(face_tensor.shape) == 3:
                face_tensor = face_tensor.unsqueeze(0)  # add batch dimension
            face_tensor = face_tensor.to(device)
            embedding = resnet(face_tensor)
            return F.normalize(embedding, p=2, dim=1)[0].detach().cpu()
        return None

    training_sessions[session_id] = {"status": "in_progress", "log": "", "processed_info": []}

    try:
        embeddings_db = {}
        dataset_path = './static/dataset'
        output_path = './static/trained'
        os.makedirs(output_path, exist_ok=True)

        for person_name in os.listdir(dataset_path):
            person_path = os.path.join(dataset_path, person_name)
            if not os.path.isdir(person_path):
                continue

            embeddings = []
            image_paths = []
            person_output = os.path.join(output_path, person_name)
            os.makedirs(person_output, exist_ok=True)

            for img_file in os.listdir(person_path):
                if img_file.lower().endswith(('.jpg', '.jpeg', '.png')):
                    img_path = os.path.join(person_path, img_file)
                    img = cv2.imread(img_path)

                    if img is not None:
                        embedding = get_embedding(img)
                        if embedding is not None:
                            embeddings.append(embedding)
                            shutil.copy(img_path, os.path.join(person_output, img_file))
                            image_paths.append(f"/static/trained/{person_name}/{img_file}")

            if embeddings:
                avg_embedding = torch.stack(embeddings).mean(dim=0)
                embeddings_db[person_name] = avg_embedding
                print(f"✅ Processed {len(image_paths)} images for {person_name}")
                training_sessions[session_id]["processed_info"].append({
                    "name": person_name,
                    "images": image_paths
                })
            else:
                print(f"⚠️ No valid faces found for {person_name}")

        with open("embeddings.pkl", "wb") as f:
            pickle.dump(embeddings_db, f)

        print(f"\n🎉 Embeddings generation completed! Saved {len(embeddings_db)} identities.")
        training_sessions[session_id]["status"] = "done"

    except Exception as e:
        training_sessions[session_id]["status"] = "error"
        print(f"❌ Error during training: {e}")
    finally:
        sys.stdout = sys_stdout
        training_sessions[session_id]["log"] = log_stream.getvalue()

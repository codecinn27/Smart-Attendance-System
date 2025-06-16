import os
from pathlib import Path
from PIL import Image
import numpy as np
import torch
from facenet_pytorch import MTCNN, InceptionResnetV1
from sklearn.metrics import classification_report, accuracy_score
import pickle
import json

# Load MTCNN and Inception Resnet
mtcnn = MTCNN(image_size=160, margin=20)
resnet = InceptionResnetV1(pretrained='vggface2').eval()

# Load trained embeddings and labels
with open('embeddings.pkl', 'rb') as f:
    stored_embeddings, stored_labels = pickle.load(f)

print(f"Type of first embedding before conversion: {type(stored_embeddings[0])}")

clean_embeddings = []
for e in stored_embeddings:
    try:
        if isinstance(e, str):
            arr = np.array(json.loads(e))
        else:
            arr = np.array(e)
        if arr.size == 512:
            clean_embeddings.append(arr)
        else:
            print("⚠️ Invalid embedding size:", arr.shape)
    except Exception as err:
        print(f"⚠️ Failed to decode embedding: {str(err)}")

if not clean_embeddings:
    print("❌ No valid embeddings found. Exiting...")
    exit()

stored_embeddings = torch.tensor(clean_embeddings, dtype=torch.float32)

# Decode labels if needed
if isinstance(stored_labels[0], bytes):
    stored_labels = [label.decode() for label in stored_labels]

# Prepare test data
test_dir = Path("./static/test")
y_true = []
y_pred = []
skipped = 0

for person_dir in test_dir.iterdir():
    if person_dir.is_dir():
        true_label = person_dir.name
        for img_path in person_dir.glob("*.jpg"):
            try:
                img = Image.open(img_path).convert("RGB")
                face = mtcnn(img)
                if face is None:
                    print(f"⚠️ No face detected in {img_path.name}")
                    skipped += 1
                    continue

                face_embedding = resnet(face.unsqueeze(0)).detach()  # (1, 512)

                if stored_embeddings.size(0) == 0:
                    print("❌ No valid stored embeddings to compare.")
                    skipped += 1
                    continue

                sims = torch.nn.functional.cosine_similarity(face_embedding, stored_embeddings)
                best_match_idx = torch.argmax(sims).item()
                predicted_label = stored_labels[best_match_idx]

                y_true.append(true_label)
                y_pred.append(predicted_label)

                print(f"[✓] {img_path.name} → Predicted: {predicted_label} | Actual: {true_label}")

            except Exception as e:
                print(f"❌ Error with {img_path.name}: {e}")
                skipped += 1

# Evaluation
print("\n=== Evaluation Report ===")
print(f"Total Images Tested: {len(y_true)}")
print(f"Total Skipped (no face/errors): {skipped}")

if y_true and y_pred:
    print("Accuracy:", accuracy_score(y_true, y_pred))
    print("\nDetailed Classification Report:")
    print(classification_report(y_true, y_pred))
else:
    print("❌ No successful predictions to evaluate.")

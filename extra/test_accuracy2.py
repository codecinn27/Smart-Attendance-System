import os
import numpy as np
from PIL import Image
from tqdm import tqdm
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics.pairwise import cosine_similarity
from facenet_pytorch import InceptionResnetV1
import torch

# Setup
DATASET_PATH = "./static/dataset"
TRAIN_COUNT = 25
TEST_COUNT = 5
THRESHOLD = 0.7  # Cosine similarity threshold

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = InceptionResnetV1(pretrained='vggface2').eval().to(device)

def get_face_embedding(image_path):
    img = Image.open(image_path).convert("RGB").resize((160, 160))
    img_tensor = torch.tensor(np.array(img)).permute(2, 0, 1).float() / 255.0
    img_tensor = img_tensor.unsqueeze(0).to(device)
    with torch.no_grad():
        embedding = model(img_tensor)
    return embedding.cpu().numpy()

# Build training database
print("[INFO] Generating embeddings...")
train_db = {}
labels = []
true_labels = []
predicted_labels = []

for person in os.listdir(DATASET_PATH):
    person_path = os.path.join(DATASET_PATH, person)
    if not os.path.isdir(person_path):
        continue

    # Get training images
    all_images = sorted(os.listdir(person_path))
    train_images = all_images[:TRAIN_COUNT]
    test_images = all_images[TRAIN_COUNT:TRAIN_COUNT + TEST_COUNT]

    embeddings = []
    for img_name in train_images:
        path = os.path.join(person_path, img_name)
        emb = get_face_embedding(path)
        embeddings.append(emb)

    # Average embedding per person
    mean_embedding = np.mean(embeddings, axis=0)
    train_db[person] = mean_embedding

    # Test
    for img_name in test_images:
        test_path = os.path.join(person_path, img_name)
        test_emb = get_face_embedding(test_path)

        max_similarity = -1
        best_match = "Unknown"
        for registered_name, ref_emb in train_db.items():
            similarity = cosine_similarity(test_emb, ref_emb)[0][0]
            if similarity > max_similarity:
                max_similarity = similarity
                best_match = registered_name

        # Threshold
        if max_similarity >= THRESHOLD:
            predicted_labels.append(best_match)
        else:
            predicted_labels.append("Unknown")

        true_labels.append(person)
        labels.append(person)

# Evaluation
print("\n=== CLASSIFICATION REPORT ===")
print(classification_report(true_labels, predicted_labels, zero_division=0))

print("\n=== CONFUSION MATRIX ===")
print(confusion_matrix(true_labels, predicted_labels))

correct = sum([1 for t, p in zip(true_labels, predicted_labels) if t == p])
accuracy = (correct / len(true_labels)) * 100
print(f"\n✅ Accuracy: {accuracy:.2f}% on {len(true_labels)} test images")

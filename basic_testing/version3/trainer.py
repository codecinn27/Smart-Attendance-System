import os
import cv2
import torch
import torch.nn.functional as F
from facenet_pytorch import InceptionResnetV1, MTCNN
import pickle

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
        # Move tensor to the same device as model
        face_tensor = face_tensor.to(device)
        
        # Get embedding and normalize
        embedding = resnet(face_tensor.unsqueeze(0))
        return F.normalize(embedding, p=2, dim=1)[0].detach().cpu()  # Return CPU tensor
    
    return None

# Process all images in the dataset folder
dataset_path = 'dataset'
embeddings_db = {}

for person_name in os.listdir(dataset_path):
    person_path = os.path.join(dataset_path, person_name)
    if not os.path.isdir(person_path):
        continue
    
    embeddings = []
    processed_count = 0
    
    for img_file in os.listdir(person_path):
        if img_file.lower().endswith(('.jpg', '.jpeg', '.png')):
            img_path = os.path.join(person_path, img_file)
            img = cv2.imread(img_path)
            
            if img is not None:
                embedding = get_embedding(img)
                if embedding is not None:
                    embeddings.append(embedding)
                    processed_count += 1
    
    if embeddings:
        # Average all embeddings for this person
        avg_embedding = torch.stack(embeddings).mean(dim=0)
        embeddings_db[person_name] = avg_embedding
        print(f"Processed {processed_count} images for {person_name}")
    else:
        print(f"No valid faces found for {person_name}")

# Save embeddings to file
with open('embeddings.pkl', 'wb') as f:
    pickle.dump(embeddings_db, f)

print(f"Embeddings generation completed! Saved {len(embeddings_db)} identities.")
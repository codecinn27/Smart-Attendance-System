import os
import cv2
import torch
import torch.nn.functional as F
from facenet_pytorch import InceptionResnetV1, MTCNN
from types import MethodType

# Set device (GPU if available)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

### helper function
def encode(img_tensor):
    # img_tensor should be 3D or 4D tensor, already preprocessed by MTCNN (normalized)
    if img_tensor.ndim == 3:
        img_tensor = img_tensor.unsqueeze(0)  # add batch dim
    img_tensor = img_tensor.to(device)
    with torch.no_grad():
        embedding = resnet(img_tensor)
        embedding = F.normalize(embedding, p=2, dim=1)  # Normalize embeddings
    return embedding

def detect_box(self, img, save_path=None):
    # Detect faces
    batch_boxes, batch_probs, batch_points = self.detect(img, landmarks=True)
    # Select faces
    if not self.keep_all:
        batch_boxes, batch_probs, batch_points = self.select_boxes(
            batch_boxes, batch_probs, batch_points, img, method=self.selection_method
        )
    # Extract faces
    faces = self.extract(img, batch_boxes, save_path)
    return batch_boxes, faces


### load model
resnet = InceptionResnetV1(pretrained='vggface2').eval().to(device)
mtcnn = MTCNN(
    image_size=224, keep_all=True, thresholds=[0.4, 0.5, 0.5], min_face_size=60,
    device=device
)
mtcnn.detect_box = MethodType(detect_box, mtcnn)

### get encoded features for all saved images
saved_pictures = "./saved2/"
all_people_faces = {}

for person_name in os.listdir(saved_pictures):
    person_path = os.path.join(saved_pictures, person_name)
    if not os.path.isdir(person_path):
        continue  # skip files, we only want folders

    embeddings = []

    for file in os.listdir(person_path):
        if file.lower().endswith((".jpg", ".png")):
            img_path = os.path.join(person_path, file)
            img = cv2.imread(img_path)

            if img is None:
                continue

            # Convert BGR (cv2) to RGB
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            cropped = mtcnn(img_rgb)  # cropped is [3, 224, 224] tensor if face detected, else None

            if cropped is not None:
                emb = encode(cropped)[0]  # correct: add batch dim here only once
                embeddings.append(emb)


    if embeddings:
        # Average embedding for each person, then normalize again
        avg_emb = torch.stack(embeddings).mean(dim=0)
        avg_emb = F.normalize(avg_emb.unsqueeze(0), p=2, dim=1)[0]
        all_people_faces[person_name] = avg_emb
        print(f"[INFO] Loaded {len(embeddings)} images for {person_name}")
        
def detect(cam=0, thres=0.7):
    vdo = cv2.VideoCapture(cam)
    while vdo.isOpened():
            ret, img0 = vdo.read()
            if not ret:
                break
            batch_boxes, cropped_images = mtcnn.detect_box(img0)

            if cropped_images is not None:
                for box, cropped in zip(batch_boxes, cropped_images):
                    x, y, x2, y2 = [int(x) for x in box]
                    img_embedding = encode(cropped)[0]
                    detect_dict = {}
                    for name, emb in all_people_faces.items():
                        cosine_sim = F.cosine_similarity(img_embedding.unsqueeze(0), emb.unsqueeze(0)).item()
                        detect_dict[name] = cosine_sim

                    if detect_dict:
                        # Find person with highest cosine similarity
                        max_key = max(detect_dict, key=detect_dict.get)
                        max_sim = detect_dict[max_key]

                        if max_sim < thres:
                            max_key = "Undetected"
                    else:
                        max_key = "Undetected"

                    # Draw box and label
                    cv2.rectangle(img0, (x, y), (x2, y2), (0, 0, 255), 2)
                    cv2.putText(
                        img0, max_key, (x + 5, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2
                    )

            cv2.imshow("Face Recognition", img0)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    vdo.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    detect(0)
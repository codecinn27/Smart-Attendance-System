import os
import cv2
import torch
from facenet_pytorch import InceptionResnetV1, MTCNN
from types import MethodType

### helper function
def encode(img_tensor):
    # img_tensor should be 4D: [batch, 3, 224, 224]
    with torch.no_grad():
        res = resnet(img_tensor)
    return res

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
resnet = InceptionResnetV1(pretrained='vggface2').eval()
mtcnn = MTCNN(
  image_size=224, keep_all=True, thresholds=[0.4, 0.5, 0.5], min_face_size=60
)
mtcnn.detect_box = MethodType(detect_box, mtcnn)

### get encoded features for all saved images
saved_pictures = "./saved/"
all_people_faces = {}

for person_name in os.listdir(saved_pictures):
    person_path = os.path.join(saved_pictures, person_name)
    if not os.path.isdir(person_path):
        continue  # skip files, we only want folders

    embeddings = []

    for file in os.listdir(person_path):
        if file.endswith(".jpg") or file.endswith(".png"):
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
        # Average embedding for each person
        all_people_faces[person_name] = torch.stack(embeddings).mean(dim=0)
        print(f"[INFO] Loaded {len(embeddings)} images for {person_name}")
        
def detect(cam=0, thres=0.7):
    vdo = cv2.VideoCapture(cam)
    while vdo.grab():
        _, img0 = vdo.retrieve()
        batch_boxes, cropped_images = mtcnn.detect_box(img0)

        if cropped_images is not None:
            for box, cropped in zip(batch_boxes, cropped_images):
                x, y, x2, y2 = [int(x) for x in box]
                img_embedding = encode(cropped.unsqueeze(0))[0]
                detect_dict = {}
                for k, v in all_people_faces.items():
                    detect_dict[k] = (v - img_embedding).norm().item()
                min_key = min(detect_dict, key=detect_dict.get)

                if detect_dict[min_key] >= thres:
                    min_key = 'Undetected'
                
                cv2.rectangle(img0, (x, y), (x2, y2), (0, 0, 255), 2)
                cv2.putText(
                  img0, min_key, (x + 5, y + 10), 
                   cv2.FONT_HERSHEY_DUPLEX, 0.5, (255, 255, 255), 1)
                
        ### display
        cv2.imshow("output", img0)
        if cv2.waitKey(1) == ord('q'):
            cv2.destroyAllWindows()
            break

if __name__ == "__main__":
    detect(0)
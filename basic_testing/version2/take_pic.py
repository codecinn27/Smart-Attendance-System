import cv2
import os
from facenet_pytorch import MTCNN
from torchvision.transforms.functional import to_pil_image

# Step 1: Get user input
name = input("Enter your name: ").strip()
save_dir = f"./saved2/{name}"

# Step 2: Create save directory if it doesn't exist
os.makedirs(save_dir, exist_ok=True)

# Step 3: Initialize webcam and MTCNN
cap = cv2.VideoCapture(0)
mtcnn = MTCNN(image_size=224, margin=20, keep_all=False, post_process=True)

count = 0
scale = 1.3

while count < 20:
    ret, frame = cap.read()
    if not ret:
        print("[ERROR] Failed to read from webcam.")
        break

    # Step 1: Scale the frame
    h, w = frame.shape[:2]
    scaled_frame = cv2.resize(frame, (int(w * scale), int(h * scale)))

    # Step 2: Convert to RGB for MTCNN
    scaled_rgb = cv2.cvtColor(scaled_frame, cv2.COLOR_BGR2RGB)

    # Step 3: Detect face and crop
    face = mtcnn(scaled_rgb)

    # Step 4: Save if face detected
    if face is not None:
        try:
            img = to_pil_image(face)  # Already RGB
            img.save(f"{save_dir}/{name}_{count+1}.jpg")
            count += 1
            print(f"[INFO] Saved face {count}/20")
        except Exception as e:
            print(f"[WARNING] Could not save image: {e}")

    # Step 5: Show live preview (original frame)
    preview = frame.copy()
    cv2.putText(preview, f"Captured: {count}/20", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow("Face Capture", preview)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        print("[INFO] Exiting early.")
        break

cap.release()
cv2.destroyAllWindows()
print("[DONE] Face capture complete.")
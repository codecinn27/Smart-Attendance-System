import cv2
import os
from pathlib import Path

def capture_images_for_evaluation(name, save_dir="./static/test", max_images=50):
    # Create save directory
    person_dir = Path(save_dir) / name
    person_dir.mkdir(parents=True, exist_ok=True)

    # Open webcam
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ Cannot open webcam.")
        return

    print(f"📸 Capturing images for: {name}")
    count = 0

    while count < max_images:
        ret, frame = cap.read()
        if not ret:
            print("❌ Failed to grab frame.")
            break

        cv2.imshow("Press SPACE to capture, ESC to exit", frame)

        key = cv2.waitKey(1)
        if key % 256 == 27:
            # ESC pressed
            print("❌ Capture cancelled.")
            break
        elif key % 256 == 32:
            # SPACE pressed
            img_path = person_dir / f"image_{count+1}.jpg"
            cv2.imwrite(str(img_path), frame)
            print(f"✅ Saved {img_path.name}")
            count += 1

    cap.release()
    cv2.destroyAllWindows()
    print("✅ Done capturing.")

if __name__ == "__main__":
    person_name = input("Enter name for test dataset: ").strip()
    if person_name:
        capture_images_for_evaluation(person_name)
    else:
        print("❌ Name cannot be empty.")

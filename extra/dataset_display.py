import os
import matplotlib.pyplot as plt
import cv2

def display_all_images_grid(dataset_path='./static/dataset'):
    image_paths = []

    # Collect image paths
    for student_folder in os.listdir(dataset_path):
        student_path = os.path.join(dataset_path, student_folder)
        if not os.path.isdir(student_path):
            continue

        for img_file in os.listdir(student_path):
            if img_file.lower().endswith(('.jpg', '.jpeg', '.png')):
                full_path = os.path.join(student_path, img_file)
                image_paths.append((full_path, student_folder))  # Save with label

    if not image_paths:
        print("⚠️ No images found in dataset.")
        return

    # Plot images in grid
    num_images = len(image_paths)
    cols = 6
    rows = (num_images + cols - 1) // cols

    plt.figure(figsize=(15, rows * 2.5))

    for i, (img_path, label) in enumerate(image_paths):
        img = cv2.imread(img_path)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        plt.subplot(rows, cols, i + 1)
        plt.imshow(img_rgb)
        plt.title(label, fontsize=8)
        plt.axis('off')

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    display_all_images_grid()

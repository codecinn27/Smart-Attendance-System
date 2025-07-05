# 🧠 Smart Attendance System using Face Recognition

A lightweight and scalable **Smart Attendance System** that uses **facial recognition** to automate and simplify attendance tracking. Built with `FastAPI` for the backend, `FaceNet-PyTorch` for facial recognition, and `SQLite` for local data storage. Designed to be accurate, fast, and practical for real-world applications in classrooms, offices, and events.

---

## 🚀 Features

- 👤 **Face Detection** using **MTCNN**
- 🧠 **Face Recognition** using **FaceNet (InceptionResnet pretrained on VGGFace2)**
- 📝 **Attendance Logging** stored in a local **SQLite database**
- ⚡ Fast backend powered by **FastAPI**
- 🎨 Clean UI styled with **Semantic UI** + **TailwindCSS**
- 🔄 Quick training: only ~30 images needed per person to register a new face

---

## 🧩 Tech Stack

| Layer       | Technology                     |
|-------------|--------------------------------|
| **Backend** | FastAPI                        |
| **Detection** | MTCNN                        |
| **Recognition** | FaceNet (InceptionResnet, VGGFace2) |
| **Database** | SQLite                        |
| **Frontend CSS** | Semantic UI + TailwindCSS |

---

## 📸 How It Works

1. User uploads ~30 images of a new person.
2. System uses **MTCNN** to detect the face and crop it.
3. Features are extracted using **FaceNet (InceptionResnet)**.
4. Embeddings are stored and labeled in the database.
5. During attendance:
   - The system detects and recognizes the face.
   - The timestamp is recorded in SQLite as the attendance log.

---

## 🖥️ Demo
[Youtube video demo](https://youtu.be/E7SeZHRXmcY?si=ai1RQx5f7wagXHoG)

1) Main page : http://localhost:8000
![image_2025-06-26_14-15-21](https://github.com/user-attachments/assets/3a0426d6-4199-4f62-b6b8-3b47f5e8c029)

  
2) Click to navbar enroll to access to this and input the user details 
   ![image](https://github.com/user-attachments/assets/0d0fc7fc-1a50-4ce5-a413-d3c479647242)

3) Click the submit enrollment button to start capturing 30 images
![image](https://github.com/user-attachments/assets/ce9510a4-74a8-461c-920f-f135b065b2f3)

4) Click the start training button to store the face embeddings from the image into embeddings.pkl
![image](https://github.com/user-attachments/assets/beb1ad92-c052-4d84-bb53-cefb0f5184b7)

5) Go to recognize page pressed from the navbar, select the class enroll to start getting attendance
![image](https://github.com/user-attachments/assets/1a2c765d-2a91-42a5-8903-0c8f052af274)
![image](https://github.com/user-attachments/assets/586b895b-e7b2-4721-ac35-90f53436574e)

6) Track the attendance record at records page by pressing at the navbar
![image](https://github.com/user-attachments/assets/545f7b9f-84e3-4dc1-a6e3-15f1164b2afa)

---

## 📂 Project Structure
```graphql
smart-attendance/
├── app.py                         # Main FastAPI application
├── recognition/
│   └── face_recognizer.py         # Face recognition logic (FaceNet, embedding, matching)
├── database/
│   └── reset_db.py                # Script to reset or initialize the database
├── static/
│   ├── dataset/                   # Stored face image samples
│   └── sound/                     # Audio feedback (optional)
├── templates/                     # HTML templates for UI
│   ├── enroll.html                # Page for face enrollment
│   ├── home.html                  # Landing page
│   ├── recognize.html             # Face recognition and check-in page
│   └── record.html                # Attendance record page
├── attendance.db                  # SQLite database to store attendance records
├── embeddings.pkl                 # Pickled face embeddings trained from images
├── requirements.txt               # Python dependencies
└── README.md                      # Project documentation
```



---

## 🛠️ Installation

### 1. Clone the Repository
```bash
git clone https://github.com/yourusername/smart-attendance-system.git
cd smart-attendance-system
```

#### 2. Create Virtual Environment & Install Dependencies
```bash
py -3.10 -m venv env
.\env\Scripts\Activate  # or venv\Scripts\activate on Windows
pip install torch==2.2.2+cu118 torchvision==0.17.2+cu118 torchaudio==2.2.2 --index-url https://download.pytorch.org/whl/cu118 #run this manually to install facenet which support GPU acceleration
pip install -r requirements.txt
```

#### 3. Run the Server
```bash
uvicorn app.main:app --reload
```

#### 4. Access the Web App
Open your browser and go to http://localhost:8000


from fastapi import FastAPI, Request, WebSocket, Form, WebSocketDisconnect, BackgroundTasks
from fastapi.responses import HTMLResponse, RedirectResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import base64
import cv2
import os
import asyncio
from pathlib import Path
from io import BytesIO
from PIL import Image
import numpy as np
from recognition.face_recognizer import recognize_faces, mtcnn, generate_embeddings_async #for face recognition
import sqlite3
from urllib.parse import quote
from typing import Dict
import uuid
from database.helper_function import get_class_id_by_value
from datetime import datetime
import json
import csv
import io
from fastapi.middleware.cors import CORSMiddleware
from database.reset_database import clearDatabase

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Set up Jinja2 templates
templates = Jinja2Templates(directory="templates")


# Create folders if they don't exist
os.makedirs("static", exist_ok=True)
os.makedirs("templates", exist_ok=True)

# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")


@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("home.html", {"request": request, "title": "Home"})

@app.get("/enroll", response_class=HTMLResponse)
async def get_enroll(request: Request):
    return templates.TemplateResponse("enroll.html", {"request": request, "title": "Enroll Student"})


@app.post("/enroll")
async def post_enroll(
    request: Request,
    name: str = Form(...),
    age: int = Form(...),
    enrolled_classes: list[str] = Form(None)  # for multiple checkboxes with same name
):
    
    conn = sqlite3.connect("attendance.db")
    cursor = conn.cursor()

    # 1. Insert student into students table
    cursor.execute("INSERT INTO students (name, age) VALUES (?, ?)", (name, age))
    student_id = cursor.lastrowid

    # 2. For each class, ensure it exists in classes table (insert if not exist)
    if enrolled_classes is None:
        enrolled_classes = []

    for class_name in enrolled_classes:
        cursor.execute("SELECT class_id FROM classes WHERE class_name = ?", (class_name,))
        class_row = cursor.fetchone()
        if class_row:
            class_id = class_row[0]
        else:
            # Insert class without teacher for now (you can update teacher info elsewhere)
            cursor.execute("INSERT INTO classes (class_name, teacher) VALUES (?, ?)", (class_name, ""))
            class_id = cursor.lastrowid

        # 3. Insert into enrollments table
        cursor.execute("INSERT INTO enrollments (student_id, class_id) VALUES (?, ?)", (student_id, class_id))

    conn.commit()
    conn.close()

    # Trigger redirect with query param for JS to pick up name and start webcam
    return RedirectResponse(url=f"/enroll?name={quote(name)}", status_code=303)
@app.websocket("/ws/enroll_capture/{name}")
async def enroll_capture_ws(websocket: WebSocket, name: str):
    await websocket.accept()
    save_dir = Path(f"static/dataset/{name}")
    save_dir.mkdir(parents=True, exist_ok=True)

    captured_count = 0
    MAX_IMAGES = 30

    try:
        while True:
            data_url = await websocket.receive_text()
            header, encoded = data_url.split(",", 1)
            img_bytes = base64.b64decode(encoded)
            img_pil = Image.open(BytesIO(img_bytes)).convert("RGB")
            frame = np.array(img_pil)  # RGB image as numpy array

            # Convert RGB to BGR for OpenCV
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

            # Detect faces with mtcnn
            boxes, _ = mtcnn.detect(frame)

            if boxes is None or len(boxes) == 0:
                await websocket.send_text("no_face")
                continue

            # Save detected face images (optional)
            for box in boxes:
                x1, y1, x2, y2 = [int(b) for b in box]

                # Clamp coordinates within frame boundaries
                h, w = frame_bgr.shape[:2]
                x1 = max(0, min(x1, w - 1))
                x2 = max(0, min(x2, w - 1))
                y1 = max(0, min(y1, h - 1))
                y2 = max(0, min(y2, h - 1))

                # Check box validity
                if x2 <= x1 or y2 <= y1:
                    continue  # skip invalid box

                face_img = frame_bgr[y1:y2, x1:x2]

                if face_img.size == 0:
                    continue  # skip empty face

                face_filename = save_dir / f"image_{captured_count + 1}.jpg"
                cv2.imwrite(str(face_filename), face_img)
                captured_count += 1
                # ✅ Send count update to frontend
                await websocket.send_text(f"captured:{captured_count}")
                
                if captured_count >= MAX_IMAGES:
                    break

            # Draw bounding boxes on frame
            for box in boxes:
                x1, y1, x2, y2 = [int(b) for b in box]
                cv2.rectangle(frame_bgr, (x1, y1), (x2, y2), (0, 0, 255), 2)

            # Encode frame with boxes back to base64
            _, buffer = cv2.imencode(".jpg", frame_bgr)
            jpg_as_text = base64.b64encode(buffer).decode("utf-8")
            await websocket.send_text(f"data:image/jpeg;base64,{jpg_as_text}")

            if captured_count >= MAX_IMAGES:
                await websocket.send_text("done")
                break

    except WebSocketDisconnect:
        print(f"[Enroll WS] Client disconnected: {name}")
    except Exception as e:
        print(f"[Enroll WS] Error: {e}")
    finally:
        await websocket.close()
      
training_sessions: Dict[str, Dict] = {}

@app.post("/train/start")
async def start_training():
    session_id = str(uuid.uuid4())
    training_sessions[session_id] = {"status": "queued", "log": "", "processed_info": []}
    asyncio.create_task(asyncio.to_thread(generate_embeddings_async, session_id, training_sessions))
    return {"session_id": session_id}

@app.get("/train/status/{session_id}")
async def get_training_status(session_id: str):
    session = training_sessions.get(session_id)
    if not session:
        return {"status": "not_found"}
    return session

@app.get("/recognize", response_class=HTMLResponse)
async def recognize(request: Request):
    return templates.TemplateResponse("recognize.html", {"request": request, "title": "Recognize"})

@app.websocket("/ws/recognize")
async def websocket_recognize(websocket: WebSocket):
    await websocket.accept()

    # Get class from query
    class_name = websocket.query_params.get("class")
    print("▶ Class name from frontend:", class_name)
    
    if not class_name:
        print("❌ No class selected")
        await websocket.close()
        return

    class_id = get_class_id_by_value(class_name)
    if not class_id:
        print("❌ Class not found in DB")
        await websocket.close()
        return

    print(f"➡️ Recognizing attendance for class: {class_name} (ID: {class_id})")

    # Start webcam
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    
    if not cap.isOpened():
        print("❌ Failed to open webcam")
        await websocket.close()
        return
    else:
        print("✅ Webcam opened")

    last_saved = None
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("❌ Failed to read from webcam")
                break
            
            # 🟢 Add class name overlay to the frame
            cv2.putText(
                frame, 
                f"Class: {class_name}", 
                (20, 40), 
                cv2.FONT_HERSHEY_SIMPLEX, 
                1.2, 
                (0, 255, 0), 
                3
            )

            frame, message = recognize_faces(frame, class_id=class_id, class_name=class_name)  # ← use class_id for attendance
            _, buffer = cv2.imencode('.jpg', frame)
            jpg_as_text = base64.b64encode(buffer).decode('utf-8')
            
            if message:
                student_key = (message["student_name"], message["class_name"])
                
                if last_saved != student_key:
                    last_saved = student_key  # Save only (name, class) tuple
                    await websocket.send_text(json.dumps({
                        "frame": jpg_as_text,
                        "message": message
                    }))
                else:
                    # Just send the frame without attendance message
                    await websocket.send_text(json.dumps({
                        "frame": jpg_as_text
                    }))
            else:
                # Send the frame only if no message
                await websocket.send_text(json.dumps({
                    "frame": jpg_as_text
                }))

 
            await asyncio.sleep(0.03)
    except WebSocketDisconnect:
        print("[WebSocket] Client disconnected")
    finally:
        cap.release()
        print("[WebSocket] Disconnected")

        
@app.get("/records", response_class=HTMLResponse)
async def records(request: Request):
    conn = sqlite3.connect("attendance.db")
    cursor = conn.cursor()

    # Section 1: Students with their data (no image)
    cursor.execute("""
        SELECT s.student_id, s.name, s.age
        FROM students s
        GROUP BY s.student_id
    """)
    students = cursor.fetchall()

    # Get classes each student is enrolled in
    student_enrollments = {}
    for student_id, _, _ in students:
        cursor.execute("""
            SELECT c.class_name FROM enrollments e
            JOIN classes c ON e.class_id = c.class_id
            WHERE e.student_id = ?
        """, (student_id,))
        classes_for_student = [row[0] for row in cursor.fetchall()]
        student_enrollments[student_id] = classes_for_student

    # Section 2: Classes with enrolled student names
    cursor.execute("""
        SELECT c.class_id, c.class_name, c.teacher
        FROM classes c
    """)
    classes = cursor.fetchall()

    class_enrollments = {}
    for class_id, class_name, teacher in classes:
        cursor.execute("""
            SELECT s.name FROM enrollments e
            JOIN students s ON e.student_id = s.student_id
            WHERE e.class_id = ?
        """, (class_id,))
        enrolled_students = [row[0] for row in cursor.fetchall()]
        class_enrollments[class_id] = enrolled_students

    # Section 3: Attendance records grouped by date and class
    cursor.execute("""
        SELECT a.date, c.class_name, s.name, a.status, a.clock_in_time, a.clock_out_time
        FROM attendance a
        JOIN enrollments e ON a.enrollment_id = e.id
        JOIN students s ON e.student_id = s.student_id
        JOIN classes c ON e.class_id = c.class_id
        ORDER BY a.date, c.class_name, s.name
    """)
    attendance_records = cursor.fetchall()

    conn.close()

    return templates.TemplateResponse("record.html", {
        "request": request,
        "title": "Records",
        "students": students,
        "student_enrollments": student_enrollments,
        "classes": classes,
        "class_enrollments": class_enrollments,
        "attendance_records": attendance_records
    })

@app.get("/download_csv")
async def download_csv():
    conn = sqlite3.connect("attendance.db")
    cursor = conn.cursor()

    cursor.execute("""
        SELECT a.date, c.class_name, s.name, a.status, a.clock_in_time, a.clock_out_time
        FROM attendance a
        JOIN enrollments e ON a.enrollment_id = e.id
        JOIN students s ON e.student_id = s.student_id
        JOIN classes c ON e.class_id = c.class_id
        ORDER BY a.date DESC, c.class_name, s.name
    """)
    rows = cursor.fetchall()
    conn.close()

    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow(["Date", "Class", "Student", "Status", "Clock In", "Clock Out"])
    for row in rows:
        writer.writerow(row)
    output.seek(0)

    return StreamingResponse(
        iter([output.getvalue()]),
        media_type="text/csv",
        headers={"Content-Disposition": "attachment; filename=attendance_records.csv"}
    )
    
@app.post("/clear-data")
def clear_data():
    return clearDatabase()
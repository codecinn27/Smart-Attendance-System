
from fastapi import FastAPI, Request, WebSocket, Form, WebSocketDisconnect
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import base64
import cv2
from typing import List
import os
import asyncio
from recognition.face_recognizer import recognize_faces #for face recognition


app = FastAPI()
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
    enrolled_classes: List[str] = Form(...)
):
    # No database, just redirect after form submission
    print(f"[INFO] Enrolled: {name}, Age: {age}, Classes: {enrolled_classes}")
    return RedirectResponse("/enroll", status_code=303)

@app.get("/recognize", response_class=HTMLResponse)
async def recognize(request: Request):
    return templates.TemplateResponse("recognize.html", {"request": request, "title": "Recognize"})
@app.websocket("/ws/recognize")
async def websocket_recognize(websocket: WebSocket):
    await websocket.accept()

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # ✅ Call your face recognition function
            frame = recognize_faces(frame)

            # ✅ Encode to JPEG and send as base64
            _, buffer = cv2.imencode('.jpg', frame)
            jpg_as_text = base64.b64encode(buffer).decode('utf-8')

            await websocket.send_text(jpg_as_text)
            await asyncio.sleep(0.03)  # Optional: Reduce CPU usage

    except WebSocketDisconnect:
        print("[WebSocket] Client disconnected")
        
    except Exception as e:
        print(f"[WebSocket Error] {e}")
    finally:
        cap.release()
        print("[WebSocket] Disconnected")

        
@app.get("/records", response_class=HTMLResponse)
async def records(request: Request):
    return templates.TemplateResponse("record.html", {"request": request, "title": "Records"})




import cv2
from fastapi import FastAPI, File, UploadFile, Form, HTTPException,WebSocket
from starlette.websockets import WebSocketDisconnect
from ultralytics import YOLO
import uvicorn
import random
from fastapi.middleware.cors import CORSMiddleware
import asyncio
from PIL import Image
import numpy as np
import time
from azure_utils import *
import io
import uuid
from Detection_Model.utils import *
from Detection_Model.database import *
from Detection_Model.face_utils import *

app = FastAPI()
CONFIDENCE_THRESHOLD = 0.57

# Allow CORS for React to communicate with FastAPI
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Replace * with your React app's domain in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load YOLO models
models = {
    "face": YOLO("./Models/Face.pt"),
    "mask": YOLO("./Models/Face Mask.pt"),
    "helmet": YOLO("./Models/HardHat.pt"),
    "hairnet": YOLO("./Models/Hairnet.pt"),
}

# Confidence thresholds for each model
thresholds = {
    "face": 0.5,
    "mask": 0.5,
    "helmet": 0.5,
    "hairnet": 0.5  # Can be adjusted as needed
}
global employee_id

frame_counter = 0
active_websockets = {}
last_upload_time = {}  # Dictionary to store last upload time for each model

# Open RTSP Camera
# rtsp_url = "rtsp://admin:Meridian@2024@14.97.235.83:554/Streaming/Channels/101"
# rtsp_url = "rtsp://170.93.143.139/rtplive/470011e600ef003a004ee33696235daa"
rtsp_url = 0
cap = cv2.VideoCapture(rtsp_url)

# Set buffer size to reduce latency
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

# Dictionary to store assigned colors for each class
COLORS = {}

# Function to generate a random color
def get_random_color():
    return tuple(random.randint(50, 255) for _ in range(3))  # Generate a bright color

# Function to check if one bounding box is inside another
def is_inside(box1, box2):
    x1_1, y1_1, x2_1, y2_1 = box1
    x1_2, y1_2, x2_2, y2_2 = box2
    return x1_1 >= x1_2 and y1_1 >= y1_2 and x2_1 <= x2_2 and y2_1 <= y2_2

KEY_LAYOUT = "right"  # Options: ["right"]

# Function to draw bounding boxes correctly
def draw_bounding_boxes(frame, results,model_name):
    detections = []
    label_positions = {}  # Reset every frame

    # Store all detections
    for result in results:
        boxes = result.boxes.xyxy.cpu().numpy()
        confidences = result.boxes.conf.cpu().numpy()
        classes = result.boxes.cls.cpu().numpy().astype(int)
        names = result.names

        for i, (box, conf, class_id) in enumerate(zip(boxes, confidences, classes)):
            if conf < thresholds[model_name]:
                continue
            if model_name == 'hainet' and class_id not in [1,5]: # filtering classes
                continue

            x1, y1, x2, y2 = map(int, box)
            class_name = names[class_id]

            if class_name not in COLORS:
                COLORS[class_name] = get_random_color()

            detections.append({
                "box": (x1, y1, x2, y2),
                "class": class_name,
                "color": COLORS[class_name]
            })

    # Detect Nested Boxes
    nested_boxes = set()
    parent_boxes = {}

    for i, det1 in enumerate(detections):
        for j, det2 in enumerate(detections):
            if i != j and is_inside(det1["box"], det2["box"]):
                nested_boxes.add(i)
                parent_boxes[i] = j

    # Assign combined keys to parent boxes
    combined_labels = {}

    for i, detection in enumerate(detections):
        parent_index = parent_boxes.get(i, None)
        target_index = parent_index if parent_index is not None else i

        if target_index not in combined_labels:
            combined_labels[target_index] = []
        if (detection["class"], detection["color"]) not in combined_labels[target_index]:
            combined_labels[target_index].append((detection["class"], detection["color"]))

    # Draw Bounding Boxes
    for i, detection in enumerate(detections):
        x1, y1, x2, y2 = detection["box"]
        color = detection["color"]
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2, lineType=cv2.LINE_AA)

    # Draw Keys outside the Parent Box always on the right
    for target_index, labels in combined_labels.items():
        x1, y1, x2, y2 = detections[target_index]["box"]
        rightmost_x2 = x2 + 20  # Fixed position outside the parent box on the right
        start_y = min(y1, y2)  # Start from the top of the parent box
        spacing = 35  # Spacing between each key
        circle_radius = 10
        padding = 10 


        for index, (class_name, color) in enumerate(labels):
        # Calculate text size dynamically
            text_size = cv2.getTextSize(class_name, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
            bg_padding_x = circle_radius * 2 + text_size[0] + (padding * 3)  # Circle + Text + Padding
            bg_padding_y = max(text_size[1], circle_radius * 2) + padding  # Height of text or circle + Padding

            # Center the whole key in the box
            center_x = rightmost_x2 + bg_padding_x // 2
            center_y = start_y + (index * spacing) + bg_padding_y // 2

            # Draw Transparent Black Background
            overlay = frame.copy()
            cv2.rectangle(overlay, (rightmost_x2, start_y + (index * spacing)),
                        (rightmost_x2 + bg_padding_x, start_y + (index * spacing) + bg_padding_y), (0, 0, 0), -1)
            alpha = 0.4  # Transparency Level
            cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

            # Draw Color Circle (Left Side)
            circle_x = rightmost_x2 + circle_radius + padding
            circle_y = center_y
            cv2.circle(frame, (circle_x, circle_y), circle_radius, color, -1, lineType=cv2.LINE_AA)

            # Draw Class Name Text (Right Side)
            text_x = circle_x + circle_radius + padding
            text_y = circle_y + (text_size[1] // 2)
            cv2.putText(frame, class_name, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, lineType=cv2.LINE_AA)
    return frame,detections

# Video processing function
async def process_video(model,model_name):
    global frame_counter
    frame_counter += 1

    # Read frame
    success, frame = cap.read()
    if not success:
        print("[ERROR] Failed to read frame")
        return None

    # Skip processing if not the 5th frame (adjustable)
    if frame_counter % 4 != 0:
        return None

    # Run YOLO detection
    results = model(frame)

    # Draw enhanced bounding boxes
    processed_frame,detections = draw_bounding_boxes(frame, results,model_name)

    # Encode frame as JPEG
    success, buffer = cv2.imencode(".jpg", processed_frame)
    image_bytes = buffer.tobytes()

    # Upload image if there are any detections, throttled to once every 5 seconds per model
    if detections:
        current_time = time.time()
        if (model_name not in last_upload_time) or (current_time - last_upload_time[model_name] >= 3):
            last_upload_time[model_name] = current_time
            await upload_image_to_azure(model_name, image_bytes)

    return image_bytes

# WebSocket Route Generator
def create_websocket_route(model_name):
    @app.websocket(f"/ws/{model_name}")
    async def websocket_endpoint(websocket: WebSocket):
        global active_websockets


        # choosing a previous websocket if it exists 

        if model_name in active_websockets:
            try :
                await active_websockets[model_name].close()
            except:
                pass

        active_websockets[model_name] = websocket    

        await websocket.accept()
        print(f"[INFO] WebSocket Connection Established for {model_name}")

        try:
            while True:
                frame_bytes = await process_video(models[model_name],model_name)
                if frame_bytes:
                    await websocket.send_bytes(frame_bytes)
                await asyncio.sleep(0.03)

        except WebSocketDisconnect:
            print(f"[INFO] WebSocket Disconnected: {model_name}")

        finally:
            if model_name in active_websockets:
                del active_websockets[model_name] # Remove from active connections    

# Dynamically create WebSocket routes
for model_name in models.keys():
    create_websocket_route(model_name)


@app.get("/images/{model_name}")
async def get_images(model_name: str):
    """
    Fetch images for a given model from the corresponding collection.
    Each document in the collection should have an 'image_url' key.
    """
    model_name = model_name.lower()
    # Get the collection for the provided model name.
    if model_name == 'face':
        model_name = 'face_verification'
    else:
        model_name = model_name    
    collection = db[model_name]

    # Fetch all documents, projecting only the "image_url" field and excluding the MongoDB _id.
    images = list(collection.find({}, {"_id": 0, "image_url": 1}))

    if not images:
        raise HTTPException(status_code=404, detail=f"No images found for model: {model_name}")

    return images



# ------------------------ Face detection routes -----------------------------------------

# Custom Exception for Face Detection Issues
class FaceDetectionError(Exception):
    pass

@app.post("/register/")
async def register_user(
    name: str = Form(...),
    employee_id: str = Form(...),
    email: str = Form(...),
    phone: str = Form(...),
    gender: str = Form(...),
    dob: str = Form(...),  # Expect the date of birth (dd-mm-yyyy) string from frontend
    image: UploadFile = File(...),
):
    try:
        """
        Register a new user by extracting embeddings, storing user details in MongoDB,
        saving images to Azure, and adding embeddings to FAISS for efficient retrieval.
        """
        year, month, day = dob.split("-")

        # Read and process image
        image_bytes = await image.read()
        image_pil = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        frame = np.array(image_pil)

        # Detect face and extract embedding
        detections, embeddings = await asyncio.to_thread(face_detector.detect_faces, frame)
        if not detections:
            raise FaceDetectionError("No face detected in the provided image. Please try again with a clearer image.")

        if len(detections) > 1:
            raise FaceDetectionError("Multiple faces detected. Please upload an image with only one person.")

        if not embeddings:
            raise FaceDetectionError("Failed to extract embeddings. Ensure the face is visible and clear.")

        unique_id = str(uuid.uuid4())
        # Convert UUID string to an int format for FAISS (using UUID int conversion)
        uuid_int = int(uuid.UUID(unique_id)) % (2**63)  # Ensures compatibility with FAISS int64 format

        # Upload image to Azure
        image_url = await upload_image_to_azure("Registered_Photos", image_bytes)
        if not image_url:
            return {"success": False, "message": "Image upload failed"}

        # Prepare user data
        user_data = {
            "employee_id":employee_id,
            "unique_id": unique_id,
            "FAISS":uuid_int,
            "name": name,
            "email": email,
            "phone": phone,
            "gender": gender,
            "date": {"day": day, "month": month, "year": year},
            "user_image_url": image_url,
            "embedding": embeddings[0].tolist(),  # Store as JSON-friendly list
        }

        # Check if user already exists
        if await user_exists(email, users_collection):
            return {"success": False, "message": "User with this E-mail already exists"}

        # Save to MongoDB
        await users_collection.insert_one(user_data) 

        # Add embedding to FAISS
        faiss_embedding = np.array(embeddings[0], dtype=np.float32)
        await asyncio.to_thread(add_to_faiss, faiss_embedding, uuid_int)

        return {"success": True, "message": "User registered successfully"}

    except FaceDetectionError as e:
        return {"success": False, "message": str(e)}

    except Exception as e:
        print(f"Unexpected error in registration: {e}")
        return {"success": False, "message": "An unexpected error occurred. Please try again."}
    


@app.websocket("/ws/verify")
async def websocket_verify(websocket: WebSocket):
    global active_websockets

    # Accept the WebSocket connection
    await websocket.accept()
    print("[INFO] WebSocket Connection Established for Face Verification")

    # Add this WebSocket to active connections
    active_websockets["verify"] = websocket

    try:
        while True:
            # Read a frame from the RTSP stream or webcam
            success, frame = cap.read()
            if not success:
                print("[ERROR] Failed to read frame")
                break

            try:
                person_verified = False
                # Perform face detection and verification
                faces = [(x1, y1, x2, y2, embedding) for x1, y1, x2, y2, embedding in detect_faces(frame)]
                if faces:
                    embeddings = np.array([face[4] for face in faces], dtype=np.float32)

                    # Perform FAISS search for matching users
                    D, I = index.search(embeddings, k=5)  # Search top-5 closest matches
                    matched_ids = {int(i) for i in I.flatten() if i != -1}

                    # Fetch user details from MongoDB
                    users_info = {
                        int(user["FAISS"]): user
                        for user in face_collection.find(
                            {"FAISS": {"$in": list(matched_ids)}},
                            {"_id": 0, "FAISS": 1, "name": 1, "unique_id": 1, "gender": 1, "employee_id": 1}
                        )
                    }

                    # Track occupied regions for labels
                    occupied_regions = []

                    # Draw bounding boxes and labels for matched users
                    for (x1, y1, x2, y2, embedding), i_row, d_row in zip(faces, I, D):
                        for i, dist in zip(i_row, d_row):
                            user = users_info.get(int(i))
                            if user and dist <= CONFIDENCE_THRESHOLD:
                                # Draw bounding box
                                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

                                # Prepare the text to display (name, gender, employee ID)
                                person_verified = True
                                employee_id = user['employee_id']
                                details = [
                                    f"Name: {user['name']}",
                                    f"Gender: {user['gender']}",
                                    f"Employee ID: {user['employee_id']}"
                                ]

                                # Calculate maximum text width and total height for the label
                                max_text_width = 0
                                total_text_height = 0
                                for text in details:
                                    text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
                                    max_text_width = max(max_text_width, text_size[0])
                                    total_text_height += text_size[1] + 10  # Add spacing between lines

                                # Add padding to the label dimensions
                                label_width = max_text_width + 20  # Add padding
                                label_height = total_text_height + 10  # Add padding

                                # Initial position for the label (right side of the bounding box)
                                label_x = x2 + 20
                                label_y = y1

                                # Adjust label position to avoid overlapping
                                for region in occupied_regions:
                                    if (label_x < region[0] + region[2] and
                                        label_x + label_width > region[0] and
                                        label_y < region[1] + region[3] and
                                        label_y + label_height > region[1]):
                                        # Overlap detected, shift label down
                                        label_y = region[1] + region[3] + 10  # Add 10px spacing

                                # Draw Transparent Black Background for the label
                                overlay = frame.copy()
                                cv2.rectangle(overlay, (label_x, label_y),
                                              (label_x + label_width, label_y + label_height),
                                              (0, 0, 0), -1)
                                alpha = 0.4  # Transparency Level
                                cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

                                # Draw each line of text
                                current_y = label_y + 20  # Start drawing text with padding
                                for text in details:
                                    text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
                                    text_x = label_x + 10  # Add padding
                                    cv2.putText(frame, text, (text_x, current_y),
                                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, lineType=cv2.LINE_AA)
                                    current_y += text_size[1] + 10  # Move to the next line

                                # Add this label's region to the occupied regions list
                                occupied_regions.append((label_x, label_y, label_width, label_height))

                # Encode the processed frame as JPEG
                success, buffer = cv2.imencode(".jpg", frame)
                if not success:
                    print("[ERROR] Failed to encode frame")
                    continue

                frame_bytes = buffer.tobytes()

                # Upload the frame to Azure Blob Storage only if a person is verified
                if person_verified:
                    unique_id = str(uuid.uuid4())
                    await upload_image_to_azure_face("face_verification", frame_bytes, employee_id)
                    print(f"[INFO] Image uploaded to Azure for verified person: {unique_id}")

                # Try sending the frame. If the connection is closed, break out.
                try:
                    await websocket.send_bytes(frame_bytes)
                except Exception as send_error:
                    print(f"[ERROR] Face verification error: {send_error}")
                    break  # Exit the loop if sending fails

            except Exception as e:
                print(f"[ERROR] Face verification error during processing: {e}")
                break

            # Add a small delay to control the frame rate
            await asyncio.sleep(0.03)

    except WebSocketDisconnect:
        print("[INFO] WebSocket Disconnected: Face Verification")

    finally:
        # Remove the WebSocket from active connections
        if "verify" in active_websockets:
            del active_websockets["verify"]


# Run FastAPI
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)

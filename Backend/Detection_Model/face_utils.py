import insightface
import numpy as np

# Load InsightFace model
face_analyzer = insightface.app.FaceAnalysis()
face_analyzer.prepare(ctx_id=0, det_size=(1280, 1280))

# Extract face embeddings
def get_face_embedding(image):
    faces = face_analyzer.get(image)
    if len(faces) == 0:
        return None  # No face detected
    return faces[0].embedding / np.linalg.norm(faces[0].embedding)  # Normalize

def detect_faces(image):
    faces = face_analyzer.get(image)
    
    if not faces:
        return []
    
    face_data = []
    for face in faces:
        x1, y1, x2, y2 = face.bbox.astype(int)
        embedding = face.embedding / np.linalg.norm(face.embedding)  # Normalize
        face_data.append((x1, y1, x2, y2, embedding))

    return face_data
import numpy as np
from pymongo.collection import Collection
from azure.storage.blob import BlobServiceClient
import os
import dotenv
import os 
import shutil
from motor.motor_asyncio import AsyncIOMotorClient
from insightface.app import FaceAnalysis

dotenv.load_dotenv()


# Initialize Azure Blob Storage client
AZURE_STORAGE_CONNECTION_STRING=os.getenv("AZURE_STORAGE_CONNECTION_STRING")
blob_service_client = BlobServiceClient.from_connection_string(AZURE_STORAGE_CONNECTION_STRING)


# Connect to MongoDB
client = AsyncIOMotorClient(os.getenv('MONGO_URI'))
db = client[os.getenv('MONGO_DB')]
users_collection = db[os.getenv('MONGO_COLLECTION')]
results_collection = db[os.getenv('MONGO_RESULTS_COLLECTION')]



class FaceDetector:
    def __init__(self):
        """
        Initialize the FaceDetector using InsightFace.
        """
        self.face_app = FaceAnalysis()
        self.face_app.prepare(ctx_id=-1, det_size=(1280, 1280))  # GPU (ctx_id=0), adjust det_size if needed

    def detect_faces(self, frame):
        """
        Detect faces in a given frame.
        :param frame: Input image/frame (numpy array).
        :return: List of detected faces with bounding boxes and embeddings.
        """
        faces = self.face_app.get(frame)  # Detect faces
        detections = []
        embeddings = []

        for face in faces:
            bbox = list(map(int, face.bbox))  # Convert bbox to integers (x1, y1, x2, y2)
            confidence = float(face.det_score)  # Ensure confidence is a float
            print("Raw Bounding Box:", bbox, "Confidence:", confidence)  # Debug print

            if len(bbox) == 4:  # Ensure bbox has correct format
                detections.append([bbox[0], bbox[1], bbox[2], bbox[3], confidence])
                embeddings.append(face.normed_embedding)
            else:
                print("Invalid bbox format:", bbox)

        return detections, embeddings
    


face_detector = FaceDetector()

async def user_exists(email: str, collection) -> bool: 
    existing_user = await collection.find_one({"email": email})
    return existing_user is not None


async def load_all_embeddings(users_collection: Collection):
    """
    Load all user embeddings from the database along with their unique IDs and names.
    Returns:
        - embeddings (numpy array): Array of stored face embeddings.
        - users_info (list): List of user details corresponding to embeddings.
    """
    # Use Motor's to_list to convert the AsyncIOMotorCursor to a list
    docs = await users_collection.find(
        {},
        {"_id": 0, "embedding": 1, "unique_id": 1, "name": 1}
    ).to_list(length=None)

    if not docs:
        return None, None

    # Convert the stored list of embeddings into a NumPy array directly.
    embeddings = np.array([np.array(doc["embedding"]) for doc in docs])
    users_info = [{"unique_id": doc["unique_id"], "name": doc["name"]} for doc in docs]

    return embeddings, users_info


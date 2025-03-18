import os
from azure.storage.blob import BlobServiceClient
from fastapi import UploadFile
from pymongo import MongoClient
from dotenv import load_dotenv
from io import BytesIO
import uuid
from datetime import datetime,timezone

load_dotenv()

# Initialize Azure Blob Storage Client
AZURE_CONNECTION_STRING = os.getenv("AZURE_STORAGE_CONNECTION_STRING")
CONTAINER_NAME = os.getenv("CONTAINER_NAME")
blob_service_client = BlobServiceClient.from_connection_string(AZURE_CONNECTION_STRING)

# Initialize MongoDB Client
MONGO_URI = os.getenv("MONGO_URI")
mongo_client = MongoClient(MONGO_URI)
db = mongo_client["Streamer"]  # Change DB name if needed

async def upload_image_to_azure(model_name: str, image_data: UploadFile) -> str:
    """
    Uploads an image to Azure Blob Storage inside a model-specific folder and stores the URL in MongoDB.

    Parameters:
        - model_name (str): Name of the detection model (face, mask, helmet, hairnet).
        - unique_id (str): Unique ID of the detected person.
        - name (str): Name of the detected person.
        - image_file (UploadFile): Image file from FastAPI request.

    Returns:
        - (str): Public URL of the uploaded image if successful.
        - (None): If the upload fails.
    """
    try:
        # Define folder path based on the model
        unique_name = str(uuid.uuid4())
        file_name = f"{unique_name}.jpg"
        blob_path = f"{model_name}/{file_name}"


        # Use BytesIO to avoid saving file locally
        file_like = BytesIO(image_data)
        blob_client = blob_service_client.get_blob_client(container=CONTAINER_NAME, blob=blob_path)
        blob_client.upload_blob(file_like, overwrite=True)

        # Construct blob URL (update the storage account name as needed)
        blob_url = f"https://visiondetect.blob.core.windows.net/{CONTAINER_NAME}/{blob_path}"
        print(f"✅ Image uploaded successfully: {blob_url}")


        # Insert the URL into MongoDB under a model-specific collection
        collection = db[model_name]
        document = {
            "image_url": blob_url,
            "unique_name": unique_name
        }
        collection.insert_one(document)

        return blob_url

    except Exception as e:
        print(f"❌ Error uploading image to Azure: {e}")
        return None

async def upload_image_to_azure_face(model_name: str, image_data: bytes, employee_id: str) -> str:
    """
    Uploads an image to Azure Blob Storage inside a model-specific folder and stores the URL in MongoDB.
    Ensures that the same employee's image is not uploaded multiple times within 5 seconds.

    Parameters:
        - model_name (str): Name of the detection model (face, mask, helmet, hairnet).
        - image_data (bytes): Image data in bytes.
        - employee_id (str): Employee ID of the detected person.

    Returns:
        - (str): Public URL of the uploaded image if successful.
        - (None): If the upload fails or the image was already uploaded within 5 seconds.
    """
    try:
        # Check if the employee_id exists in the database and the last upload was within 5 seconds
        collection = db[model_name]
        last_upload = collection.find_one(
            {"employee_id": employee_id},
            sort=[("timestamp", -1)]  # Get the most recent upload
        )

        if last_upload:
            last_upload_time = last_upload["timestamp"]
            if last_upload_time.tzinfo is None:
                last_upload_time = last_upload_time.replace(tzinfo=timezone.utc)
            current_time = datetime.now(timezone.utc)
            time_difference = (current_time - last_upload_time).total_seconds()

            if time_difference < 86400:  # 24 hours
                print(f"🕒 Skipping upload for employee {employee_id}. Last upload was {time_difference / 3600:.2f} hours ago.")
                return None

        # Define folder path based on the model
        unique_name = str(uuid.uuid4())
        file_name = f"{unique_name}.jpg"
        blob_path = f"{model_name}/{file_name}"

        # Use BytesIO to avoid saving file locally
        file_like = BytesIO(image_data)
        blob_client = blob_service_client.get_blob_client(container=CONTAINER_NAME, blob=blob_path)
        blob_client.upload_blob(file_like, overwrite=True)

        # Construct blob URL (update the storage account name as needed)
        blob_url = f"https://visiondetect.blob.core.windows.net/{CONTAINER_NAME}/{blob_path}"
        print(f"✅ Image uploaded successfully: {blob_url}")

        # Insert the URL into MongoDB under a model-specific collection
        document = {
            "employee_id": employee_id,
            "image_url": blob_url,
            "unique_name": unique_name,
            "timestamp": datetime.now(timezone.utc)  # Store the current timestamp
        }
        collection.insert_one(document)

        return blob_url

    except Exception as e:
        print(f"❌ Error uploading image to Azure: {e}")
        return None
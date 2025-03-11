# Handles face detection using InsightFace


''' RetinaFace uses feature maps with strides of 8, 16, and 32, 
 the input resolution should ideally be divisible by 32. 
 so thats why it will not work for 1920x1080 and 1080x1080 but it will work for 1920x1088
 that is why 2048x2048 and 1024x1024 works and 1280x1280 works also 
'''

from insightface.app import FaceAnalysis


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


@app.post("/verify/")
async def verify_user(image: UploadFile = File(...)):
    try:
        # Read image
        image_bytes = await image.read()
        np_arr = np.frombuffer(image_bytes, np.uint8)
        frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        if frame is None:
            raise HTTPException(status_code=400, detail="Invalid image file. Could not decode.")

        # Detect faces & extract embeddings
        faces = [(x1, y1, x2, y2, embedding) for x1, y1, x2, y2, embedding in detect_faces(frame)]
        if not faces:
            return {"success": False, "message": "No face detected in the image."}

        # Extract embeddings & bounding boxes
        embeddings = np.array([face[4] for face in faces], dtype=np.float32)

        # Perform FAISS batch search
        D, I = index.search(embeddings, k=5)  # Search top-5 closest matches
        matched_users = []
        matched_ids = {int(i) for i in I.flatten() if i != -1}  # Convert FAISS indices to integers

        print(f"Matched FAISS IDs from index.search: {I}")  # Raw FAISS output
        print(f"Corresponding distances: {D}")  # Distance scores
        print(f"Extracted matched IDs: {matched_ids}")  # Final IDs to query MongoDB

        db_ids = face_collection.distinct("FAISS")
        print("Stored FAISS IDs in MongoDB:", db_ids)

        # Batch fetch user details from MongoDB
        users_info = {
            int(user["FAISS"]): user
            for user in face_collection.find(
                {"FAISS": {"$in": list(matched_ids)}},
                {"_id": 0, "FAISS": 1, "name": 1, "unique_id": 1, "email": 1, "phone": 1, "gender": 1, "date": 1}
            )
        }
        print(f"Users info from MongoDB: {users_info}")  # Debug print

        # Assign matched users to bounding boxes
        for (x1, y1, x2, y2, embedding), i_row, d_row in zip(faces, I, D):
            for i, dist in zip(i_row, d_row):
                user = users_info.get(int(i))  # Ensure i is treated as an integer
                if user and dist <= CONFIDENCE_THRESHOLD:
                    matched_users.append({
                        "name": user["name"],
                        "unique_id": user["unique_id"],
                        "email": user.get("email", ""),
                        "phone": user.get("phone", ""),
                        "gender": user.get("gender", ""),
                        "date": user.get("date", ""),
                        "similarity_score": float(dist),
                    })

                    # Draw bounding box & label
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(frame, user["name"], (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Save processed image locally
        output_path = "verified_faces.jpg"
        cv2.imwrite(output_path, frame)

        # Upload image to Azure & get the URL
        unique_id = str(uuid.uuid4())
        image_url = await upload_image_to_azure_results(unique_id, output_path)

        # Save results to MongoDB
        results = {
            "success": True,
            "message": f"{len(matched_users)} users verified",
            "matched_users": matched_users,
            "saved_image_path": output_path,
            "image_url": image_url
        }
        await results_collection.insert_one(results)

        # Return response
        return {
            "success": True,
            "message": "User verified successfully" if matched_users else "No matching users found",
            "image_url": image_url,
            "matched_users": matched_users
        }

    except Exception as e:
        print(f"Unexpected error in verification: {e}")
        return {"success": False, "message": "An unexpected error occurred. Please try again."}

    except FaceDetectionError as e:
        return {"success": False, "message": str(e)}



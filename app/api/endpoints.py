from fastapi import APIRouter, HTTPException, UploadFile, File, Query, Form
from fastapi.responses import JSONResponse
from typing import List, Optional
import numpy as np
import cv2
import os
import logging
from datetime import datetime
import tempfile
import time

# Import your modules
from app.database.db_handler import DatabaseHandler
from app.config import config
from app.face_processing.encoding import FaceEncoder
from app.face_processing.recognition import FaceRecognizer

router = APIRouter()
logger = logging.getLogger(__name__)

# Initialize components
try:
    db = DatabaseHandler(config.DATABASE_URL)
    face_encoder = FaceEncoder()
    face_recognizer = FaceRecognizer(face_encoder=face_encoder, registration_mode=True)
    logger.info("API components initialized successfully with InsightFace + SCRFD")
except Exception as e:
    logger.error(f"Failed to initialize API components: {e}")
    raise


#If you plan to support multi-angle registration (front, left, right, tilt), you can easily extend this route to accept multiple images at once:

#images: List[UploadFile] = File(...) and average their embeddings before saving — for more robust recognition

@router.post("/register")
async def register_face(
    name: str = Form(...),
    images: List[UploadFile] = File(default=[]),
    image: UploadFile = File(default=None)
):
    """
    Register a new face using InsightFace SCRFD + ArcFace.
    Supports both single image ('image' field) and multiple images ('images' field) for multi-angle registration.
    """
    try:
        embeddings = []
        processed_images = 0
        errors = []
        first_image_data = None
        detection_confidence = 0.0
        
        # Handle backward compatibility: if 'image' is provided, use it as single image
        all_images = images if images else []
        if image is not None:
            all_images = [image] if not images else [image] + images
        
        if not all_images:
            raise HTTPException(status_code=400, detail="No image provided. Please provide either 'image' or 'images' field")
        
        # Process each image
        for i, img in enumerate(all_images):
            try:
                # ✅ Validate image file
                if not img.content_type.startswith("image/"):
                    errors.append(f"Image {i+1}: File must be an image")
                    continue

                # ✅ Read uploaded image
                contents = await img.read()
                nparr = np.frombuffer(contents, np.uint8)
                frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)  # This loads as BGR
                if frame is None:
                    errors.append(f"Image {i+1}: Invalid image data")
                    continue

                logger.info(f"Processing registration for {name} - Image {i+1} shape: {frame.shape}")

                # ✅ Use FaceEncoder directly for better reliability
                face_locations, face_encodings, _ = face_encoder.process_frame(frame)
                if face_encodings:
                    embedding = face_encodings[0]  # Get the first face's embedding
                    faces = [{'bbox': loc, 'confidence': 1.0} for loc in face_locations]  # Create faces list with bbox and confidence
                else:
                    embedding, faces = None, []
                    
                if not faces:
                    errors.append(f"Image {i+1}: No face detected")
                    continue
                    
                if len(faces) > 1:
                    errors.append(f"Image {i+1}: Multiple faces detected - using first face")

                # ✅ Use the embedding from process_image (more reliable)
                if embedding is None:
                    errors.append(f"Image {i+1}: Failed to generate face encoding")
                    continue

                # Add embedding to list
                embeddings.append(embedding)
                processed_images += 1
                
                # Store the first valid image for database storage
                if first_image_data is None:
                    success, encoded_img = cv2.imencode(".jpg", frame)
                    if success:
                        first_image_data = encoded_img.tobytes()
                        face_info = faces[0]
                        detection_confidence = face_info['confidence']
                    else:
                        errors.append(f"Image {i+1}: Failed to encode image")
                        continue
                        
            except Exception as e:
                errors.append(f"Image {i+1}: Processing error - {str(e)}")
                continue
        
        # Validate that we have at least one valid embedding
        if not embeddings:
            error_msg = "No valid faces detected in any of the provided images. Please ensure:\n" \
                      "- The face is clearly visible\n" \
                      "- Good lighting conditions\n" \
                      "- Front-facing pose\n" \
                      "- No extreme angles or occlusions"
            if errors:
                error_msg += f"\n\nDetails:\n" + "\n".join(errors)
            raise HTTPException(status_code=400, detail=error_msg)
        
        # Calculate average embedding
        avg_embedding = np.mean(embeddings, axis=0)

        # ✅ Save user in database (using first image data and pre-computed embeddings)
        result = db.add_user(
            name=name,
            image_data=first_image_data,
            is_face_crop=False,  # Use full image, not cropped
            embeddings=embeddings,  # Pass pre-computed embeddings
            images_processed=processed_images  # Pass number of images processed
        )

        if not result or result.get("status") != "success":
            raise HTTPException(status_code=400, detail=result.get("message", "Failed to save user to database"))

        # ✅ Update in-memory recognizer cache with averaged embedding
        face_recognizer.known_face_encodings.append(avg_embedding)
        face_recognizer.known_face_names.append(name)
        face_recognizer.known_face_ids.append(result["id"])

        # ✅ Return success response
        return {
            "status": "success",
            "message": f"Face registered successfully with {processed_images} image(s)",
            "user_id": result["id"],
            "face_confidence": float(detection_confidence),
            "images_processed": processed_images,
            "model": "InsightFace (SCRFD + ArcFace)",
            "warnings": errors if errors else None
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in register_face: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Registration failed: {str(e)}")

@router.post("/recognize")
async def recognize_face(image: UploadFile = File(...), threshold: float = 0.7):
    """Recognize a face from the uploaded image using InsightFace."""
    try:
        # Validate file type
        if not image.content_type.startswith('image/'):
            raise HTTPException(status_code=400, detail="File must be an image")
        
        # Read and decode image
        contents = await image.read()
        nparr = np.frombuffer(contents, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if img is None:
            raise HTTPException(status_code=400, detail="Invalid image file")
        
        # Convert to bytes for recognition (matching your db_handler)
        _, buffer = cv2.imencode('.jpg', img)
        image_bytes = buffer.tobytes()
        
        # Find similar face - CORRECTED to match your db_handler
        match = db.find_similar_face(image_bytes, threshold=threshold)
        
        if not match:
            return {
                "status": "success", 
                "recognized": False, 
                "message": "No matching face found",
                "model": "InsightFace (SCRFD + ArcFace)"  # NEW: Added model info
            }
        
        return {
            "status": "success",
            "recognized": True,
            "user": {
                "id": match["id"],
                "name": match["name"],
                "confidence": float(match["similarity"]),
                "distance": float(match.get("distance", 0)),
                "faces_detected": match.get("faces_detected", 1),
                "primary_face_confidence": match.get("primary_face_confidence", 0)
            },
            "model": "InsightFace (SCRFD + ArcFace)"  # NEW: Added model info
        }
        
    except Exception as e:
        logger.error(f"Error in recognize_face: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Recognition failed: {str(e)}")

@router.get("/users")
async def get_users():
    """Get all registered users."""
    try:
        users = db.get_all_users()
        return {
            "status": "success", 
            "count": len(users), 
            "users": users,
            "model": "InsightFace (SCRFD + ArcFace)"  # NEW: Added model info
        }
    except Exception as e:
        logger.error(f"Error in get_users: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to fetch users: {str(e)}")

@router.get("/users/{user_id}")
async def get_user(user_id: int):
    """Get a specific user by ID."""
    try:
        user = db.get_user(user_id)  # This returns a User object
        if not user:
            raise HTTPException(status_code=404, detail="User not found")
        
        # Convert User object to dictionary
        user_data = {
            "id": user.id,
            "name": user.name,
            "date_created": user.date_created,
            "last_accessed": user.last_accessed,
            "image_format": user.image_format
        }
        
        return {
            "status": "success", 
            "user": user_data,
            "model": "InsightFace (SCRFD + ArcFace)"  # NEW: Added model info
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in get_user: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to fetch user: {str(e)}")

@router.get("/users/{user_id}/image")
async def get_user_image(user_id: int):
    """Get user image data."""
    try:
        result = db.get_user_image(user_id)
        if not result:
            raise HTTPException(status_code=404, detail="User image not found")
        
        image_data, image_format = result
        
        # Return as base64 or direct bytes depending on your needs
        from fastapi.responses import Response
        return Response(content=image_data, media_type=f"image/{image_format}")
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in get_user_image: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to fetch user image: {str(e)}")

@router.delete("/users/{user_id}")
async def delete_user(user_id: int):
    """Delete a user by ID."""
    try:
        result = db.delete_user(user_id)
        if result.get('status') != 'success':
            raise HTTPException(status_code=404, detail=result.get('message', 'User not found'))
        
        return {
            "status": "success", 
            "message": "User deleted successfully",
            "model": "InsightFace (SCRFD + ArcFace)"  # NEW: Added model info
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in delete_user: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to delete user: {str(e)}")

@router.put("/users/{user_id}/face")
async def update_user_face(user_id: int, image: UploadFile = File(...)):
    """Update user's face encoding using InsightFace."""
    try:
        # Validate file type
        if not image.content_type.startswith('image/'):
            raise HTTPException(status_code=400, detail="File must be an image")
        
        # Read and decode image
        contents = await image.read()
        
        # Update user face
        result = db.update_user_face(user_id, contents)
        
        if result.get('status') != 'success':
            raise HTTPException(status_code=400, detail=result.get('message', 'Failed to update face'))
        
        return {
            "status": "success",
            "message": "Face updated successfully",
            "user_id": user_id,
            "faces_detected": result.get('faces_detected', 0),
            "confidence": result.get('primary_face_confidence', 0),
            "model": "InsightFace (SCRFD + ArcFace)"  # NEW: Added model info
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in update_user_face: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to update face: {str(e)}")

@router.get("/logs")
async def get_logs(limit: int = 100):
    """Get access logs."""
    try:
        logs = db.get_access_logs(limit=limit)
        return {
            "status": "success", 
            "count": len(logs), 
            "logs": logs,
            "model": "InsightFace (SCRFD + ArcFace)"  # NEW: Added model info
        }    
    except Exception as e:
        logger.error(f"Error in get_logs: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to fetch logs: {str(e)}")
              

@router.get("/stats")
async def get_stats():
    """Get system statistics."""
    try:
        user_count = db.get_user_count()
        recent_logs = db.get_recent_access_logs(hours=24)
        
        return {
            "status": "success",
            "stats": {
                "total_users": user_count,
                "access_attempts_24h": len(recent_logs),
                "successful_access": len([log for log in recent_logs if log.get('access_granted')]),
                "failed_access": len([log for log in recent_logs if not log.get('access_granted')])
            },
            "model": "InsightFace (SCRFD + ArcFace)"  # NEW: Added model info
        }
    except Exception as e:
        logger.error(f"Error in get_stats: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to fetch stats: {str(e)}")

@router.post("/logs/cleanup")
async def cleanup_logs(days: int = 30):
    """Clean up old access logs."""
    try:
        deleted_count = db.cleanup_old_logs(days=days)
        return {
            "status": "success",
            "message": f"Cleaned up {deleted_count} logs older than {days} days",
            "model": "InsightFace (SCRFD + ArcFace)"  # NEW: Added model info
        }
    except Exception as e:
        logger.error(f"Error in cleanup_logs: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to cleanup logs: {str(e)}")

@router.get("/health")
async def health_check():
    """Health check endpoint."""
    try:
        # Test database connection by getting user count
        user_count = db.get_user_count()
        return {
            "status": "healthy",
            "database": "connected",
            "total_users": user_count,
            "model": "InsightFace (SCRFD + ArcFace)",  # NEW: Added model info
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        raise HTTPException(status_code=503, detail="Service unhealthy")
            
@router.post("/video_recognition")
async def video_recognition(
    video: UploadFile = File(...),
    threshold: float = Query(0.7, description="Face match similarity threshold"),
    frame_skip: int = Query(15, description="Process every Nth frame for speed optimization"),
    batch_size: int = Query(16, description="Number of frames to process in each batch"),
):
    """
    Process uploaded video for face recognition using InsightFace with batch processing.
    - Processes frames in batches for better GPU utilization
    - Detects faces using SCRFD
    - Matches faces with registered users using ArcFace
    - Returns recognition results with timing information
    """
    from fastapi.concurrency import run_in_threadpool
    import asyncio
    from concurrent.futures import ThreadPoolExecutor
    from typing import List, Tuple, Dict, Any
    
    async def process_video_batch(frames_batch: List[Tuple[int, np.ndarray]], 
                                threshold: float) -> List[Dict[str, Any]]:
        """Process a batch of frames and return recognition results."""
        batch_results = []
        
        # Process frames in parallel using ThreadPoolExecutor
        with ThreadPoolExecutor() as executor:
            loop = asyncio.get_event_loop()
            tasks = []
            
            for frame_idx, frame in frames_batch:
                # Convert frame to bytes for the database function
                success, buffer = cv2.imencode(".jpg", frame)
                if not success:
                    continue
                frame_bytes = buffer.tobytes()
                
                # Run recognition in thread pool
                task = loop.run_in_executor(
                    executor, 
                    db.find_similar_face,
                    frame_bytes,
                    threshold
                )
                tasks.append((frame_idx, frame, task))
            
            # Gather all results
            for frame_idx, frame, task in tasks:
                try:
                    match = await task
                    
                    if match:
                        user_id = match["id"]
                        db.log_access_attempt(
                            user_id=user_id, 
                            confidence=match["similarity"], 
                            access_granted=True
                        )
                        
                        batch_results.append({
                            "frame": frame_idx,
                            "user_id": user_id,
                            "name": match["name"],
                            "confidence": match["similarity"],
                            "status": "VALID"
                        })
                    else:
                        db.log_access_attempt(
                            user_id=None, 
                            confidence=0.0, 
                            access_granted=False
                        )
                        
                        batch_results.append({
                            "frame": frame_idx,
                            "user_id": None,
                            "name": "Unknown",
                            "confidence": 0.0,
                            "status": "INVALID"
                        })
                except Exception as e:
                    logger.error(f"Error processing frame {frame_idx}: {e}")
                    
        return batch_results

    start_time = time.time()
    tmp_path = None
    
    try:
        # Save uploaded video temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp_video:
            tmp_video.write(await video.read())
            tmp_path = tmp_video.name

        # Open video file
        cap = cv2.VideoCapture(tmp_path)
        if not cap.isOpened():
            raise HTTPException(status_code=400, detail="Cannot open uploaded video")

        # Get video properties
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_rate = int(cap.get(cv2.CAP_PROP_FPS)) or 30
        
        # Initialize counters
        processed = 0
        recognized = 0
        unrecognized = 0
        results = []
        frame_index = 0
        
        # Process frames in batches
        while True:
            frames_batch = []
            
            # Read a batch of frames
            while len(frames_batch) < batch_size:
                ret, frame = cap.read()
                if not ret:
                    break
                    
                # Skip frames according to frame_skip
                if frame_index % frame_skip == 0:
                    frames_batch.append((frame_index, frame))
                    processed += 1
                
                frame_index += 1
                
                # Stop if we've reached the end of the video
                if frame_index >= total_frames:
                    break
            
            # If no frames were read, we're done
            if not frames_batch:
                break
            
            # Process the current batch
            batch_results = await process_video_batch(frames_batch, threshold)
            
            # Update counters
            for result in batch_results:
                if result["status"] == "VALID":
                    recognized += 1
                else:
                    unrecognized += 1
            
            results.extend(batch_results)
            
            # If we've reached the end of the video, break the loop
            if frame_index >= total_frames:
                break
        
        # Release video capture
        cap.release()
        
        # Calculate processing time
        total_time = time.time() - start_time
        
        # Prepare response
        summary = {
            "status": "success",
            "video_info": {
                "frames_total": total_frames,
                "frames_processed": processed,
                "frame_rate": frame_rate,
                "batch_size": batch_size,
                "frame_skip": frame_skip
            },
            "results": {
                "recognized": recognized,
                "unrecognized": unrecognized,
                "recognition_rate": round(recognized / max(processed, 1), 2),
                "processing_time_sec": round(total_time, 2),
                "avg_time_per_frame_ms": round((total_time / max(processed, 1)) * 1000, 2),
                "fps": round(processed / max(total_time, 0.1), 2)
            },
            "logs": results[-10:],  # Last 10 detections
            "model": "InsightFace (SCRFD + ArcFace)",
            "timestamp": datetime.now().isoformat()
        }
        
        return JSONResponse(content=summary)

    except Exception as e:
        logger.exception(f"Error in video recognition: {e}")
        raise HTTPException(status_code=500, detail=f"Video recognition failed: {str(e)}")

    finally:
        # Cleanup
        try:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
        except Exception:
            pass
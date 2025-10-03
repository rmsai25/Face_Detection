import cv2
import numpy as np
import os
import logging
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Any, Union
import time

from retinaface import RetinaFace

logger = logging.getLogger(__name__)

class FaceRecognizer:
    def __init__(self, face_encoder, registration_mode: bool = False, create_dirs: bool = False):
        """
        Initialize FaceRecognizer.

        Args:
            face_encoder: Instance of FaceEncoder for encoding & matching
            registration_mode: True for registration mode, False for recognition mode
            create_dirs: If True, create directories (default False)
        """
        self.face_encoder = face_encoder
        self.registration_mode = registration_mode
        self.known_face_encodings = []
        self.known_face_names = []
        self.known_face_ids = []
        
        # Registration state management
        self.registration_pending = False
        self.last_registration_time = 0
        self.registration_cooldown = 2  # seconds

        # Directories for registered faces and access logs (not used for storage, only for compatibility)
        self.registered_faces_dir = Path("registered_faces")
        self.access_logs_dir = Path("access_logs")
        
        # Disable local file creation - all data will be stored in the database only
        self.use_local_storage = False

        # Performance tracking
        self.processing_times = []
        self.max_processing_time_history = 100

        logger.info(f"FaceRecognizer initialized in {'registration' if registration_mode else 'recognition'} mode")

    def load_known_faces(self, users: List[Dict]) -> bool:
        """
        Load known face encodings and names from database with validation.

        Args:
            users: List of user dicts containing face encodings and metadata

        Returns:
            True if successful, False otherwise
        """
        self.known_face_encodings = []
        self.known_face_names = []
        self.known_face_ids = []

        if not isinstance(users, list):
            logger.error("Invalid users data: expected list")
            return False

        if not users:
            logger.warning("No users provided for loading known faces")
            return True

        valid_count = 0
        for user in users:
            try:
                if not isinstance(user, dict):
                    logger.warning(f"Skipping invalid user data: {user}")
                    continue

                user_id = user.get('id')
                user_name = user.get('name', 'Unknown')
                encoding = user.get('face_encoding', None)

                if encoding is None:
                    logger.debug(f"User {user_id} has no face encoding")
                    continue

                # Convert encoding to numpy array if needed
                if not isinstance(encoding, np.ndarray):
                    try:
                        if isinstance(encoding, list):
                            encoding = np.array(encoding, dtype=np.float32)
                        else:
                            logger.warning(f"Unexpected encoding type for user {user_id}: {type(encoding)}")
                            continue
                    except Exception as e:
                        logger.warning(f"Failed to convert encoding for user {user_id}: {e}")
                        continue

                if encoding.size == 0:
                    logger.warning(f"Skipping empty encoding for user {user_id}")
                    continue

                # Normalize encoding
                encoding = encoding.flatten()
                norm = np.linalg.norm(encoding)
                if norm < 1e-10:  # Avoid division by zero
                    logger.warning(f"Skipping zero-vector encoding for user {user_id}")
                    continue

                encoding = encoding / norm

                self.known_face_encodings.append(encoding)
                self.known_face_names.append(user_name)
                self.known_face_ids.append(user_id)
                valid_count += 1

                logger.debug(f"Loaded face for user {user_id}: {user_name} (encoding shape: {encoding.shape})")

            except Exception as e:
                logger.error(f"Error loading user {user.get('id', 'unknown')}: {e}")

        logger.info(f"Successfully loaded {valid_count} out of {len(users)} users with valid face encodings")
        return valid_count > 0

    def _detect_faces_retinaface(self, frame: np.ndarray) -> List[Dict[str, Any]]:
        """
        Detect faces using RetinaFace.

        Args:
            frame: BGR image from OpenCV

        Returns:
            List of detected faces with bounding boxes & confidence
        """
        try:
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            faces = RetinaFace.detect_faces(rgb_frame)

            face_data = []
            if isinstance(faces, dict):
                for face_id, face_info in faces.items():
                    try:
                        confidence = face_info.get('score', 0)
                        
                        # Filter by confidence
                        if confidence < 0.7:  # Lowered threshold for better detection
                            continue
                            
                        facial_area = face_info.get('facial_area')
                        if facial_area is None:
                            continue
                            
                        x1, y1, x2, y2 = map(int, facial_area)
                        
                        # Validate bounding box
                        if x1 >= x2 or y1 >= y2:
                            continue
                            
                        if x1 < 0 or y1 < 0 or x2 > frame.shape[1] or y2 > frame.shape[0]:
                            continue

                        face_data.append({
                            'facial_area': (x1, y1, x2, y2),
                            'confidence': confidence,
                            'landmarks': face_info.get('landmarks', {})
                        })
                        
                    except Exception as e:
                        logger.warning(f"Error processing face {face_id}: {e}")
                        continue
                        
            return face_data
            
        except Exception as e:
            logger.error(f"Error in face detection: {e}")
            return []
            
    '''def _detect_faces_retinaface(self, frame: np.ndarray) -> List[Dict[str, Any]]:
        """
        Optimized face detection with RetinaFace using frame resizing and a lighter model.
        """
        try:
            # --- 1. Frame Resizing for Speed (The main optimization) ---
            height, width = frame.shape[:2]
        
            # Set a fixed maximum dimension for the detection process.
            # Detecting faces on a smaller image is much faster.
            MAX_DETECTION_DIM = 640  
        
            # Calculate scale factor
            scale = min(MAX_DETECTION_DIM / width, MAX_DETECTION_DIM / height)
        
            # Only resize if the frame is larger than the max detection dimension
            if scale < 1.0:
                # New dimensions for the smaller frame
                new_width = int(width * scale)
                new_height = int(height * scale)
            
                # Use INTER_AREA for image shrinking (better quality)
                small_frame = cv2.resize(frame, (new_width, new_height), interpolation=cv2.INTER_AREA)
            else:
                small_frame = frame
                scale = 1.0 # Ensure scale is 1.0 if no resizing occurred

            # Convert to RGB (RetinaFace requirement)
            rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
        
            # --- 2. Optimized Face Detection Call ---
            # Use the lighter model ('mobile0.25') and set the confidence threshold directly
            # in the detector for an efficient filter.
            FACES_THRESHOLD = 0.7  
            faces = RetinaFace.detect_faces(
                rgb_small_frame,
                threshold=FACES_THRESHOLD, 
                model='mobile0.25' 
            )

            face_data = []
            if isinstance(faces, dict):
                for face_info in faces.values():
                    try:
                        confidence = face_info.get('score', 0)
                        # No need for a second confidence check if the threshold was applied in detect_faces
                    
                        facial_area = face_info.get('facial_area')
                        if facial_area is None:
                            continue
                        
                        # Scale coordinates back to the original frame size
                        # Note: facial_area from RetinaFace is [x1, y1, x2, y2]
                        x1_s, y1_s, x2_s, y2_s = map(int, facial_area)
                    
                        # Inverse scaling to map to original frame
                        x1 = int(x1_s / scale)
                        y1 = int(y1_s / scale)
                        x2 = int(x2_s / scale)
                        y2 = int(y2_s / scale)
                    
                        # Clip coordinates to frame boundaries for safety
                        x1 = max(0, x1)
                        y1 = max(0, y1)
                        x2 = min(width, x2)
                        y2 = min(height, y2)
                    
                        # Final validation
                        if x1 >= x2 or y1 >= y2:
                            continue
                        
                        face_data.append({
                            'facial_area': (x1, y1, x2, y2),
                            'confidence': confidence,
                            'landmarks': face_info.get('landmarks', {})
                        })
                        
                    except Exception as e:
                        # NOTE: Assuming 'logger' is accessible/imported
                        logger.warning(f"Error processing a detected face: {e}")
                        continue
                        
            return face_data
            
        except Exception as e:
        # NOTE: Assuming 'logger' is accessible/imported
            logger.error(f"Critical error in RetinaFace detection: {e}")
            return []  '''      

    def process_frame(self, frame: np.ndarray, db_handler = None) -> Tuple[np.ndarray, List[Dict]]:
        """
        Detect, recognize or register faces in a frame.

        Args:
            frame: BGR image frame from OpenCV
            db_handler: Database handler for registration/logging

        Returns:
            Tuple of (processed_frame, face_data_list)
        """
        start_time = time.time()
        processed_frame = frame.copy()
        face_data = []

        try:
            # Detect faces
            detected_faces = self._detect_faces_retinaface(frame)
            if not detected_faces:
                # Show "No faces detected" message on frame
                cv2.putText(processed_frame, "No faces detected", (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                return processed_frame, face_data

            # Process each detected face
            for face_info in detected_faces:
                x1, y1, x2, y2 = face_info['facial_area']
                detection_confidence = face_info['confidence']

                # Extract face region with padding
                padding = int((x2 - x1) * 0.1)  # Reduced padding for better encoding
                x1_padded = max(0, x1 - padding)
                y1_padded = max(0, y1 - padding)
                x2_padded = min(frame.shape[1], x2 + padding)
                y2_padded = min(frame.shape[0], y2 + padding)

                face_region = frame[y1_padded:y2_padded, x1_padded:x2_padded]
                if face_region.size == 0:
                    logger.warning("Empty face region after padding")
                    continue

                # Encode the face region
                face_encoding = self.face_encoder.encode_face_from_crop(face_region)
                
                if face_encoding is None:
                    # Draw red box for encoding failure
                    cv2.rectangle(processed_frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                    cv2.putText(processed_frame, "Encoding failed", 
                               (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
                    continue

                # Recognize or register face
                recognized_face = None
                
                if self.registration_mode:
                    recognized_face = self._register_face(face_region, face_encoding, db_handler, (x1, y1, x2, y2))
                else:
                    recognized_face = self._recognize_face(face_encoding, db_handler, (x1, y1, x2, y2))

                # Draw visualization
                self._draw_face_annotation(processed_frame, face_info, recognized_face, self.registration_mode)

                if recognized_face:
                    face_data.append(recognized_face)

            # Track processing performance
            processing_time = time.time() - start_time
            self.processing_times.append(processing_time)
            if len(self.processing_times) > self.max_processing_time_history:
                self.processing_times.pop(0)

            # Display performance info
            avg_time = np.mean(self.processing_times) if self.processing_times else 0
            fps = 1.0 / avg_time if avg_time > 0 else 0
            cv2.putText(processed_frame, f"FPS: {fps:.1f}", (10, processed_frame.shape[0] - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

            return processed_frame, face_data

        except Exception as e:
            logger.error(f"Error processing frame: {e}")
            cv2.putText(processed_frame, "Processing error", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            return processed_frame, []
            
    '''def process_frame(self, frame: np.ndarray, db_handler=None) -> Tuple[np.ndarray, List[Dict]]:
        """
        Detect, recognize or register faces in a frame with optimized batch processing.

        Args:
            frame: BGR image frame from OpenCV
            db_handler: Database handler for registration/logging

        Returns:
            Tuple of (processed_frame, face_data_list)
        """
        start_time = time.time()
        processed_frame = frame.copy() # Use a copy for drawing annotations
        face_data_list = [] # Final list of processed face data

        try:
            # 1. Face Detection (Calls the optimized _detect_faces_retinaface)
            detected_faces = self._detect_faces_retinaface(frame)
        
            if not detected_faces:
                # Show "No faces detected" message on frame
                cv2.putText(processed_frame, "No faces detected", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                # Track processing time even for missed detections
                self._update_performance_metrics(start_time, processed_frame)
                return processed_frame, face_data_list

            # --- Setup for Batch Processing ---
            face_encodings = []
            valid_faces = []

            # 2. Extract and Encode Valid Faces
            for face_info in detected_faces:
                x1, y1, x2, y2 = face_info['facial_area']
            
                # Use original detected area for encoding, skipping complex padding logic
                # This relies on the detector providing a tight, clean bounding box.
                face_region = frame[y1:y2, x1:x2]
            
                # Robustness check: Skip small or empty faces
                if face_region.size == 0 or face_region.shape[0] < 40 or face_region.shape[1] < 40:
                    cv2.rectangle(processed_frame, (x1, y1), (x2, y2), (255, 0, 0), 2) # Blue box for skipped
                    continue
                
                # Get face encoding
                encoding = self.face_encoder.encode_face_from_crop(face_region)
            
                if encoding is not None:
                    face_encodings.append(encoding)
                    # Store the original detection data (which will be updated later)
                    valid_faces.append(face_info) 
                else:
                    # Draw red box for encoding failure
                    cv2.rectangle(processed_frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                    cv2.putText(processed_frame, "Encoding failed",
                                (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
                    continue

            # 3. Recognition/Registration Logic (Batch for Recognition)
            if self.registration_mode:
                # Registration is typically single-face, so we iterate and call the method
                for i, face in enumerate(valid_faces):
                    face_encoding = face_encodings[i]
                    x1, y1, x2, y2 = face['facial_area']
                    face_region = frame[y1:y2, x1:x2] # Re-extract for registration
                
                    recognized_face = self._register_face(face_region, face_encoding, db_handler, face['facial_area'])
                
                    # Draw visualization
                    self._draw_face_annotation(processed_frame, face, recognized_face, self.registration_mode)
                
                    if recognized_face:
                        face_data_list.append(recognized_face)

            elif face_encodings:
                # BATCH RECOGNITION
                recognized_faces = self._recognize_faces(face_encodings)
            
                # Match results back to the original face data
                for i, (face, recognized) in enumerate(zip(valid_faces, recognized_faces)):
                    # recognized contains 'name', 'distance', 'is_recognized'
                    face.update(recognized)
                    face_data_list.append(face)
                
                    # Draw annotations
                    self._draw_face_annotation(processed_frame, face, recognized, self.registration_mode)

            # 4. Performance Tracking (Simplified and moved to a helper)
            self._update_performance_metrics(start_time, processed_frame)
        
            return processed_frame, face_data_list

        except Exception as e:
            # NOTE: Assuming 'logger' is accessible/imported
            logger.error(f"Error processing frame: {e}")
            cv2.putText(processed_frame, "Processing error", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            return processed_frame, []

    # NOTE: This assumes the class has a new helper method for cleaner FPS display/tracking
    # You would need to add this helper method to your class (e.g., FaceRecognitionSystem)
    def _update_performance_metrics(self, start_time: float, frame: np.ndarray):
        """Internal helper to calculate and display FPS"""
        import numpy as np # Need numpy for mean
    
        processing_time = time.time() - start_time
        self.processing_times.append(processing_time)
    
        # Maintain history size
        if len(self.processing_times) > self.max_processing_time_history:
            self.processing_times.pop(0)

        # Calculate average time and FPS
        avg_time = np.mean(self.processing_times) if self.processing_times else 0
        fps = 1.0 / avg_time if avg_time > 0 else 0
    
        # Display performance info (using a different color/position for clarity from error)
        cv2.putText(frame, f"FPS: {fps:.1f}", (10, frame.shape[0] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1) '''       

    def _draw_face_annotation(self, frame: np.ndarray, face_info: Dict, 
                            recognized_face: Optional[Dict], registration_mode: bool) -> None:
        """
        Draw face bounding box and annotations on frame.
        """
        x1, y1, x2, y2 = face_info['facial_area']
        detection_confidence = face_info['confidence']

        # Determine color and label based on mode and recognition result
        if registration_mode:
            label = "Press 's' to register"
            color = (0, 255, 255)  # Yellow for registration
        else:
            if recognized_face and recognized_face.get('name') != "Unknown":
                name = recognized_face['name']
                confidence = recognized_face.get('confidence', 0)
                label = f"{name} ({confidence:.2f})"
                color = (0, 255, 0)  # Green for recognized
            else:
                label = "Unknown"
                color = (0, 0, 255)  # Red for unknown

        # Draw bounding box
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        
        # Draw detection confidence
        cv2.putText(frame, f"Det: {detection_confidence:.2f}", 
                   (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Draw name label (adjust position if near top of frame)
        label_y = y1 - 30 if y1 > 50 else y2 + 25
        cv2.putText(frame, label, (x1, label_y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        # Draw landmarks if available
        landmarks = face_info.get('landmarks', {})
        for landmark_name, (lx, ly) in landmarks.items():
            cv2.circle(frame, (int(lx), int(ly)), 2, (255, 0, 0), -1)

    def _register_face(self, face_region: np.ndarray, face_encoding: np.ndarray, 
                      db_handler, face_location: Tuple) -> Optional[Dict]:
        """
        Register a new face.

        Args:
            face_region: Cropped face image (BGR)
            face_encoding: Face embedding vector
            db_handler: Database handler instance
            face_location: (x1, y1, x2, y2) bounding box

        Returns:
            Face info for registered face or None
        """
        try:
            # Check for registration trigger (space key)
            key = cv2.waitKey(1) & 0xFF
            current_time = time.time()
            
            if key == ord(' ') and not self.registration_pending and \
               (current_time - self.last_registration_time) > self.registration_cooldown:
                self.registration_pending = True
                self.last_registration_time = current_time
                logger.info("Registration triggered by space key")

            if not self.registration_pending:
                return None
                
            # Verify that we have a valid face region and encoding
            if face_region is None or face_region.size == 0 or face_encoding is None:
                logger.error("No valid face detected for registration")
                print("No face detected in camera view. Please position your face in the frame.")
                self.registration_pending = False
                return None

            # Get name from user input (non-blocking)
            name = input("Enter name for new face (or press Enter to skip): ").strip()
            if not name:
                logger.info("Registration skipped - no name provided")
                self.registration_pending = False
                return None

            # Convert face region to bytes for database
            success, encoded_image = cv2.imencode('.jpg', face_region)
            if not success:
                logger.error("Failed to encode face image for registration")
                self.registration_pending = False
                return None

            image_data = encoded_image.tobytes()

            # Add user to database
            result = db_handler.add_user(name=name, image_data=image_data, is_face_crop=True)
            
            if result and result.get('status') == 'success':
                user_id = result.get('id')
                logger.info(f"Registered new user: {name} with ID: {user_id}")

                # Update in-memory cache
                self.known_face_encodings.append(face_encoding)
                self.known_face_names.append(name)
                self.known_face_ids.append(user_id)

                x1, y1, x2, y2 = face_location
                self.registration_pending = False
                
                return {
                    'name': name,
                    'user_id': user_id,
                    'confidence': 1.0,
                    'distance': 0.0,
                    'registered': True,
                    'face_location': (y1, x2, y2, x1)
                }
            else:
                logger.error(f"Failed to register user in database: {result}")
                self.registration_pending = False

        except Exception as e:
            logger.error(f"Error registering face: {e}")
            self.registration_pending = False

        return None

    def _recognize_face(self, face_encoding: np.ndarray, db_handler, 
                       face_location: Tuple) -> Optional[Dict]:
        """
        Recognize a face from encoding.

        Args:
            face_encoding: Face embedding vector
            db_handler: Database handler instance
            face_location: (x1, y1, x2, y2) bounding box

        Returns:
            Face info for recognized face or None
        """
        if not self.known_face_encodings:
            return {
                'name': "Unknown",
                'user_id': None,
                'confidence': 0.0,
                'distance': 1.0,
                'registered': False,
                'face_location': face_location
            }

        try:
            best_match_name, confidence, verified = self.face_encoder.find_best_match(
                face_encoding, self.known_face_encodings, self.known_face_names)

            if best_match_name and verified and confidence > 0.5:  # Additional confidence check
                match_idx = self.known_face_names.index(best_match_name)
                user_id = self.known_face_ids[match_idx]

                # Log access attempt
                if db_handler:
                    try:
                        db_handler.log_access_attempt(
                            user_id=user_id,
                            confidence=confidence,
                            access_granted=True
                        )
                    except Exception as e:
                        logger.warning(f"Failed to log access attempt: {e}")

                x1, y1, x2, y2 = face_location
                return {
                    'name': best_match_name,
                    'user_id': user_id,
                    'confidence': confidence,
                    'distance': 1.0 - confidence,
                    'registered': True,
                    'face_location': (y1, x2, y2, x1)
                }

        except Exception as e:
            logger.error(f"Error in face recognition: {e}")

        return {
            'name': "Unknown",
            'user_id': None,
            'confidence': 0.0,
            'distance': 1.0,
            'registered': False,
            'face_location': face_location
        }

    def _get_face_region_from_location(self, face_location: Tuple) -> np.ndarray:
        """Extract face region from frame based on location (placeholder)."""
        # This would need access to the original frame
        # For now, return a black image as placeholder
        x1, y1, x2, y2 = face_location
        return np.zeros((y2-y1, x2-x1, 3), dtype=np.uint8)

    def recognize_faces(self, face_encodings: List[Union[np.ndarray, List]]) -> List[Dict]:
        """
        Recognize multiple face encodings against known faces.

        Args:
            face_encodings: List of face encoding vectors

        Returns:
            List of recognized face info dicts
        """
        recognized_faces = []

        for i, face_encoding in enumerate(face_encodings):
            try:
                if not self.known_face_encodings:
                    recognized_faces.append({
                        'name': "Unknown",
                        'user_id': None,
                        'confidence': 0.0
                    })
                    continue

                if isinstance(face_encoding, list):
                    face_encoding = np.array(face_encoding, dtype=np.float32)

                best_match_name, confidence, verified = self.face_encoder.find_best_match(
                    face_encoding, self.known_face_encodings, self.known_face_names)

                if best_match_name and verified:
                    match_idx = self.known_face_names.index(best_match_name)
                    user_id = self.known_face_ids[match_idx]
                else:
                    best_match_name = "Unknown"
                    user_id = None
                    confidence = 0.0

                recognized_faces.append({
                    'name': best_match_name,
                    'user_id': user_id,
                    'confidence': float(confidence)
                })

            except Exception as e:
                logger.error(f"Error recognizing face encoding {i}: {e}")
                recognized_faces.append({
                    'name': "Unknown",
                    'user_id': None,
                    'confidence': 0.0
                })

        return recognized_faces

    def get_performance_stats(self) -> Dict[str, float]:
        """
        Get performance statistics.

        Returns:
            Dictionary with performance metrics
        """
        if not self.processing_times:
            return {'avg_processing_time': 0, 'fps': 0}

        avg_time = np.mean(self.processing_times)
        fps = 1.0 / avg_time if avg_time > 0 else 0
        
        return {
            'avg_processing_time': avg_time,
            'fps': fps,
            'samples': len(self.processing_times)
        }

    def set_registration_mode(self, enabled: bool):
        """
        Set registration mode.

        Args:
            enabled: True for registration mode, False for recognition mode
        """
        self.registration_mode = enabled
        logger.info(f"Registration mode {'enabled' if enabled else 'disabled'}")

    def clear_known_faces(self):
        """Clear all known face data from memory."""
        self.known_face_encodings.clear()
        self.known_face_names.clear()
        self.known_face_ids.clear()
        logger.info("Cleared all known face data from memory")



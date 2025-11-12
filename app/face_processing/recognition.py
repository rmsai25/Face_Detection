import cv2
import numpy as np
import os
import logging
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Any, Union
import time
import torch
import threading
import asyncio
from concurrent.futures import ThreadPoolExecutor, as_completed
from queue import Queue, Empty

# GPU optimization settings
torch.set_num_threads(os.cpu_count() or 4)
torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

import insightface
from insightface.app import FaceAnalysis

logger = logging.getLogger(__name__)

class FaceRecognizer:
    def __init__(self, face_encoder, registration_mode: bool = False, create_dirs: bool = False, 
                 max_workers: int = 4, batch_size: int = 4):
        """
        Initialize FaceRecognizer with parallel processing and GPU optimization.
        
        Args:
            face_encoder: Instance of FaceEncoder for encoding & matching
            registration_mode: True for registration mode, False for recognition mode
            create_dirs: If True, create directories (default False)
            max_workers: Number of parallel workers for face detection
            batch_size: Batch size for parallel processing
        """
        # 1. DETERMINE DEVICE WITH OPTIMIZATION
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"FaceRecognizer using device: {self.device}")
        
        # 2. PARALLEL PROCESSING SETUP
        self.max_workers = max_workers
        self.batch_size = batch_size
        self.parallel_executor = ThreadPoolExecutor(max_workers=max_workers)
        self.processing_queue = Queue(maxsize=10)
        self.results_queue = Queue()
        
        # 3. INITIALIZE FACE ENCODER AND INSIGHTFACE APP
        self.face_encoder = face_encoder
        
        # Store reference to InsightFace app from face_encoder or create optimized instance
        if hasattr(face_encoder, 'app'):
            self.insightface_app = face_encoder.app
            logger.info("Using InsightFace model from FaceEncoder")
        else:
            # Create optimized InsightFace instance
            providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'] if self.device.type == 'cuda' else ['CPUExecutionProvider']
            self.insightface_app = FaceAnalysis(
                name='buffalo_l',
                providers=providers
            )
            ctx_id = 0 if self.device.type == 'cuda' else -1
            self.insightface_app.prepare(ctx_id=ctx_id, det_size=(640, 640))
            logger.info("Created optimized InsightFace model")
            
        # 4. INITIALIZE GPU SETTINGS (after insightface_app is initialized)
        self._init_gpu_settings()
        
        # 5. INITIALIZE STATE VARIABLES
        self.registration_mode = registration_mode
        self.known_face_encodings = []
        self.known_face_names = []
        self.known_face_ids = []
        
        # Thread-safe data structures
        self.known_face_lock = threading.Lock()
        
        # Registration state management
        self.registration_pending = False
        self.last_registration_time = 0
        self.registration_cooldown = 2  # seconds

        # Performance tracking
        self.processing_times = []
        self.max_processing_time_history = 100
        self.frame_counter = 0
        self.batch_processing = False

        # Start background processing thread
        self.processing_thread = threading.Thread(target=self._process_frames_worker, daemon=True)
        self.processing_thread.start()

        logger.info(f"Optimized FaceRecognizer initialized in {'registration' if registration_mode else 'recognition'} mode")
        logger.info(f"Parallel workers: {max_workers}, Batch size: {batch_size}")

    def _init_gpu_settings(self):
        """Initialize GPU settings and find optimal batch size."""
        if self.device.type == 'cuda':
            torch.cuda.empty_cache()
            # Set GPU memory growth to prevent OOM
            torch.backends.cudnn.benchmark = True
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            
            # Find optimal batch size
            self.batch_size = self._find_optimal_batch_size()
            logger.info(f"Optimal batch size: {self.batch_size}")

    def _find_optimal_batch_size(self, frame_size=(640, 640), max_batch_size=16):
        """Dynamically find the optimal batch size for GPU processing."""
        if not torch.cuda.is_available() or not hasattr(self, 'insightface_app'):
            return 1
            
        # Start with a small batch size
        batch_size = 1
        torch.cuda.empty_cache()
        
        try:
            while batch_size <= max_batch_size:
                # Test if we can allocate this batch size
                test_tensor = torch.randn((batch_size, 3, *frame_size), 
                                        device=self.device)
                # Try a forward pass
                with torch.no_grad():
                    # Test with different model access patterns
                    if hasattr(self.insightface_app, 'get'):
                        # For newer versions of InsightFace
                        img_np = (test_tensor[0].permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
                        _ = self.insightface_app.get(img_np)
                    elif hasattr(self.insightface_app, 'models') and 'detection' in self.insightface_app.models:
                        # For versions with models dictionary
                        _ = self.insightface_app.models['detection'](test_tensor)
                    elif hasattr(self.insightface_app, 'det_model'):
                        # For some versions that use det_model
                        _ = self.insightface_app.det_model.detect(test_tensor)
                    elif hasattr(self.insightface_app, 'model'):
                        # Fallback to old model attribute if it exists
                        if hasattr(self.insightface_app.model, 'detect'):
                            _ = self.insightface_app.model.detect(test_tensor)
                        else:
                            _ = self.insightface_app.model(test_tensor)
                    else:
                        logger.warning("Could not determine how to access detection model")
                        return 1
                batch_size *= 2
        except RuntimeError as e:
            if 'out of memory' in str(e).lower():
                batch_size = max(1, batch_size // 2)
                torch.cuda.empty_cache()
        except Exception as e:
            logger.warning(f"Error in batch size detection: {e}")
            return 1  # Fallback to batch size 1 on any error
        
        return batch_size

    def _fallback_sequential_processing(self, frame_batch: List[np.ndarray], frame_indices: List[int]) -> List[Tuple]:
        """Fallback to sequential processing when batch processing fails."""
        logger.warning("Falling back to sequential processing")
        results = []
        for idx, frame in zip(frame_indices, frame_batch):
            try:
                # Process single frame
                processed_frame, face_data = self.process_frame(frame)
                results.append((idx, processed_frame, face_data, 0.0))
            except Exception as e:
                logger.error(f"Error in sequential fallback for frame {idx}: {e}")
                results.append((idx, frame, [], 0.0))
        return results

    def _process_frames_worker(self):
        """Background worker for parallel frame processing."""
        while True:
            try:
                batch_data = self.processing_queue.get(timeout=1.0)
                if batch_data is None:  # Shutdown signal
                    break
                
                frame_batch, frame_indices = batch_data
                results = self._process_frame_batch(frame_batch, frame_indices)
                self.results_queue.put(results)
                
            except Empty:
                continue
            except Exception as e:
                logger.error(f"Error in processing worker: {e}")

    def _process_frame_batch(self, frame_batch: List[np.ndarray], frame_indices: List[int]) -> List[Tuple]:
        """Process a batch of frames in parallel with GPU acceleration."""
        results = []
        
        # Process using GPU batch processing if available
        if self.device.type == 'cuda':
            try:
                # Preprocess frames and ensure consistent size
                target_size = (640, 640)  # Standard size for detection
                batch_tensors = []
                
                for frame in frame_batch:
                    # Resize frame to target size while maintaining aspect ratio
                    h, w = frame.shape[:2]
                    scale = min(target_size[0] / h, target_size[1] / w)
                    new_h, new_w = int(h * scale), int(w * scale)
                    
                    # Resize with padding to maintain aspect ratio
                    resized = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
                    
                    # Create a black canvas of target size
                    padded = np.zeros((target_size[0], target_size[1], 3), dtype=np.uint8)
                    
                    # Calculate padding (centered)
                    top = (target_size[0] - new_h) // 2
                    left = (target_size[1] - new_w) // 2
                    
                    # Place the resized image in the center of the canvas
                    padded[top:top+new_h, left:left+new_w] = resized
                    
                    # Convert to tensor and normalize
                    img_tensor = torch.from_numpy(padded).permute(2, 0, 1).float() / 255.0
                    batch_tensors.append(img_tensor)
                
                # Stack into a single batch tensor
                try:
                    batch_tensor = torch.stack(batch_tensors).to(self.device)
                except RuntimeError as e:
                    logger.error(f"Error stacking tensors: {e}")
                    return self._fallback_sequential_processing(frame_batch, frame_indices)
                
                # Process batch through model
                with torch.no_grad():
                    # Enable TF32 for faster processing on Ampere GPUs
                    with torch.amp.autocast(device_type='cuda', enabled=True):
                        # Get detections using the detection model
                        if hasattr(self.insightface_app, 'get'):
                            # For newer versions of InsightFace
                            detections = []
                            for img_tensor in batch_tensor:
                                # Convert tensor back to numpy and BGR for InsightFace
                                img_np = (img_tensor.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
                                faces = self.insightface_app.get(img_np)
                                detections.append([face.bbox for face in faces])
                        elif hasattr(self.insightface_app, 'models') and 'detection' in self.insightface_app.models:
                            # For versions with models dictionary
                            detections = self.insightface_app.models['detection'](batch_tensor)
                        elif hasattr(self.insightface_app, 'det_model'):
                            # For some versions that use det_model
                            detections = self.insightface_app.det_model.detect(batch_tensor)
                        else:
                            raise AttributeError("Could not find detection model in InsightFace app")
                
                # Process detections in parallel
                with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                    future_to_idx = {}
                    for idx, (frame, dets) in enumerate(zip(frame_batch, detections)):
                        # Convert detections to the original frame coordinates if needed
                        if hasattr(dets, 'bbox') and hasattr(dets.bbox, 'cpu'):
                            dets = dets.bbox.cpu().numpy()
                        elif isinstance(dets, (list, tuple)) and len(dets) > 0 and hasattr(dets[0], 'bbox'):
                            dets = [d.bbox for d in dets]
                        future = executor.submit(self._process_detection_batch, frame, dets, idx)
                        future_to_idx[future] = idx
                    
                    for future in as_completed(future_to_idx):
                        idx = future_to_idx[future]
                        try:
                            result = future.result()
                            results.append((idx, frame_batch[idx], result, 0.0))
                        except Exception as e:
                            logger.error(f"Error processing detection batch {idx}: {e}")
                            results.append((idx, frame_batch[idx], [], 0.0))
                
                return sorted(results, key=lambda x: x[0])
                
            except Exception as e:
                logger.error(f"GPU batch processing failed: {e}")
                # Fallback to CPU processing
                return self._fallback_sequential_processing(frame_batch, frame_indices)
        
        # Fallback to CPU processing if GPU is not available
        return self._process_frames_cpu(frame_batch, frame_indices)
    
    def _process_detection_batch(self, frame: np.ndarray, detections: List[Dict], idx: int) -> List[Dict]:
        """Process detections for a single frame."""
        face_data = []
        
        if not detections:
            return face_data
            
        for det in detections:
            try:
                x1, y1, x2, y2 = det['facial_area']
                face_embedding = det.get('embedding')
                
                if face_embedding is None:
                    face_region = frame[max(0, y1):min(frame.shape[0], y2), 
                                      max(0, x1):min(frame.shape[1], x2)]
                    if face_region.size > 0:
                        face_embedding = self.face_encoder.encode_face_from_crop(face_region)
                
                if face_embedding is not None and not self.registration_mode:
                    recognized_face = self._recognize_face(face_embedding, None, (x1, y1, x2, y2))
                    face_data.append(recognized_face)
                else:
                    face_data.append({
                        'name': "Unknown",
                        'user_id': None,
                        'confidence': 0.0,
                        'distance': 1.0,
                        'registered': False,
                        'face_location': (x1, y1, x2, y2)
                    })
            except Exception as e:
                logger.error(f"Error processing detection: {e}")
                continue
                
        return face_data
        
    def _process_frames_cpu(self, frame_batch: List[np.ndarray], frame_indices: List[int]) -> List[Tuple]:
        """Process frames using CPU with parallel processing."""
        results = []
        
        def process_single_frame(frame, idx):
            try:
                start_time = time.time()
                # Detect faces using InsightFace
                detected_faces = self._detect_faces_insightface(frame)
                face_data = []
                
                if detected_faces:
                    for face_info in detected_faces:
                        x1, y1, x2, y2 = face_info['facial_area']
                        face_embedding = face_info.get('embedding')
                        
                        if face_embedding is not None:
                            face_encoding = face_embedding
                        else:
                            face_region = frame[max(0, y1):min(frame.shape[0], y2), 
                                              max(0, x1):min(frame.shape[1], x2)]
                            if face_region.size > 0:
                                face_encoding = self.face_encoder.encode_face_from_crop(face_region)
                            else:
                                face_encoding = None
                        
                        if face_encoding is not None and not self.registration_mode:
                            recognized_face = self._recognize_face(face_encoding, None, (x1, y1, x2, y2))
                            face_data.append(recognized_face)
                        else:
                            face_data.append({
                                'name': "Unknown",
                                'user_id': None,
                                'confidence': 0.0,
                                'distance': 1.0,
                                'registered': False,
                                'face_location': (x1, y1, x2, y2)
                            })
                
                processing_time = time.time() - start_time
                return idx, frame, face_data, processing_time
                
            except Exception as e:
                logger.error(f"Error processing frame in batch: {e}")
                return idx, frame, [], 0.0
        
        # Process frames in parallel
        with ThreadPoolExecutor(max_workers=min(self.max_workers, len(frame_batch))) as executor:
            future_to_frame = {
                executor.submit(process_single_frame, frame, idx): idx 
                for idx, frame in enumerate(frame_batch)
            }
            
            for future in future_to_frame:
                try:
                    result = future.result(timeout=5.0)  # 5 second timeout
                    results.append(result)
                except Exception as e:
                    idx = future_to_frame[future]
                    logger.error(f"Frame {idx} processing failed: {e}")
                    results.append((idx, frame_batch[idx], [], 0.0))
        
        return sorted(results, key=lambda x: x[0])  # Sort by original index

    def load_known_faces(self, users: List[Dict]) -> bool:
        """Thread-safe loading of known faces."""
        with self.known_face_lock:
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
                    if norm < 1e-10:
                        logger.warning(f"Skipping zero-vector encoding for user {user_id}")
                        continue

                    encoding = encoding / norm

                    self.known_face_encodings.append(encoding)
                    self.known_face_names.append(user_name)
                    self.known_face_ids.append(user_id)
                    valid_count += 1

                except Exception as e:
                    logger.error(f"Error loading user {user.get('id', 'unknown')}: {e}")

            logger.info(f"Successfully loaded {valid_count} out of {len(users)} users with valid face encodings")
            return valid_count > 0

    def _detect_faces_insightface(self, frame: np.ndarray) -> List[Dict[str, Any]]:
        """Optimized face detection with GPU acceleration."""
        try:
            bgr_frame = frame.copy()
            faces = self.insightface_app.get(bgr_frame)
            
            face_data = []
            for face in faces:
                try:
                    confidence = face.det_score
                    
                    if confidence < 0.5:  # Lowered threshold for better detection
                        continue
                    
                    bbox = face.bbox.astype(int)
                    x1, y1, x2, y2 = bbox
                    
                    if x1 >= x2 or y1 >= y2:
                        continue
                    
                    if x1 < 0 or y1 < 0 or x2 > frame.shape[1] or y2 > frame.shape[0]:
                        continue

                    landmarks = {}
                    if face.kps is not None:
                        landmark_points = face.kps.astype(int)
                        landmarks = {
                            'left_eye': tuple(landmark_points[0]),
                            'right_eye': tuple(landmark_points[1]),
                            'nose': tuple(landmark_points[2]),
                            'mouth_left': tuple(landmark_points[3]),
                            'mouth_right': tuple(landmark_points[4])
                        }

                    face_data.append({
                        'facial_area': (x1, y1, x2, y2),
                        'confidence': confidence,
                        'landmarks': landmarks,
                        'embedding': face.embedding
                    })
                    
                    logger.debug(f"Detected face - Confidence: {confidence:.4f}, BBox: [{x1}, {y1}, {x2}, {y2}]")
                    
                except Exception as e:
                    logger.warning(f"Error processing face: {e}")
                    continue
                
            return face_data
        
        except Exception as e:
            logger.error(f"Error in face detection: {e}")
            return []

    def process_frame(self, frame: np.ndarray, db_handler = None) -> Tuple[np.ndarray, List[Dict]]:
        """
        Process single frame with optional parallel processing.
        
        Args:
            frame: BGR image frame from OpenCV
            db_handler: Database handler for registration/logging

        Returns:
            Tuple of (processed_frame, face_data_list)
        """
        start_time = time.time()
        
        if self.registration_mode:
            # Registration mode uses sequential processing for better UX
            return self._process_frame_sequential(frame, db_handler)
        else:
            # Recognition mode can use parallel processing
            return self._process_frame_parallel(frame, db_handler)

    def _process_frame_sequential(self, frame: np.ndarray, db_handler = None) -> Tuple[np.ndarray, List[Dict]]:
        """Sequential processing for registration mode."""
        processed_frame = frame.copy()
        face_data = []

        try:
            detected_faces = self._detect_faces_insightface(frame)
            if not detected_faces:
                cv2.putText(processed_frame, "No faces detected", (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                return processed_frame, face_data

            for face_info in detected_faces:
                x1, y1, x2, y2 = face_info['facial_area']
                face_embedding = face_info.get('embedding')

                if face_embedding is not None:
                    face_encoding = face_embedding
                else:
                    face_region = frame[max(0, y1):min(frame.shape[0], y2), 
                                      max(0, x1):min(frame.shape[1], x2)]
                    if face_region.size > 0:
                        face_encoding = self.face_encoder.encode_face_from_crop(face_region)
                    else:
                        face_encoding = None

                recognized_face = None
                if self.registration_mode:
                    recognized_face = self._register_face(face_region, face_encoding, db_handler, (x1, y1, x2, y2))
                else:
                    recognized_face = self._recognize_face(face_encoding, db_handler, (x1, y1, x2, y2))

                self._draw_face_annotation(processed_frame, face_info, recognized_face, self.registration_mode)

                if recognized_face:
                    face_data.append(recognized_face)

            # Performance tracking
            processing_time = time.time() - start_time
            self._update_performance_stats(processing_time)

            # Display performance info
            self._display_performance_info(processed_frame)

            return processed_frame, face_data

        except Exception as e:
            logger.error(f"Error processing frame sequentially: {e}")
            cv2.putText(processed_frame, "Processing error", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            return processed_frame, []

    def _process_frame_parallel(self, frame: np.ndarray, db_handler = None) -> Tuple[np.ndarray, List[Dict]]:
        """Parallel processing for recognition mode."""
        processed_frame = frame.copy()
        
        # For single frame, use immediate processing
        try:
            detected_faces = self._detect_faces_insightface(frame)
            face_data = []

            if not detected_faces:
                cv2.putText(processed_frame, "No faces detected", (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                return processed_frame, face_data

            for face_info in detected_faces:
                x1, y1, x2, y2 = face_info['facial_area']
                face_embedding = face_info.get('embedding')

                if face_embedding is not None:
                    face_encoding = face_embedding
                else:
                    face_region = frame[max(0, y1):min(frame.shape[0], y2), 
                                      max(0, x1):min(frame.shape[1], x2)]
                    if face_region.size > 0:
                        face_encoding = self.face_encoder.encode_face_from_crop(face_region)
                    else:
                        face_encoding = None

                recognized_face = self._recognize_face(face_encoding, db_handler, (x1, y1, x2, y2))
                self._draw_face_annotation(processed_frame, face_info, recognized_face, self.registration_mode)

                if recognized_face:
                    face_data.append(recognized_face)

            # Performance tracking
            processing_time = time.time() - start_time
            self._update_performance_stats(processing_time)
            self._display_performance_info(processed_frame)

            return processed_frame, face_data

        except Exception as e:
            logger.error(f"Error in parallel frame processing: {e}")
            cv2.putText(processed_frame, "Processing error", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            return processed_frame, []

    def process_frames_batch(self, frames: List[np.ndarray]) -> List[Tuple[np.ndarray, List[Dict]]]:
        """
        Process multiple frames in parallel batches.
        
        Args:
            frames: List of frames to process
            
        Returns:
            List of (processed_frame, face_data) tuples
        """
        if not frames:
            return []
            
        results = []
        batch_size = min(self.batch_size, len(frames))
        
        # Process in batches
        for i in range(0, len(frames), batch_size):
            batch = frames[i:i + batch_size]
            batch_indices = list(range(i, i + len(batch)))
            
            # Submit batch for processing
            self.processing_queue.put((batch, batch_indices))
            
            # Wait for results (with timeout)
            try:
                batch_results = self.results_queue.get(timeout=10.0)
                for idx, processed_frame, face_data, processing_time in batch_results:
                    self._update_performance_stats(processing_time)
                    results.append((processed_frame, face_data))
            except Empty:
                logger.error("Timeout waiting for batch processing results")
                # Fallback to sequential processing for this batch
                for frame in batch:
                    processed_frame, face_data = self.process_frame(frame)
                    results.append((processed_frame, face_data))
        
        return results

    def _update_performance_stats(self, processing_time: float):
        """Update performance statistics in a thread-safe manner."""
        self.processing_times.append(processing_time)
        if len(self.processing_times) > self.max_processing_time_history:
            self.processing_times.pop(0)

    def _display_performance_info(self, frame: np.ndarray):
        """Display performance information on frame."""
        avg_time = np.mean(self.processing_times) if self.processing_times else 0
        fps = 1.0 / avg_time if avg_time > 0 else 0
        
        cv2.putText(frame, f"FPS: {fps:.1f}", (10, frame.shape[0] - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Show parallel processing info
        if self.max_workers > 1:
            cv2.putText(frame, f"Workers: {self.max_workers}", (10, frame.shape[0] - 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    def _draw_face_annotation(self, frame: np.ndarray, face_info: Dict, 
                            recognized_face: Optional[Dict], registration_mode: bool) -> None:
        """Optimized face annotation drawing."""
        x1, y1, x2, y2 = face_info['facial_area']
        detection_confidence = face_info['confidence']

        if registration_mode:
            label = "Press 's' to register"
            color = (0, 255, 255)  # Yellow
        else:
            if recognized_face and recognized_face.get('name') != "Unknown":
                name = recognized_face['name']
                confidence = recognized_face.get('confidence', 0)
                label = f"{name} ({confidence:.2f})"
                color = (0, 255, 0)  # Green
            else:
                label = "Unknown"
                color = (0, 0, 255)  # Red

        # Draw bounding box
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        
        # Draw detection confidence
        cv2.putText(frame, f"Det: {detection_confidence:.2f}", 
                   (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Draw name label
        label_y = y1 - 30 if y1 > 50 else y2 + 25
        cv2.putText(frame, label, (x1, label_y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        # Draw landmarks
        landmarks = face_info.get('landmarks', {})
        for landmark_name, (lx, ly) in landmarks.items():
            cv2.circle(frame, (int(lx), int(ly)), 2, (255, 0, 0), -1)

    def _register_face(self, face_region: np.ndarray, face_encoding: np.ndarray, 
                      db_handler, face_location: Tuple) -> Optional[Dict]:
        """Face registration logic."""
        try:
            key = cv2.waitKey(1) & 0xFF
            current_time = time.time()
            
            if key == ord(' ') and not self.registration_pending and \
               (current_time - self.last_registration_time) > self.registration_cooldown:
                self.registration_pending = True
                self.last_registration_time = current_time
                logger.info("Registration triggered by space key")

            if not self.registration_pending:
                return None
                
            if face_region is None or face_region.size == 0 or face_encoding is None:
                logger.error("No valid face detected for registration")
                print("No face detected in camera view. Please position your face in the frame.")
                self.registration_pending = False
                return None

            name = input("Enter name for new face (or press Enter to skip): ").strip()
            if not name:
                logger.info("Registration skipped - no name provided")
                self.registration_pending = False
                return None

            success, encoded_image = cv2.imencode('.jpg', face_region)
            if not success:
                logger.error("Failed to encode face image for registration")
                self.registration_pending = False
                return None

            image_data = encoded_image.tobytes()
            result = db_handler.add_user(name=name, image_data=image_data, is_face_crop=True)
            
            if result and result.get('status') == 'success':
                user_id = result.get('id')
                logger.info(f"Registered new user: {name} with ID: {user_id}")

                # Thread-safe update
                with self.known_face_lock:
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
        """Thread-safe face recognition."""
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
            with self.known_face_lock:
                best_match_name, confidence, verified = self.face_encoder.find_best_match(
                    face_encoding, self.known_face_encodings, self.known_face_names)

            if best_match_name and verified and confidence > 0.5:
                match_idx = self.known_face_names.index(best_match_name)
                user_id = self.known_face_ids[match_idx]

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

    def recognize_faces(self, face_encodings: List[Union[np.ndarray, List]]) -> List[Dict]:
        """Batch face recognition with thread safety."""
        recognized_faces = []

        with self.known_face_lock:
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
        """Get performance statistics."""
        if not self.processing_times:
            return {'avg_processing_time': 0, 'fps': 0}

        avg_time = np.mean(self.processing_times)
        fps = 1.0 / avg_time if avg_time > 0 else 0
        
        return {
            'avg_processing_time': avg_time,
            'fps': fps,
            'samples': len(self.processing_times),
            'parallel_workers': self.max_workers
        }

    def set_registration_mode(self, enabled: bool):
        """Set registration mode."""
        self.registration_mode = enabled
        logger.info(f"Registration mode {'enabled' if enabled else 'disabled'}")

    def clear_known_faces(self):
        """Clear all known face data from memory."""
        with self.known_face_lock:
            self.known_face_encodings.clear()
            self.known_face_names.clear()
            self.known_face_ids.clear()
        logger.info("Cleared all known face data from memory")

    def shutdown(self):
        """Clean shutdown of parallel workers."""
        self.processing_queue.put(None)  # Shutdown signal
        self.parallel_executor.shutdown(wait=True)
        logger.info("FaceRecognizer shutdown complete")
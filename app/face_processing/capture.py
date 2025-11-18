import cv2
import time
from datetime import datetime
import os
import logging
import numpy as np
import torch
import torchvision.transforms as transforms
from typing import Tuple, List, Optional, Dict, Any
import threading
from queue import Queue

import insightface
from insightface.app import FaceAnalysis
from insightface.data import get_image as ins_get_image

logger = logging.getLogger(__name__)   

class FaceDetector:
    """Dedicated face detection class with caching and optimization using InsightFace SCRFD"""
    
    def __init__(self, device: torch.device):
        self.device = device
        self.detection_cache = {}
        self.cache_size = 50
        self.min_confidence = 0.7
        
        # Initialize InsightFace with SCRFD detector
        self.app = FaceAnalysis(
            name='buffalo_l',  # This uses SCRFD as detector
            providers=['CUDAExecutionProvider'] if device.type == 'cuda' else ['CPUExecutionProvider']
        )
        self.app.prepare(ctx_id=0 if device.type == 'cuda' else -1, det_size=(640, 640))
        
        logger.info(f"FaceDetector using device: {self.device} with InsightFace SCRFD")
        
    def detect_faces(self, frame: np.ndarray) -> Optional[Dict[str, Any]]:
        """Detect faces in frame with caching using InsightFace SCRFD"""
        try:
            # Simple frame hash for caching
            frame_hash = hash(frame.tobytes())
            
            if frame_hash in self.detection_cache:
                return self.detection_cache[frame_hash]
            
            # InsightFace expects BGR format by default
            bgr_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR) if len(frame.shape) == 3 and frame.shape[2] == 3 else frame
            
            # Perform face detection with InsightFace
            faces = self.app.get(bgr_frame)
            
            if not faces:
                return None
            
            # Convert InsightFace results to our format
            valid_faces = []
            for i, face in enumerate(faces):
                if face.det_score >= self.min_confidence:
                    bbox = face.bbox.astype(int)
                    valid_faces.append({
                        'bbox': [bbox[0], bbox[1], bbox[2], bbox[3]],  # [x1, y1, x2, y2]
                        'confidence': face.det_score,
                        'landmarks': {
                            'left_eye': face.kps[0].tolist() if face.kps is not None else [],
                            'right_eye': face.kps[1].tolist() if face.kps is not None else [],
                            'nose': face.kps[2].tolist() if face.kps is not None else [],
                            'mouth_left': face.kps[3].tolist() if face.kps is not None else [],
                            'mouth_right': face.kps[4].tolist() if face.kps is not None else []
                        },
                        'embedding': face.embedding if hasattr(face, 'embedding') else None,
                        'face_id': i
                    })
            
            result = {'faces': valid_faces} if valid_faces else None
            
            # Cache result
            if len(self.detection_cache) >= self.cache_size:
                self.detection_cache.pop(next(iter(self.detection_cache)))
            self.detection_cache[frame_hash] = result
            
            return result
            
        except Exception as e:
            logger.error(f"Face detection failed: {e}")
            return None


class VideoCapture:
    def __init__(self, source: str = "webcam", width: int = 640, height: int = 480, 
                 enable_depth: bool = True, enable_face: bool = True):
        """
        Enhanced video capture with optional features
        
        Args:
            source: "webcam" or "phone"
            width: frame width
            height: frame height  
            enable_depth: enable depth estimation
            enable_face: enable face detection
        """
        self.source = source
        self.width = width
        self.height = height
        self.enable_depth = enable_depth
        self.enable_face = enable_face
        self.cap = None
        self.last_capture_time = 0
        self.frame_count = 0
        self.fps = 0
        self.fps_time = time.time()

        # Initialize device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {self.device}")

        # Initialize models based on enabled features
        #if enable_depth:
        #   self.depth_estimator = DepthEstimator()
        
        if enable_face:
            self.face_detector = FaceDetector(self.device)
            # InsightFace already provides embeddings, so we don't need separate embedder
            # But we keep the transform for compatibility if needed
            self.transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize((112, 112)),  # InsightFace uses 112x112 for recognition
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
            ])

    def initialize(self, phone_url: str = None, timeout: int = 10) -> bool:
        """Initialize video capture source with timeout"""
        if self.cap is not None:
            self.cap.release()

        start_time = time.time()
        
        while time.time() - start_time < timeout:
            try:
                if self.source == "phone":
                    if not phone_url:
                        raise ValueError("❌ Phone source selected but no URL provided!")
                    self.cap = cv2.VideoCapture(phone_url)
                else:
                    self.cap = cv2.VideoCapture(0)  # default webcam

                if self.cap.isOpened():
                    # Configure camera properties
                    self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
                    self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
                    self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                    self.cap.set(cv2.CAP_PROP_FPS, 30)
                    self.cap.set(cv2.CAP_PROP_AUTOFOCUS, 1)
                    self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
                    
                    # Test frame capture
                    ret, test_frame = self.cap.read()
                    if ret and test_frame is not None:
                        logger.info(
                            f"✅ Camera initialized (source={self.source}, "
                            f"resolution={self.width}x{self.height})"
                        )
                        return True
                
                time.sleep(1)
                
            except Exception as e:
                logger.error(f"Camera initialization attempt failed: {e}")
                time.sleep(1)
        
        logger.error(f"❌ Camera initialization timeout after {timeout} seconds")
        return False

    def calculate_fps(self):
        """Calculate current FPS"""
        self.frame_count += 1
        if time.time() - self.fps_time >= 1.0:
            self.fps = self.frame_count
            self.frame_count = 0
            self.fps_time = time.time()

    def capture_frame(self, interval: float = 1.0, max_retries: int = 3) -> Optional[np.ndarray]:
        """Capture frame at intervals with FPS calculation"""
        current_time = time.time()
        
        # Maintain capture interval
        if current_time - self.last_capture_time < interval:
            # Skip frames efficiently
            self.cap.grab()
            return None

        for attempt in range(max_retries):
            ret, frame = self.cap.read()
            if ret and frame is not None and frame.size > 0:
                self.last_capture_time = current_time
                self.calculate_fps()
                
                # Ensure correct dimensions
                if frame.shape[1] != self.width or frame.shape[0] != self.height:
                    frame = cv2.resize(frame, (self.width, self.height))
                
                return frame
            
            if attempt < max_retries - 1:
                logger.warning(f"Frame capture failed, retry {attempt + 1}/{max_retries}")
                time.sleep(0.1)
                self.initialize()
        
        logger.error("❌ All attempts to read frame failed")
        return None

    def detect_and_embed(self, frame: np.ndarray) -> Tuple[Optional[List], Optional[List]]:
        """
        Detect faces and compute embeddings with enhanced error handling using InsightFace
        """
        if not self.enable_face:
            return None, None
            
        try:
            # InsightFace works directly with BGR frames, so we convert if needed
            if len(frame.shape) == 3 and frame.shape[2] == 3:
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            else:
                rgb_frame = frame
                
            detection_result = self.face_detector.detect_faces(rgb_frame)
            if not detection_result:
                return None, None

            embeddings = []
            bboxes = []

            for face_data in detection_result['faces']:
                x1, y1, x2, y2 = face_data['bbox']
                
                # Add margin to face crop (optional)
                margin = 0.2
                h, w = y2 - y1, x2 - x1
                x1 = max(0, int(x1 - margin * w))
                y1 = max(0, int(y1 - margin * h))
                x2 = min(frame.shape[1], int(x2 + margin * w))
                y2 = min(frame.shape[0], int(y2 + margin * h))
                
                # Get embedding directly from InsightFace result
                if face_data.get('embedding') is not None:
                    embedding = face_data['embedding']
                    # Normalize embedding
                    embedding = embedding / np.linalg.norm(embedding)
                    embeddings.append(embedding)
                    bboxes.append((x1, y1, x2, y2, face_data['confidence']))
                else:
                    # Fallback: extract face and compute embedding if not available
                    try:
                        cropped_face = rgb_frame[y1:y2, x1:x2]
                        if cropped_face.size == 0:
                            continue
                            
                        face_tensor = self.transform(cropped_face).unsqueeze(0).to(self.device)
                        # Note: This would require a separate InsightFace model for embedding only
                        # For now, we rely on the embedding from detection
                        logger.warning("Embedding not available in detection result")
                        continue
                        
                    except Exception as e:
                        logger.warning(f"Face embedding fallback failed: {e}")
                        continue

            return embeddings, bboxes if embeddings else (None, None)
            
        except Exception as e:
            logger.error(f"Face detection/embedding failed: {e}")
            return None, None

    def estimate_depth(self, frame: np.ndarray) -> Optional[np.ndarray]:
        """Generate depth map from frame"""
        if not self.enable_depth:
            return None
        # Placeholder for depth estimation
        # return self.depth_estimator.predict_depth(frame)
        return None

    def draw_detections(self, frame: np.ndarray, bboxes: List, fps: int = None) -> np.ndarray:
        """Draw face detections and info on frame"""
        display_frame = frame.copy()
        
        # Draw FPS
        if fps is not None:
            cv2.putText(display_frame, f"FPS: {fps}", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Draw face bounding boxes
        if bboxes:
            for i, (x1, y1, x2, y2, confidence) in enumerate(bboxes):
                # Draw bounding box
                cv2.rectangle(display_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                
                # Draw confidence label
                label = f"Face {i+1}: {confidence:.2f}"
                cv2.putText(display_frame, label, (x1, y1 - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        
        return display_frame

    def save_frame(self, frame: np.ndarray, directory: str = None, 
                   prefix: str = "frame", create_dir: bool = True) -> Optional[str]:
        """Save frame to disk with enhanced error handling"""
        if directory is None:
            return None
            
        try:
            if create_dir:
                os.makedirs(directory, exist_ok=True)
            elif not os.path.exists(directory):
                logger.error(f"Directory {directory} does not exist")
                return None

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            filename = f"{prefix}_{timestamp}.jpg"
            filepath = os.path.join(directory, filename)
            
            # Save with optimized JPEG quality
            success = cv2.imwrite(filepath, frame, [cv2.IMWRITE_JPEG_QUALITY, 90])
            
            if success:
                logger.debug(f"📸 Frame saved: {filepath}")
                return filepath
            else:
                logger.error(f"Failed to write image: {filepath}")
                return None
                
        except Exception as e:
            logger.error(f"Failed to save frame: {e}")
            return None

    def get_camera_info(self) -> Dict[str, Any]:
        """Get camera information and capabilities"""
        if self.cap is None:
            return {}
            
        info = {
            'width': self.cap.get(cv2.CAP_PROP_FRAME_WIDTH),
            'height': self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT),
            'fps': self.cap.get(cv2.CAP_PROP_FPS),
            'backend': self.cap.getBackendName(),
            'source': self.source
        }
        return info

    def release(self):
        """Release resources"""
        if self.cap is not None:
            self.cap.release()
            logger.info("📷 Camera released successfully")
        
        # Clear GPU cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

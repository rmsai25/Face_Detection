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

from retinaface import RetinaFace
from facenet_pytorch import InceptionResnetV1

logger = logging.getLogger(__name__)   

class DepthEstimator:
    def __init__(self, model_type: str = "DPT_Hybrid"):
        """Initialize depth estimation model"""
        logger.info(f"Initializing depth estimator with {model_type}")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # GPU info logging
        if torch.cuda.is_available():
            logger.info(f"GPU: {torch.cuda.device_count()}x {torch.cuda.get_device_name()}")
            logger.info(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f}GB")
        
        try:
            # Load model with error handling
            self.model = torch.hub.load("intel-isl/MiDaS", model_type, trust_repo=True)
            self.model.to(self.device).eval()
            
            # Load transforms
            midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms", trust_repo=True)
            self.transform = (
                midas_transforms.dpt_transform
                if "DPT" in model_type
                else midas_transforms.small_transform
            )
            
            logger.info("✅ Depth estimator initialized successfully")
        except Exception as e:
            logger.error(f"❌ Failed to initialize depth estimator: {e}")
            raise

    def predict_depth(self, frame: np.ndarray) -> np.ndarray:
        """Predict depth map from frame"""
        try:
            # Convert BGR to RGB for MiDaS
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Apply transform and predict
            input_batch = self.transform(rgb_frame).to(self.device)
            
            with torch.no_grad():
                prediction = self.model(input_batch)
                prediction = torch.nn.functional.interpolate(
                    prediction.unsqueeze(1),
                    size=frame.shape[:2],
                    mode="bicubic",
                    align_corners=False,
                ).squeeze()
            
            # Convert to 8-bit depth map
            depth_map = prediction.cpu().numpy()
            depth_map = cv2.normalize(depth_map, None, 0, 255, cv2.NORM_MINMAX)
            return depth_map.astype(np.uint8)
            
        except Exception as e:
            logger.error(f"Depth prediction failed: {e}")
            return np.zeros(frame.shape[:2], dtype=np.uint8)


class FaceDetector:
    """Dedicated face detection class with caching and optimization"""
    
    def __init__(self, device: torch.device):
        self.device = device
        self.detection_cache = {}
        self.cache_size = 50
        self.min_confidence = 0.7
        
    def detect_faces(self, frame: np.ndarray) -> Optional[Dict[str, Any]]:
        """Detect faces in frame with caching"""
        try:
            # Simple frame hash for caching
            frame_hash = hash(frame.tobytes())
            
            if frame_hash in self.detection_cache:
                return self.detection_cache[frame_hash]
            
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            detections = RetinaFace.detect_faces(rgb_frame)
            
            if not detections or not isinstance(detections, dict):
                return None
            
            # Filter by confidence and extract face data
            valid_faces = []
            for face_id, face_info in detections.items():
                if face_info['score'] >= self.min_confidence:
                    valid_faces.append({
                        'bbox': face_info['facial_area'],
                        'confidence': face_info['score'],
                        'landmarks': face_info.get('landmarks', {}),
                        'face_id': face_id
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
        if enable_depth:
            self.depth_estimator = DepthEstimator()
        
        if enable_face:
            self.face_detector = FaceDetector(self.device)
            self.embedder = InceptionResnetV1(pretrained="vggface2").eval().to(self.device)
            
            # Enhanced face preprocessing
            self.transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize((160, 160)),
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
        Detect faces and compute embeddings with enhanced error handling
        """
        if not self.enable_face:
            return None, None
            
        try:
            detection_result = self.face_detector.detect_faces(frame)
            if not detection_result:
                return None, None

            embeddings = []
            bboxes = []
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            for face_data in detection_result['faces']:
                x1, y1, x2, y2 = face_data['bbox']
                
                # Add margin to face crop
                margin = 0.2
                h, w = y2 - y1, x2 - x1
                x1 = max(0, int(x1 - margin * w))
                y1 = max(0, int(y1 - margin * h))
                x2 = min(frame.shape[1], int(x2 + margin * w))
                y2 = min(frame.shape[0], int(y2 + margin * h))
                
                cropped_face = rgb_frame[y1:y2, x1:x2]
                
                if cropped_face.size == 0:
                    continue

                try:
                    # Process face for embedding
                    face_tensor = self.transform(cropped_face).unsqueeze(0).to(self.device)
                    
                    with torch.no_grad():
                        embedding = self.embedder(face_tensor).cpu().numpy()[0]
                    
                    # Normalize embedding
                    embedding = embedding / np.linalg.norm(embedding)
                    embeddings.append(embedding)
                    bboxes.append((x1, y1, x2, y2, face_data['confidence']))
                    
                except Exception as e:
                    logger.warning(f"Face embedding failed: {e}")
                    continue

            return embeddings, bboxes if embeddings else (None, None)
            
        except Exception as e:
            logger.error(f"Face detection/embedding failed: {e}")
            return None, None

    def estimate_depth(self, frame: np.ndarray) -> Optional[np.ndarray]:
        """Generate depth map from frame"""
        if not self.enable_depth:
            return None
        return self.depth_estimator.predict_depth(frame)

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


def main():
    """Enhanced main function with better error handling"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Video Capture with Face Detection and Depth Estimation')
    parser.add_argument('--source', type=str, default='webcam', choices=['webcam', 'phone'],
                       help='Video source (webcam or phone)')
    parser.add_argument('--width', type=int, default=640, help='Frame width')
    parser.add_argument('--height', type=int, default=480, help='Frame height')
    parser.add_argument('--phone-url', type=str, help='Phone camera URL')
    parser.add_argument('--no-depth', action='store_true', help='Disable depth estimation')
    parser.add_argument('--no-face', action='store_true', help='Disable face detection')
    parser.add_argument('--save-dir', type=str, help='Directory to save frames')
    
    args = parser.parse_args()
    
    # Initialize camera
    cam = VideoCapture(
        source=args.source,
        width=args.width,
        height=args.height,
        enable_depth=not args.no_depth,
        enable_face=not args.no_face
    )
    
    if not cam.initialize(phone_url=args.phone_url):
        logger.error("Failed to initialize camera")
        return
    
    logger.info(f"Camera Info: {cam.get_camera_info()}")
    
    try:
        while True:
            frame = cam.capture_frame(interval=0.033)  # ~30 FPS
            
            if frame is None:
                continue
            
            # Process frame based on enabled features
            embeddings, bboxes = cam.detect_and_embed(frame)
            depth_map = cam.estimate_depth(frame)
            
            # Draw detections
            display_frame = cam.draw_detections(frame, bboxes or [], cam.fps)
            
            # Display results
            cv2.imshow("Live Feed", display_frame)
            
            if depth_map is not None:
                depth_colored = cv2.applyColorMap(depth_map, cv2.COLORMAP_MAGMA)
                cv2.imshow("Depth Map", depth_colored)
            
            # Save frame if directory provided
            if args.save_dir and bboxes:
                cam.save_frame(frame, args.save_dir, "detection")
            
            # Exit on 'q' press
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
    finally:
        cam.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()



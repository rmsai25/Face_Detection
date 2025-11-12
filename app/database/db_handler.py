from sqlalchemy.orm import Session
from sqlalchemy import text
from .models import User, AccessLog, init_db as init_models
from datetime import datetime, timedelta
import logging
import cv2
import numpy as np
from typing import Tuple, Optional, List, Dict, Any
import io
from PIL import Image
import tempfile
import os
import torch
import insightface
from insightface.app import FaceAnalysis

logger = logging.getLogger(__name__)

def cosine_similarity(a, b):
    """Calculate cosine similarity between two vectors"""
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

class InsightFaceEmbedder:
    """InsightFace with SCRFD for face detection and ArcFace for embedding generation"""

    def __init__(self, device: str = None):
        logger.info("Initializing InsightFace with SCRFD and ArcFace")
        try:
            self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
            
            # Initialize InsightFace with SCRFD detector and ArcFace recognizer
            self.insightface_app = FaceAnalysis(
                name='buffalo_l',  # This includes SCRFD detector and ArcFace recognizer
                providers=['CUDAExecutionProvider'] if self.device == 'cuda' else ['CPUExecutionProvider']
            )
            
            # Prepare the model with appropriate context
            ctx_id = 0 if self.device == 'cuda' else -1
            self.insightface_app.prepare(ctx_id=ctx_id, det_size=(640, 640))
            
            logger.info("InsightFace with SCRFD and ArcFace initialized successfully")
        except ImportError as e:
            logger.error(f"Failed to import required libraries: {e}")
            raise
        except Exception as e:
            logger.error(f"Failed to initialize InsightFace: {e}")
            raise

    def detect_faces(self, image: np.ndarray) -> List[Dict[str, Any]]:
        """
        Detect faces using InsightFace SCRFD with enhanced error handling and logging.
        
        Args:
            image: Input image in RGB or BGR format
            
        Returns:
            List of dictionaries containing face detections with bounding boxes, landmarks, and confidence scores
        """
        try:
            if image is None or image.size == 0:
                logger.error("Empty or invalid image provided to detect_faces")
                return []
                
            logger.debug(f"Input image shape: {image.shape}, dtype: {image.dtype}")
            
            # Convert to BGR for InsightFace if needed
            if len(image.shape) != 3 or image.shape[2] != 3:
                logger.error(f"Invalid image shape: {image.shape}. Expected 3-channel image.")
                return []
                
            # Convert to BGR if needed (InsightFace expects BGR)
            if image.dtype != np.uint8:
                logger.warning(f"Converting image from {image.dtype} to uint8")
                if np.max(image) <= 1.0:  # If image is in [0,1] range
                    image = (image * 255).astype(np.uint8)
                else:
                    image = image.astype(np.uint8)
            
            # Ensure we have a BGR image
            try:
                bgr_image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR) if image.shape[-1] == 3 else image
            except cv2.error as e:
                logger.error(f"Error converting image to BGR: {e}")
                return []

            logger.debug(f"Detecting faces in image of size {bgr_image.shape}")
            
            try:
                # Use InsightFace for face detection
                faces = self.insightface_app.get(bgr_image)
                logger.info(f"Found {len(faces)} potential face(s)")
            except Exception as e:
                logger.error(f"Error in InsightFace face detection: {e}", exc_info=True)
                return []

            face_detections = []
            for i, face in enumerate(faces):
                try:
                    confidence = face.det_score
                    logger.debug(f"Face {i+1} - Confidence: {confidence:.4f}")
                    
                    # Lower confidence threshold to catch more potential faces
                    if confidence > 0.3:  # Reduced threshold from 0.7 to 0.3
                        bbox = face.bbox.astype(int)
                        x1, y1, x2, y2 = bbox
                        
                        # Validate bounding box
                        if x1 >= x2 or y1 >= y2 or x1 < 0 or y1 < 0 or x2 > image.shape[1] or y2 > image.shape[0]:
                            logger.warning(f"Invalid bounding box {[x1, y1, x2, y2]} for face {i+1}, skipping")
                            continue
                        
                        # Extract landmarks if available
                        landmarks = {}
                        if hasattr(face, 'kps') and face.kps is not None:
                            try:
                                landmark_points = face.kps.astype(int)
                                landmarks = {
                                    'left_eye': tuple(landmark_points[0]),
                                    'right_eye': tuple(landmark_points[1]),
                                    'nose': tuple(landmark_points[2]),
                                    'mouth_left': tuple(landmark_points[3]),
                                    'mouth_right': tuple(landmark_points[4])
                                }
                                logger.debug(f"Extracted landmarks for face {i+1}")
                            except Exception as e:
                                logger.warning(f"Error extracting landmarks for face {i+1}: {e}")
                        else:
                            logger.debug(f"No landmarks available for face {i+1}")

                        face_detection = {
                            'bbox': [x1, y1, x2, y2],
                            'confidence': float(confidence),
                            'landmarks': landmarks,
                            'embedding': face.embedding if hasattr(face, 'embedding') else None
                        }
                        face_detections.append(face_detection)
                        logger.debug(f"Added face detection {i+1} with confidence {confidence:.4f}")
                    else:
                        logger.debug(f"Face {i+1} below confidence threshold (score: {confidence:.4f})")
                except Exception as e:
                    logger.error(f"Error processing face {i+1}: {e}", exc_info=True)
                    continue

            logger.info(f"Successfully processed {len(face_detections)} face(s)")
            return face_detections
            
        except Exception as e:
            logger.error(f"Unexpected error in detect_faces: {e}", exc_info=True)
            return []

    def extract_embedding(self, image: np.ndarray, face_bbox: List[int] = None) -> Optional[np.ndarray]:
        """Extract face embedding using InsightFace ArcFace"""
        try:
            if face_bbox is not None:
                # Extract face region with margin
                x1, y1, x2, y2 = face_bbox
                margin = 0.2
                w, h = x2 - x1, y2 - y1
                x1 = max(0, int(x1 - margin * w))
                y1 = max(0, int(y1 - margin * h))
                x2 = min(image.shape[1], int(x2 + margin * w))
                y2 = min(image.shape[0], int(y2 + margin * h))
                face_region = image[y1:y2, x1:x2]
                if face_region.size == 0:
                    return None
                
                # Convert to BGR for InsightFace
                bgr_face_region = cv2.cvtColor(face_region, cv2.COLOR_RGB2BGR)
                
                # Get embedding using InsightFace
                faces = self.insightface_app.get(bgr_face_region)
            else:
                # For full images, use InsightFace directly
                bgr_image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                faces = self.insightface_app.get(bgr_image)

            if not faces:
                return None

            # Use the face with highest confidence
            primary_face = max(faces, key=lambda x: x.det_score)
            embedding = primary_face.embedding

            # InsightFace embeddings are already 512D and normalized
            embedding = embedding.astype(np.float32)
            
            # Double-check normalization
            norm = np.linalg.norm(embedding)
            if norm > 0:
                embedding = embedding / norm

            return embedding
            
        except Exception as e:
            logger.error(f"Error extracting face embedding: {e}")
            return None

    def process_image(self, image: np.ndarray) -> Tuple[Optional[np.ndarray], List[Dict[str, Any]]]:
        """Process image to detect faces and extract embeddings using InsightFace"""
        try:
            # Log image properties for debugging
            logger.info(f"Processing image - Shape: {image.shape}, Type: {image.dtype}, "
                      f"Min: {np.min(image)}, Max: {np.max(image)}")
            
            # Check if image is valid
            if image is None or image.size == 0:
                logger.error("Empty or invalid image provided")
                return None, []
                
            # Convert to BGR if needed (InsightFace expects BGR)
            if len(image.shape) == 3 and image.shape[2] == 3:
                if np.max(image) <= 1.0:  # If image is in [0,1] range
                    image = (image * 255).astype(np.uint8)
                if image.dtype != np.uint8:
                    image = image.astype(np.uint8)
                    
            # Detect faces
            logger.info("Detecting faces...")
            faces = self.detect_faces(image)
            logger.info(f"Detected {len(faces)} face(s)")
            
            if not faces:
                logger.warning("No faces detected in the image")
                return None, []
            
            # Log face detection details
            for i, face in enumerate(faces):
                logger.info(f"Face {i+1}: Confidence: {face.get('confidence', 0):.4f}, "
                          f"BBox: {face.get('bbox', [])}")
            
            # Get primary face (highest confidence)
            primary_face = max(faces, key=lambda x: x.get('confidence', 0))
            embedding = primary_face.get('embedding')
            
            # If no embedding in detection, try to extract it
            if embedding is None:
                logger.info("No embedding in detection, extracting separately...")
                embedding = self.extract_embedding(image, primary_face.get('bbox'))
                
            if embedding is None:
                logger.error("Failed to extract face embedding")
                return None, faces
                
            # Validate embedding
            if not isinstance(embedding, np.ndarray) or embedding.size == 0:
                logger.error(f"Invalid embedding: {type(embedding)}")
                return None, faces
                
            logger.info(f"Successfully extracted embedding: {embedding.shape}, "
                       f"Norm: {np.linalg.norm(embedding):.4f}")
                       
            return embedding, faces
            
        except Exception as e:
            logger.error(f"Error in process_image: {str(e)}", exc_info=True)
            return None, []

    def extract_embedding_from_face_crop(self, face_crop: np.ndarray) -> Optional[np.ndarray]:
        """
        Extract embedding directly from already cropped face image using InsightFace.
        
        Args:
            face_crop: Cropped face image as numpy array (RGB or BGR)
            
        Returns:
            Normalized face embedding vector or None if extraction fails
        """
        try:
            if face_crop is None or face_crop.size == 0:
                logger.error("Empty or invalid face crop provided")
                return None
                
            logger.debug(f"Processing face crop - Shape: {face_crop.shape}, Type: {face_crop.dtype}, "
                       f"Min: {np.min(face_crop)}, Max: {np.max(face_crop)}")
            
            # Ensure image is in uint8 format [0, 255]
            if face_crop.dtype != np.uint8:
                if np.max(face_crop) <= 1.0:  # [0, 1] range
                    face_crop = (face_crop * 255).astype(np.uint8)
                else:
                    face_crop = face_crop.astype(np.uint8)
            
            # Convert to BGR for InsightFace if needed
            if len(face_crop.shape) == 3 and face_crop.shape[2] == 3:
                try:
                    # Try to detect if it's RGB or BGR by checking channel order
                    if face_crop[0, 0, 0] > face_crop[0, 0, 2]:  # Likely BGR (blue > red)
                        bgr_face_crop = face_crop
                    else:  # Likely RGB, convert to BGR
                        bgr_face_crop = cv2.cvtColor(face_crop, cv2.COLOR_RGB2BGR)
                except Exception as e:
                    logger.warning(f"Error determining color space, assuming RGB: {e}")
                    bgr_face_crop = cv2.cvtColor(face_crop, cv2.COLOR_RGB2BGR)
            else:
                # Handle grayscale by converting to BGR
                if len(face_crop.shape) == 2:
                    bgr_face_crop = cv2.cvtColor(face_crop, cv2.COLOR_GRAY2BGR)
                else:
                    logger.error(f"Unexpected image format: {face_crop.shape}")
                    return None

            # Resize if too small (min 112x112 for ArcFace)
            min_size = 112
            if bgr_face_crop.shape[0] < min_size or bgr_face_crop.shape[1] < min_size:
                scale = max(min_size / bgr_face_crop.shape[0], min_size / bgr_face_crop.shape[1])
                new_size = (int(bgr_face_crop.shape[1] * scale), int(bgr_face_crop.shape[0] * scale))
                bgr_face_crop = cv2.resize(bgr_face_crop, new_size, interpolation=cv2.INTER_AREA)
                logger.debug(f"Resized face crop to {new_size}")

            logger.debug(f"Extracting embedding from face crop of size {bgr_face_crop.shape}")
            
            # Get embedding using InsightFace
            faces = self.insightface_app.get(bgr_face_crop)
            if not faces:
                logger.warning("No faces found in the cropped image")
                return None

            # Log all detected faces
            logger.debug(f"Found {len(faces)} face(s) in the cropped image")
            for i, face in enumerate(faces):
                logger.debug(f"  Face {i+1}: Confidence: {face.det_score:.4f}, "
                           f"BBox: {face.bbox.tolist() if hasattr(face, 'bbox') else 'N/A'}")

            # Use the face with highest confidence
            primary_face = max(faces, key=lambda x: x.det_score)
            if not hasattr(primary_face, 'embedding') or primary_face.embedding is None:
                logger.error("No embedding found in the primary face")
                return None
                
            embedding = primary_face.embedding.astype(np.float32)
            logger.debug(f"Extracted embedding with shape: {embedding.shape}")

            # Ensure normalization
            norm = np.linalg.norm(embedding)
            if norm <= 0:
                logger.error("Zero-length embedding vector")
                return None
                
            embedding = embedding / norm
            logger.debug(f"Normalized embedding - Norm: {np.linalg.norm(embedding):.6f}")

            return embedding
            
        except Exception as e:
            logger.error(f"Error extracting InsightFace embedding from crop: {str(e)}", exc_info=True)
            return None


class DatabaseHandler:
    """Enhanced database handler for Neon PostgreSQL with InsightFace and SCRFD"""
    
    def __init__(self, database_url: str):
        try:
            self.Session = init_models(database_url)
            self.embedder = InsightFaceEmbedder()
            logger.info("SQLAlchemy database handler initialized with InsightFace + SCRFD")
            
            # Test connection
            self._test_connection()
        except Exception as e:
            logger.error(f"Failed to initialize database handler: {e}")
            raise

    def _test_connection(self):
        """Test database connection - FIXED for SQLAlchemy 2.0"""
        session = self.Session()
        try:
            session.execute(text("SELECT 1"))
            logger.info("Database connection test successful")
        except Exception as e:
            logger.error(f"Database connection test failed: {e}")
            raise
        finally:
            session.close()

    def _bytes_to_image(self, image_data: bytes) -> Optional[np.ndarray]:
        """
        Convert image bytes to a numpy array in RGB format.
        
        Args:
            image_data: Binary image data (JPEG, PNG, etc.)
            
        Returns:
            Numpy array in RGB format with values in [0, 255] and dtype=uint8,
            or None if conversion fails
        """
        if not image_data or not isinstance(image_data, (bytes, bytearray)):
            logger.error("Invalid image data: must be non-empty bytes")
            return None
            
        try:
            # Try to determine image format from magic bytes
            image_format = None
            if image_data.startswith(b'\xff\xd8'):
                image_format = 'JPEG'
            elif image_data.startswith(b'\x89PNG\r\n\x1a\n'):
                image_format = 'PNG'
            elif image_data.startswith((b'II\x2A\x00', b'MM\x00\x2B')):
                image_format = 'TIFF'
            elif image_data.startswith(b'GIF87a') or image_data.startswith(b'GIF89a'):
                image_format = 'GIF'
            elif image_data.startswith(b'BM'):
                image_format = 'BMP'
                
            logger.debug(f"Detected image format: {image_format or 'Unknown'}")
            
            # Use OpenCV for more robust image loading
            try:
                # Convert bytes to numpy array
                nparr = np.frombuffer(image_data, np.uint8)
                
                # Decode image with OpenCV
                img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                
                if img is None:
                    logger.error("OpenCV failed to decode image data")
                    # Fall back to PIL if OpenCV fails
                    img_pil = Image.open(io.BytesIO(image_data))
                    if img_pil.mode != 'RGB':
                        img_pil = img_pil.convert('RGB')
                    img = np.array(img_pil)
                    logger.debug("Successfully decoded image using PIL fallback")
                else:
                    # Convert from BGR to RGB
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    logger.debug(f"Successfully decoded image using OpenCV. Shape: {img.shape}, Type: {img.dtype}")
                
                # Ensure we have a valid image
                if img.size == 0:
                    logger.error("Decoded image is empty")
                    return None
                    
                # Ensure correct data type and range
                if img.dtype != np.uint8:
                    if img.max() <= 1.0:  # Float image in [0,1] range
                        img = (img * 255).astype(np.uint8)
                    else:
                        img = img.astype(np.uint8)
                
                return img
                
            except Exception as cv_e:
                logger.warning(f"OpenCV image decoding failed, falling back to PIL: {str(cv_e)}")
                # Fall back to PIL if OpenCV fails
                try:
                    image = Image.open(io.BytesIO(image_data))
                    if image.mode != 'RGB':
                        image = image.convert('RGB')
                    logger.debug("Successfully decoded image using PIL")
                    return np.array(image)
                except Exception as pil_e:
                    logger.error(f"PIL image decoding failed: {str(pil_e)}")
                    raise ValueError(f"Failed to decode image data: {str(pil_e)}")
                    
        except Exception as e:
            logger.error(f"Error in _bytes_to_image: {str(e)}", exc_info=True)
            return None

    def add_user(self, name: str, image_data: bytes = None, image_path: str = None, is_face_crop: bool = False) -> Dict[str, Any]:
        """Add a new user with face encoding using InsightFace"""
        session = self.Session()
        try:
            # Validate input
            if not name or not name.strip():
                raise ValueError("Name cannot be empty")
        
            # Load image data from either bytes or file path
            if image_data is None and image_path is not None:
                with open(image_path, 'rb') as f:
                    image_data = f.read()
        
            if not image_data or len(image_data) == 0:
                raise ValueError("Image data cannot be empty")

            # Process image and extract face embedding
            image = self._bytes_to_image(image_data)
            if image is None:
                raise ValueError("Failed to convert image data")

            # Log image info for debugging
            logger.info(f"Processing image - Shape: {image.shape}, Type: {image.dtype}, "
                       f"Min: {image.min()}, Max: {image.max()}")

            if is_face_crop:
                logger.info("Processing as pre-cropped face image...")
                # For already cropped faces, extract embedding directly
                embedding = self.embedder.extract_embedding_from_face_crop(image)
                faces_detected = 1 if embedding is not None else 0
                primary_face_confidence = 1.0 if embedding is not None else 0
            
                if embedding is None:
                    # Try processing as full image as fallback
                    logger.info("Cropped face processing failed, trying as full image...")
                    embedding, faces = self.embedder.process_image(image)
                    faces_detected = len(faces) if faces else 0
                    primary_face_confidence = faces[0]['confidence'] if faces else 0
            else:
                # For full images, detect faces and then extract embedding
                logger.info("Detecting faces in full image...")
                embedding, faces = self.embedder.process_image(image)
                faces_detected = len(faces) if faces else 0
                primary_face_confidence = faces[0]['confidence'] if faces else 0

            if embedding is None:
                error_msg = (
                    "No face detected or embedding extraction failed. Possible reasons:\n"
                    "1. No face was detected in the image\n"
                    "2. The face is too small or not clearly visible\n"
                    "3. The image quality is too low\n"
                    "4. The face is at an extreme angle or partially occluded\n"
                    "Please try with a clear, front-facing image of a single face."
                )
                logger.error(f"Error adding user {name}: {error_msg}")
                raise ValueError(error_msg)

            if faces_detected > 1:
                logger.warning(f"Multiple faces detected for user {name}, using primary face")

            # Convert to list for database storage
            encoding_list = embedding.tolist()
        
            # Create and save user
            user = User(
                name=name.strip(),
                face_encoding=encoding_list,
                image_data=image_data,
                image_format='jpg'
            )
            session.add(user)
            session.commit()
        
            # Refresh to get the ID
            session.refresh(user)
        
            logger.info(f"Successfully registered user: {name} with ID: {user.id}")
        
            return {
                'id': user.id,
                'name': name,
                'faces_detected': faces_detected,
                'primary_face_confidence': primary_face_confidence,
                'status': 'success'
            }
        except Exception as e:
            session.rollback()
            logger.error(f"Error adding user {name}: {e}")
            raise
        finally:
            session.close()

    def get_user(self, user_id: int) -> Optional[User]:
        """Get user by ID"""
        session = self.Session()
        try:
            return session.query(User).filter_by(id=user_id).first()
        except Exception as e:
            logger.error(f"Error getting user {user_id}: {e}")
            return None
        finally:
            session.close()

    def get_user_image(self, user_id: int) -> Optional[Tuple[bytes, str]]:
        """Get user image data and format"""
        session = self.Session()
        try:
            user = session.query(User).filter_by(id=user_id).first()
            if user:
                return user.image_data, user.image_format
            return None
        except Exception as e:
            logger.error(f"Error getting user image {user_id}: {e}")
            return None
        finally:
            session.close()

    def get_all_users(self) -> List[Dict[str, Any]]:
        """Get all users without face encodings"""
        session = self.Session()
        try:
            users = session.query(User).all()
            return [{
                'id': user.id,
                'name': user.name,
                'date_created': user.date_created,
                'last_accessed': user.last_accessed,
            } for user in users]
        except Exception as e:
            logger.error(f"Error getting all users: {e}")
            return []
        finally:
            session.close()

    def get_all_users_with_encodings(self, include_encoding: bool = True) -> List[Dict[str, Any]]:
        """Get all users with their face encodings"""
        session = self.Session()
        try:
            users = session.query(User).all()
            result = []
            for user in users:
                user_data = {
                    'id': user.id,
                    'name': user.name,
                    'date_created': user.date_created,
                    'last_accessed': user.last_accessed,
                    'has_face_encoding': user.face_encoding is not None
                }
                if include_encoding and user.face_encoding is not None:
                    arr = np.array(user.face_encoding, dtype=np.float32)
                    user_data['face_encoding'] = arr
                    user_data['encoding_length'] = len(arr)
                result.append(user_data)
            return result
        except Exception as e:
            logger.error(f"Error getting users with encodings: {e}")
            return []
        finally:
            session.close()

    def find_similar_face(self, image_data: bytes, threshold: float = 0.7) -> Optional[Dict[str, Any]]:
        """Find similar face in database with dimension validation using InsightFace"""
        session = self.Session()
        try:
            image = self._bytes_to_image(image_data)
            if image is None:
                return None

            # Extract face and embedding using InsightFace
            embedding, faces = self.embedder.process_image(image)
            if embedding is None or not faces:
                return None

            # Get all users with their encodings
            users = session.query(User).all()
            if not users:
                return None

            best_match = None
            best_similarity = -1

            for user in users:
                if user.face_encoding is None:
                    continue

                try:
                    # Convert stored encoding to numpy array
                    stored_encoding = np.array(user.face_encoding, dtype=np.float32)
                    
                    # Calculate cosine similarity
                    similarity = cosine_similarity(embedding, stored_encoding)
                    
                    if similarity > best_similarity and similarity >= threshold:
                        best_similarity = similarity
                        best_match = {
                            'id': user.id,
                            'name': user.name,
                            'similarity': float(similarity),
                            'distance': 1.0 - float(similarity),
                            'faces_detected': len(faces),
                            'primary_face_confidence': faces[0]['confidence'] if faces else 0
                        }
                except Exception as e:
                    logger.warning(f"Error comparing with user {user.id}: {e}")
                    continue

            return best_match

        except Exception as e:
            logger.error(f"Error finding similar face: {e}")
            return None
        finally:
            session.close()

    def update_user_face(self, user_id: int, image_data: bytes) -> Dict[str, Any]:
        """Update user's face encoding using InsightFace"""
        session = self.Session()
        try:
            user = session.query(User).filter_by(id=user_id).first()
            if not user:
                return {'status': 'error', 'message': f'User {user_id} not found'}
                
            image = self._bytes_to_image(image_data)
            if image is None:
                return {'status': 'error', 'message': 'Failed to process image'}
                
            embedding, faces = self.embedder.process_image(image)
            if embedding is None:
                return {'status': 'error', 'message': 'No face detected in image'}
                
            user.face_encoding = embedding.tolist()
            user.image_data = image_data
            user.last_accessed = datetime.utcnow()
            
            session.commit()
            return {
                'status': 'success',
                'user_id': user_id,
                'faces_detected': len(faces),
                'primary_face_confidence': faces[0]['confidence'] if faces else 0
            }
        except Exception as e:
            session.rollback()
            logger.error(f"Error updating user face {user_id}: {e}")
            return {'status': 'error', 'message': str(e)}
        finally:
            session.close()

    def log_access_attempt(self, user_id: int, confidence: float, access_granted: bool, image_path: str = None) -> bool:
        """Log access attempt to database"""
        session = self.Session()
        try:
            log_data = {
                'user_id': user_id if user_id else None,
                'confidence': float(confidence),
                'access_granted': bool(access_granted),
            }
            
            log = AccessLog(**log_data)
            session.add(log)
            
            # Update user's last_accessed timestamp if user_id is provided
            if user_id is not None:
                user = session.query(User).filter_by(id=user_id).first()
                if user:
                    user.last_accessed = datetime.utcnow()
                    
            session.commit()
            logger.info(f"Access attempt logged: user_id={user_id}, confidence={confidence}, granted={access_granted}")
            return True
            
        except Exception as e:
            session.rollback()
            logger.error(f"Error logging access attempt: {e}")
            return False
        finally:
            session.close()

    def get_access_logs(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Get access logs from database"""
        session = self.Session()
        try:
            logs = session.query(AccessLog).order_by(AccessLog.timestamp.desc()).limit(limit).all()
            return [{
                'id': log.id,
                'user_id': log.user_id,
                'timestamp': log.timestamp,
                'confidence': log.confidence,
                'access_granted': log.access_granted,
            } for log in logs]
        except Exception as e:
            logger.error(f"Error getting access logs: {e}")
            return []
        finally:
            session.close()

    def delete_user(self, user_id: int) -> Dict[str, Any]:
        """Delete user from database"""
        session = self.Session()
        try:
            user = session.query(User).filter_by(id=user_id).first()
            if not user:
                return {'status': 'error', 'message': f'User {user_id} not found'}
                
            session.delete(user)
            session.commit()
            logger.info(f"User {user_id} deleted successfully")
            return {'status': 'success', 'message': f'User {user_id} deleted successfully'}
        except Exception as e:
            session.rollback()
            logger.error(f"Error deleting user {user_id}: {e}")
            return {'status': 'error', 'message': str(e)}
        finally:
            session.close()

    def get_user_count(self) -> int:
        """Get total number of users"""
        session = self.Session()
        try:
            return session.query(User).count()
        except Exception as e:
            logger.error(f"Error getting user count: {e}")
            return 0
        finally:
            session.close()

    def get_recent_access_logs(self, hours: int = 24) -> List[Dict[str, Any]]:
        """Get recent access logs from database"""
        session = self.Session()
        try:
            time_threshold = datetime.utcnow() - timedelta(hours=hours)
            logs = session.query(AccessLog).filter(
                AccessLog.timestamp >= time_threshold
            ).order_by(AccessLog.timestamp.desc()).all()
            
            return [{
                'id': log.id,
                'user_id': log.user_id,
                'timestamp': log.timestamp,
                'confidence': log.confidence,
                'access_granted': log.access_granted,
            } for log in logs]
        except Exception as e:
            logger.error(f"Error getting recent access logs: {e}")
            return []
        finally:
            session.close()

    def cleanup_old_logs(self, days: int = 30) -> int:
        """Clean up access logs older than specified days"""
        session = self.Session()
        try:
            cutoff_date = datetime.utcnow() - timedelta(days=days)
            result = session.query(AccessLog).filter(
                AccessLog.timestamp < cutoff_date
            ).delete()
            session.commit()
            logger.info(f"Cleaned up {result} access logs older than {days} days")
            return result
        except Exception as e:
            session.rollback()
            logger.error(f"Error cleaning up old logs: {e}")
            return 0
        finally:
            session.close()


if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv()
    
    db_url = os.getenv('DATABASE_URL')
    if not db_url:
        raise ValueError("DATABASE_URL not found in environment variables")
        
    db = DatabaseHandler(db_url)
    print("Database handler with InsightFace + SCRFD initialized successfully")
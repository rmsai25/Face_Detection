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
from facenet_pytorch import InceptionResnetV1

logger = logging.getLogger(__name__)

def cosine_similarity(a, b):
    """Calculate cosine similarity between two vectors"""
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

class FaceNetRetinaFaceEmbedder:
    """FaceNet (facenet-pytorch) with RetinaFace for face detection and embedding generation"""

    def __init__(self, device: str = None):
        logger.info("Initializing FaceNet (facenet-pytorch) with RetinaFace")
        try:
            from retinaface import RetinaFace
            self.retinaface = RetinaFace
            self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
            self.face_net = InceptionResnetV1(pretrained='vggface2').eval().to(self.device)
            logger.info("FaceNet (facenet-pytorch) and RetinaFace initialized successfully")
        except ImportError as e:
            logger.error(f"Failed to import required libraries: {e}")
            raise

    def detect_faces(self, image: np.ndarray) -> List[Dict[str, Any]]:
        """Detect faces using RetinaFace - optimized version"""
        try:
            # Convert to RGB for RetinaFace
            if len(image.shape) == 3 and image.shape[2] == 3:
                rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            else:
                rgb_image = image

            # Use numpy array directly instead of temp file for better performance
            faces = self.retinaface.detect_faces(rgb_image)

            face_detections = []
            if isinstance(faces, dict):
                for face_id, face_info in faces.items():
                    facial_area = face_info['facial_area']
                    confidence = face_info['score']
                    if confidence > 0.7:  # Confidence threshold
                        x1, y1, x2, y2 = map(int, facial_area)
                        face_detections.append({
                            'bbox': [x1, y1, x2, y2],
                            'confidence': float(confidence),
                            'landmarks': face_info.get('landmarks'),
                            'face_index': face_id
                        })
            return face_detections
        except Exception as e:
            logger.error(f"Error in face detection with RetinaFace: {e}")
            return []

    def extract_embedding(self, image: np.ndarray, face_bbox: List[int] = None) -> Optional[np.ndarray]:
        """Extract face embedding using FaceNet and ensure 512D output"""
        try:
            if face_bbox is not None:
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
            else:
                face_region = image

            # Preprocess for FaceNet
            face_region = cv2.resize(face_region, (160, 160))
            face_region = face_region.astype('float32') / 255.0
            face_region = np.transpose(face_region, (2, 0, 1))
            face_tensor = torch.tensor(face_region).unsqueeze(0).to(self.device)

            with torch.no_grad():
                embedding = self.face_net(face_tensor).cpu().numpy()[0]

            # Ensure 512D
            if len(embedding) != 512:
                logger.warning(f"Extracted embedding has {len(embedding)} dimensions, adjusting to 512D")
                if len(embedding) > 512:
                    embedding = embedding[:512]
                else:
                    padded = np.zeros(512, dtype=np.float32)
                    padded[:len(embedding)] = embedding
                    embedding = padded

            # Normalize embedding
            embedding = embedding / np.linalg.norm(embedding)
            return embedding.astype(np.float32)
            
        except Exception as e:
            logger.error(f"Error extracting face embedding: {e}")
            return None

    def process_image(self, image: np.ndarray) -> Tuple[Optional[np.ndarray], List[Dict[str, Any]]]:
        """Process image to detect faces and extract embeddings"""
        try:
            faces = self.detect_faces(image)
            if not faces:
                logger.warning("No faces detected in image")
                return None, []
            primary_face = max(faces, key=lambda x: x['confidence'])
            embedding = self.extract_embedding(image, primary_face['bbox'])
            return embedding, faces
        except Exception as e:
            logger.error(f"Error processing image: {e}")
            return None, []

    def extract_embedding_from_face_crop(self, face_crop: np.ndarray) -> Optional[np.ndarray]:
        """Extract embedding directly from already cropped face image"""
        try:
            # Preprocess for FaceNet
            face_region = cv2.resize(face_crop, (160, 160))
            face_region = face_region.astype('float32') / 255.0
            face_region = np.transpose(face_region, (2, 0, 1))
            face_tensor = torch.tensor(face_region).unsqueeze(0).to(self.device)

            with torch.no_grad():
                embedding = self.face_net(face_tensor).cpu().numpy()[0]

            # Ensure 512D
            if len(embedding) != 512:
                logger.warning(f"Extracted embedding has {len(embedding)} dimensions, adjusting to 512D")
                if len(embedding) > 512:
                    embedding = embedding[:512]
                else:
                    padded = np.zeros(512, dtype=np.float32)
                    padded[:len(embedding)] = embedding
                    embedding = padded

            # Normalize embedding
            embedding = embedding / np.linalg.norm(embedding)
            return embedding.astype(np.float32)
        except Exception as e:
            logger.error(f"Error extracting FaceNet embedding from crop: {e}")
            return None


class DatabaseHandler:
    """Enhanced database handler for Neon PostgreSQL with better error handling"""
    
    def __init__(self, database_url: str):
        try:
            self.Session = init_models(database_url)
            self.embedder = FaceNetRetinaFaceEmbedder()
            logger.info("SQLAlchemy database handler initialized with facenet-pytorch + RetinaFace")
            
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
        """Convert bytes to numpy image array"""
        try:
            image = Image.open(io.BytesIO(image_data))
            if image.mode != 'RGB':
                image = image.convert('RGB')
            return np.array(image)
        except Exception as e:
            logger.error(f"Error converting image data: {e}")
            return None

    def add_user(self, name: str, image_data: bytes = None, image_path: str = None, is_face_crop: bool = False) -> Dict[str, Any]:
        """Add a new user with face encoding"""
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

            if is_face_crop:
                # For already cropped faces, extract embedding directly
                embedding = self.embedder.extract_embedding_from_face_crop(image)
                faces_detected = 1
                primary_face_confidence = 1.0
            else:
                # For full images, detect faces and then extract embedding
                embedding, faces = self.embedder.process_image(image)
                faces_detected = len(faces) if faces else 0
                primary_face_confidence = faces[0]['confidence'] if faces else 0

            if embedding is None:
                raise ValueError("No face detected or embedding extraction failed")

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
        """Find similar face in database with dimension validation"""
        session = self.Session()
        try:
            image = self._bytes_to_image(image_data)
            if image is None:
                return None

            # Extract face and embedding
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
        """Update user's face encoding"""
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
    print("Database handler with facenet-pytorch + RetinaFace initialized successfully")
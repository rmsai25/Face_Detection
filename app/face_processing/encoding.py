import os
import cv2
import numpy as np
import tempfile
import logging
from typing import Tuple, List, Optional, Dict, Any, Union
import insightface
from insightface.app import FaceAnalysis
import torch
from PIL import Image
import torchvision.transforms as transforms

logger = logging.getLogger(__name__)

def find_cosine_distance(source_representation: Union[np.ndarray, List], 
                        test_representation: Union[np.ndarray, List]) -> np.ndarray:
    """
    Calculate cosine distance between two vectors.
    
    Args:
        source_representation: Source vector or list of vectors
        test_representation: Test vector or list of vectors
        
    Returns:
        Cosine distances
    """
    # Convert to numpy arrays
    if isinstance(source_representation, list):
        source_representation = np.array(source_representation, dtype=np.float32)
    if isinstance(test_representation, list):
        test_representation = np.array(test_representation, dtype=np.float32)

    # Ensure 2D arrays
    if len(source_representation.shape) == 1:
        source_representation = np.expand_dims(source_representation, axis=0)
    if len(test_representation.shape) == 1:
        test_representation = np.expand_dims(test_representation, axis=0)

    # Normalize vectors
    source_norm = np.linalg.norm(source_representation, axis=1, keepdims=True)
    test_norm = np.linalg.norm(test_representation, axis=1, keepdims=True)
    
    # Avoid division by zero
    source_norm = np.maximum(source_norm, 1e-10)
    test_norm = np.maximum(test_norm, 1e-10)
    
    a = source_representation / source_norm
    b = test_representation / test_norm
    
    cosine_similarity = np.sum(a * b, axis=1)
    return 1.0 - cosine_similarity


def cosine_similarity(source_representation: Union[np.ndarray, List], 
                     test_representation: Union[np.ndarray, List]) -> np.ndarray:
    """
    Calculate cosine similarity between two vectors.
    
    Args:
        source_representation: Source vector or list of vectors
        test_representation: Test vector or list of vectors
        
    Returns:
        Cosine similarities (0 to 1)
    """
    cosine_dist = find_cosine_distance(source_representation, test_representation)
    return 1.0 - cosine_dist

class FaceEncoder:
    """
    Face encoder using InsightFace with SCRFD for detection and ArcFace for embedding generation.
    """
    
    # Default thresholds for different distance metrics
    DEFAULT_THRESHOLDS = {
        'cosine': 0.6,      # Cosine distance threshold
        'euclidean': 0.8,   # Euclidean distance threshold
        'similarity': 0.4   # Cosine similarity threshold
    }
    
    def __init__(self, device: str = 'auto', distance_metric: str = 'cosine'):
        """
        Initialize FaceEncoder with InsightFace (SCRFD + ArcFace).
        
        Args:
            device: 'auto', 'cuda', or 'cpu'
            distance_metric: 'cosine' or 'euclidean'
        """
        if device == 'auto':
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)

        logger.info(f"Using device: {self.device}")
        
        if torch.cuda.is_available():
            logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
            logger.info(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f}GB")

        # Initialize InsightFace model with SCRFD detector and ArcFace recognizer
        self.app = FaceAnalysis(
            name='buffalo_l',  # This includes SCRFD detector and ArcFace recognizer
            providers=['CUDAExecutionProvider'] if self.device.type == 'cuda' else ['CPUExecutionProvider']
        )
        
        # Prepare the model with appropriate context
        ctx_id = 0 if self.device.type == 'cuda' else -1
        self.app.prepare(ctx_id=ctx_id, det_size=(640, 640))
        
        # Set distance metric and threshold
        self.distance_metric = distance_metric.lower()
        if self.distance_metric not in ['cosine', 'euclidean']:
            logger.warning(f"Unsupported distance metric: {distance_metric}. Using 'cosine'")
            self.distance_metric = 'cosine'
            
        self.recognition_threshold = self.DEFAULT_THRESHOLDS.get(self.distance_metric, 0.6)

        # Transform for manual face processing (fallback)
        self.transform = transforms.Compose([
            transforms.Resize((112, 112)),  # InsightFace uses 112x112
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
        
        logger.info(f"FaceEncoder initialized with {self.distance_metric} distance metric using InsightFace")

    def detect_faces(self, image: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """
        Detect faces in an image using InsightFace SCRFD.
        
        Args:
            image: BGR or RGB image array
            
        Returns:
            List of face bounding boxes as (top, right, bottom, left)
        """
        try:
            # Convert to BGR if needed (InsightFace expects BGR by default)
            if len(image.shape) == 3 and image.shape[2] == 3:
                if image.dtype == np.uint8:
                    # Check if it's RGB (convert to BGR for InsightFace)
                    bgr_image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                else:
                    bgr_image = image
            else:
                bgr_image = image

            # Detect faces using InsightFace SCRFD
            faces = self.app.get(bgr_image)
            
            if not faces:
                return []

            face_locations = []
            for face in faces:
                try:
                    confidence = face.det_score
                    
                    # Filter by confidence
                    if confidence < 0.7:  # Minimum confidence threshold
                        continue
                        
                    # Get bounding box (SCRFD returns [x1, y1, x2, y2])
                    bbox = face.bbox.astype(int)
                    x1, y1, x2, y2 = bbox
                    
                    # Convert to (top, right, bottom, left) format
                    face_locations.append((y1, x2, y2, x1))
                        
                except Exception as e:
                    logger.warning(f"Error processing face: {e}")
                    continue

            logger.debug(f"Detected {len(face_locations)} faces")
            return face_locations
            
        except Exception as e:
            logger.error(f"Error detecting faces with InsightFace SCRFD: {e}")
            return []

    def preprocess_face(self, face_image: np.ndarray) -> Optional[torch.Tensor]:
        """
        Preprocess face image for InsightFace model.
        
        Args:
            face_image: RGB face image array
            
        Returns:
            Preprocessed face tensor with batch dimension
        """
        try:
            if not isinstance(face_image, np.ndarray):
                raise ValueError(f"Expected numpy array, got {type(face_image)}")
                
            if face_image.size == 0:
                logger.warning("Empty face image provided")
                return None

            # Ensure RGB format
            if len(face_image.shape) == 2:  # Grayscale
                face_image = cv2.cvtColor(face_image, cv2.COLOR_GRAY2RGB)
            elif face_image.shape[2] == 4:  # RGBA
                face_image = cv2.cvtColor(face_image, cv2.COLOR_RGBA2RGB)
            elif face_image.shape[2] == 3:  # Assume BGR if from OpenCV
                face_image = cv2.cvtColor(face_image, cv2.COLOR_BGR2RGB)

            # Convert to PIL Image and apply transforms
            pil_image = Image.fromarray(face_image)
            face_tensor = self.transform(pil_image)
            
            return face_tensor.unsqueeze(0)  # Add batch dimension
            
        except Exception as e:
            logger.error(f"Error preprocessing face: {e}")
            return None

    def encode_face(self, image_path: Optional[str] = None, 
                   image_array: Optional[np.ndarray] = None, 
                   is_face_crop: bool = False) -> Optional[np.ndarray]:
        """
        Encode a face image to a 512D embedding using InsightFace ArcFace.
        
        Args:
            image_path: Path to image file
            image_array: Image array in BGR or RGB format
            is_face_crop: If True, input is already a cropped face
            
        Returns:
            512D face embedding or None if failed
        """
        try:
            # Load and validate image
            if image_path:
                if not os.path.exists(image_path):
                    logger.error(f"Image not found: {image_path}")
                    return None
                    
                img = cv2.imread(image_path)
                if img is None:
                    logger.error(f"Failed to read image: {image_path}")
                    return None
                    
            elif image_array is not None:
                img = image_array.copy()  # Work with copy to avoid modifying original
            else:
                raise ValueError("Either image_path or image_array must be provided")

            # Process based on input type
            if is_face_crop:
                # For cropped faces, we need to use InsightFace on the crop
                # Convert to BGR for InsightFace
                if len(img.shape) == 3 and img.shape[2] == 3:
                    if img[0, 0, 0] > img[0, 0, 2]:
                        bgr_crop = img
                    else:
                        bgr_crop = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                else:
                    bgr_crop = img
                
                # Use InsightFace on the cropped face
                faces = self.app.get(bgr_crop)
                if not faces:
                    logger.warning("No face detected in cropped image")
                    return None
                
                # Use the face with highest confidence
                face = max(faces, key=lambda x: x.det_score)
                embedding = face.embedding
                
            else:
                # For full images, let InsightFace handle detection and embedding
                # Convert to BGR for InsightFace
                if len(img.shape) == 3 and img.shape[2] == 3:
                    if img[0, 0, 0] > img[0, 0, 2]: 
                        bgr_img = img
                    else:
                        bgr_img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                else:
                    bgr_img = img
                
                faces = self.app.get(bgr_img)
                if not faces:
                    logger.warning("No face detected in image")
                    return None
                
                # Use the face with highest confidence
                face = max(faces, key=lambda x: x.det_score)
                embedding = face.embedding

            # InsightFace embeddings are already normalized
            embedding = embedding.astype(np.float32)
            logger.debug(f"Generated embedding with norm: {np.linalg.norm(embedding):.4f}")
            return embedding
            
        except Exception as e:
            logger.error(f"Error encoding face: {e}")
            return None
        
    def encode_face_from_crop(self, face_image: np.ndarray) -> Optional[np.ndarray]:
        """
    Encode already cropped face image using InsightFace.
    
    Args:
        face_image: Cropped face image (BGR or RGB)
        
    Returns:
        512D normalized embedding
    """
        try:
            # Ensure the image is valid
            if face_image is None or face_image.size == 0:
                logger.warning("Empty or invalid face image provided")
                return None

            # Debug: Log the cropped image info
            logger.debug(f"Cropped face image - Shape: {face_image.shape}, Type: {face_image.dtype}, "
                        f"Min: {face_image.min()}, Max: {face_image.max()}")

            # Handle different color formats
            if len(face_image.shape) == 3 and face_image.shape[2] == 3:
                # Check if it's likely RGB or BGR using a simple heuristic
                # BGR typically has blue channel > red channel in natural images
                if face_image.shape[0] > 10 and face_image.shape[1] > 10:  # Only if image is large enough
                    blue_mean = np.mean(face_image[:, :, 0])
                    red_mean = np.mean(face_image[:, :, 2])
                
                    if blue_mean < red_mean:  # If red > blue, likely RGB
                        logger.debug("Converting cropped face from RGB to BGR")
                        bgr_face_crop = cv2.cvtColor(face_image, cv2.COLOR_RGB2BGR)
                    else:
                        bgr_face_crop = face_image  # Already BGR
                else:
                    # For small images, try both formats
                    bgr_face_crop = face_image
            else:
                bgr_face_crop = face_image

            # Ensure the image is large enough for InsightFace
            min_face_size = 20  # Minimum face size in pixels
            if bgr_face_crop.shape[0] < min_face_size or bgr_face_crop.shape[1] < min_face_size:
                logger.warning(f"Cropped face too small: {bgr_face_crop.shape}. Minimum: {min_face_size}x{min_face_size}")
                return None

            # Get embedding using InsightFace
            faces = self.app.get(bgr_face_crop)
        
            if not faces:
                logger.warning("No face detected in cropped image by InsightFace")
                # Try with the original format as fallback
                if len(face_image.shape) == 3 and face_image.shape[2] == 3:
                    try:
                        faces = self.app.get(face_image)  # Try original format
                        if faces:
                            logger.debug("Face detected using original image format")
                    except Exception as e:
                        logger.debug(f"Fallback detection failed: {e}")
                return None

            # Use the face with highest confidence
            primary_face = max(faces, key=lambda x: x.det_score)
            embedding = primary_face.embedding.astype(np.float32)

            # Ensure normalization
            norm = np.linalg.norm(embedding)
            if norm > 0:
                embedding = embedding / norm
                logger.debug(f"Generated embedding from crop - Norm: {norm:.4f}, Shape: {embedding.shape}")
            else:
                logger.warning("Zero-norm embedding generated")
                return None

            return embedding
        
        except Exception as e:
            logger.error(f"Error extracting InsightFace embedding from crop: {e}")
            return None

    def encode_faces(self, frame: np.ndarray, 
                    face_locations: List[Tuple[int, int, int, int]]) -> Tuple[List, List]:
        """
        Encode multiple detected faces in a frame using InsightFace.
        
        Args:
            frame: RGB frame
            face_locations: List of (top, right, bottom, left) bounding boxes
            
        Returns:
            Tuple of (valid_face_locations, face_encodings)
        """
        face_encodings = []
        valid_face_locations = []

        # Convert frame to BGR for InsightFace
        bgr_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        
        # Get all faces in one go using InsightFace
        faces = self.app.get(bgr_frame)
        
        if not faces:
            return [], []

        # Match detected faces with provided locations
        for location in face_locations:
            try:
                top, right, bottom, left = location
                
                # Find the face that matches this location
                matching_face = None
                for face in faces:
                    bbox = face.bbox.astype(int)
                    x1, y1, x2, y2 = bbox
                    
                    # Check if this face matches the location (with some tolerance)
                    if (abs(x1 - left) < 10 and abs(y1 - top) < 10 and 
                        abs(x2 - right) < 10 and abs(y2 - bottom) < 10):
                        matching_face = face
                        break
                
                if matching_face is not None:
                    embedding = matching_face.embedding
                    face_encodings.append(embedding)
                    valid_face_locations.append(location)
                    
            except Exception as e:
                logger.error(f"Error encoding face at {location}: {e}")
                continue

        logger.debug(f"Encoded {len(face_encodings)} faces from {len(face_locations)} detections")
        return valid_face_locations, face_encodings

    def verify_faces(self, source_image: Union[str, np.ndarray], 
                    target_image: Union[str, np.ndarray]) -> Dict[str, Any]:
        """
        Verify if two images contain the same face using InsightFace.
        
        Args:
            source_image: Source image path or array
            target_image: Target image path or array
            
        Returns:
            Verification results dictionary
        """
        try:
            # Encode source face
            if isinstance(source_image, np.ndarray):
                encoding1 = self.encode_face(image_array=source_image)
            else:
                encoding1 = self.encode_face(image_path=source_image)

            if encoding1 is None:
                return {
                    'verified': False, 
                    'distance': float('inf'),
                    'similarity': 0.0,
                    'message': 'No face found in source image',
                    'model': 'InsightFace (ArcFace)', 
                    'distance_metric': self.distance_metric,
                    'threshold': self.recognition_threshold
                }

            # Encode target face
            if isinstance(target_image, np.ndarray):
                encoding2 = self.encode_face(image_array=target_image)
            else:
                encoding2 = self.encode_face(image_path=target_image)

            if encoding2 is None:
                return {
                    'verified': False, 
                    'distance': float('inf'),
                    'similarity': 0.0,
                    'message': 'No face found in target image',
                    'model': 'InsightFace (ArcFace)', 
                    'distance_metric': self.distance_metric,
                    'threshold': self.recognition_threshold
                }

            # Calculate distance and similarity
            if self.distance_metric == 'cosine':
                distance = find_cosine_distance(encoding1, encoding2)[0]
                similarity = 1.0 - distance
            else:  # euclidean
                distance = np.linalg.norm(encoding1 - encoding2)
                similarity = 1.0 / (1.0 + distance)  # Convert to similarity score

            verified = distance <= self.recognition_threshold

            return {
                'verified': verified,
                'distance': float(distance),
                'similarity': float(similarity),
                'threshold': self.recognition_threshold,
                'model': 'InsightFace (ArcFace)',
                'distance_metric': self.distance_metric,
                'message': 'Verification successful' if verified else 'Faces do not match'
            }
            
        except Exception as e:
            logger.error(f"Error in face verification: {e}")
            return {
                'verified': False,
                'distance': float('inf'),
                'similarity': 0.0,
                'message': str(e),
                'model': 'InsightFace (ArcFace)',
                'distance_metric': self.distance_metric,
                'threshold': self.recognition_threshold
            }

    def find_best_match(self, target_encoding: Union[np.ndarray, List], 
                       known_encodings: List[Union[np.ndarray, List]], 
                       known_names: List[str], 
                       threshold: Optional[float] = None) -> Tuple[Optional[str], float, bool]:
        """
        Find the best match for a face encoding from known encodings.
        
        Args:
            target_encoding: Target face encoding
            known_encodings: List of known face encodings
            known_names: List of names corresponding to known_encodings
            threshold: Optional custom threshold
            
        Returns:
            Tuple of (best_match_name, confidence, is_verified)
        """
        if threshold is None:
            threshold = self.recognition_threshold

        # Validate inputs
        if not known_encodings or not known_names:
            logger.warning("No known encodings or names provided")
            return None, 0.0, False
            
        if len(known_encodings) != len(known_names):
            logger.warning("Mismatch between known_encodings and known_names lengths")
            return None, 0.0, False

        try:
            # Normalize target encoding (InsightFace embeddings are already normalized, but ensure)
            target_encoding = np.array(target_encoding, dtype=np.float32).flatten()
            target_norm = np.linalg.norm(target_encoding)
            if target_norm < 1e-10:
                logger.warning("Target encoding has zero norm")
                return None, 0.0, False
            target_encoding = target_encoding / target_norm

            best_match = {
                'name': None, 
                'distance': float('inf'), 
                'similarity': -1.0,
                'index': -1
            }

            # Find best match
            for i, (enc, name) in enumerate(zip(known_encodings, known_names)):
                if enc is None:
                    continue
                    
                try:
                    # Normalize known encoding
                    enc_array = np.array(enc, dtype=np.float32).flatten()
                    enc_norm = np.linalg.norm(enc_array)
                    if enc_norm < 1e-10:
                        continue
                    enc_array = enc_array / enc_norm

                    # Calculate similarity
                    similarity = float(np.dot(target_encoding, enc_array))
                    
                    if self.distance_metric == 'cosine':
                        distance = 1.0 - similarity
                    else:  # euclidean
                        distance = np.linalg.norm(target_encoding - enc_array)
                        similarity = 1.0 / (1.0 + distance)  # Convert to similarity

                    if distance < best_match['distance']:
                        best_match.update({
                            'name': name,
                            'distance': distance,
                            'similarity': similarity,
                            'index': i
                        })
                        
                except Exception as e:
                    logger.error(f"Error processing known encoding {i} ({name}): {e}")
                    continue

            if best_match['name'] is None:
                return None, 0.0, False

            # Determine verification
            is_verified = best_match['distance'] <= threshold
            confidence = best_match['similarity']

            logger.info(f"Best match: {best_match['name']} "
                       f"(distance: {best_match['distance']:.4f}, "
                       f"similarity: {confidence:.4f}, "
                       f"verified: {is_verified})")

            return best_match['name'], confidence, is_verified

        except Exception as e:
            logger.error(f"Error in find_best_match: {e}")
            return None, 0.0, False

    def process_frame(self, frame: np.ndarray) -> Tuple[List, List, np.ndarray]:
        """
        Detect and encode faces from a BGR frame using InsightFace.
        
        Args:
            frame: BGR image from OpenCV
            
        Returns:
            Tuple of (valid_face_locations, face_encodings, rgb_frame)
        """
        try:
            # Convert to RGB for compatibility with existing code
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Detect faces using InsightFace SCRFD
            face_locations = self.detect_faces(rgb_frame)
            
            # Encode faces using InsightFace ArcFace
            valid_face_locations, face_encodings = self.encode_faces(rgb_frame, face_locations)
            
            return valid_face_locations, face_encodings, rgb_frame
            
        except Exception as e:
            logger.error(f"Error processing frame: {e}")
            return [], [], frame

    def set_threshold(self, threshold: float):
        """
        Set recognition threshold.
        
        Args:
            threshold: New threshold value
        """
        if threshold < 0:
            raise ValueError("Threshold must be non-negative")
            
        old_threshold = self.recognition_threshold
        self.recognition_threshold = threshold
        logger.info(f"Threshold changed from {old_threshold} to {threshold}")

    def get_model_info(self) -> Dict[str, Any]:
        """
        Get model information.
        
        Returns:
            Dictionary with model info
        """
        return {
            'face_detector': 'SCRFD (via InsightFace)',
            'face_recognizer': 'ArcFace (via InsightFace)',
            'embedding_dim': 512,
            'distance_metric': self.distance_metric,
            'threshold': self.recognition_threshold,
            'device': str(self.device),
            'model_name': 'buffalo_l'
        }


# Example usage and testing
if __name__ == "__main__":
    # Set up logging
    logging.basicConfig(level=logging.INFO)
    
    # Test the face encoder
    encoder = FaceEncoder()
    
    # Test with sample images
    test_image1 = "path/to/test1.jpg"  # Replace with actual path
    test_image2 = "path/to/test2.jpg"  # Replace with actual path
    
    if os.path.exists(test_image1) and os.path.exists(test_image2):
        # Test verification
        result = encoder.verify_faces(test_image1, test_image2)
        print("Verification result:", result)
        
        # Test single encoding
        encoding = encoder.encode_face(image_path=test_image1)
        if encoding is not None:
            print(f"Encoding shape: {encoding.shape}, norm: {np.linalg.norm(encoding):.4f}")
    else:
        print("Test images not found, please update paths")

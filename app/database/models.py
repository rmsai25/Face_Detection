from datetime import datetime
from sqlalchemy import create_engine, Column, Integer, String, Float, Boolean, DateTime, ForeignKey, Text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship
from sqlalchemy import types
import numpy as np
import json
import logging
from typing import Optional, Any

logger = logging.getLogger(__name__)

# Custom type for storing numpy arrays with consistent 512D output
class NumpyArray(types.TypeDecorator):
    """Custom type for storing numpy arrays with consistent 512D output"""
    impl = types.LargeBinary
    
    def __init__(self, *args, **kwargs):
        self.target_dim = 512  # Force 512 dimensions
        super().__init__()
        
    def process_bind_param(self, value, dialect):
        """Convert numpy array to bytes with consistent 512D output"""
        if value is None:
            return None
            
        try:
            # Convert input to numpy array if it isn't already
            if not isinstance(value, np.ndarray):
                value = np.array(value, dtype=np.float32)
                
            # Ensure we have a 1D array
            value = value.reshape(-1)
            
            # Handle dimension conversion
            if len(value) == self.target_dim:
                # Already correct dimension
                return value.astype(np.float32).tobytes()
            elif len(value) > self.target_dim:
                # If larger than target, take first 512 dimensions
                logger.warning(f"Truncating embedding from {len(value)}D to {self.target_dim}D")
                return value[:self.target_dim].astype(np.float32).tobytes()
            else:
                # If smaller, pad with zeros
                logger.warning(f"Padding embedding from {len(value)}D to {self.target_dim}D")
                padded = np.zeros(self.target_dim, dtype=np.float32)
                padded[:len(value)] = value
                return padded.tobytes()
                
        except Exception as e:
            logger.error(f"Error processing numpy array: {e}")
            raise ValueError(f"Could not process array of shape {getattr(value, 'shape', 'unknown')}") from e
    
    def process_result_value(self, value, dialect):
        """Convert bytes back to numpy array, ensuring 512D"""
        if value is None:
            return None
        try:
            # Convert bytes to numpy array and ensure it's float32
            arr = np.frombuffer(value, dtype=np.float32)
            # Ensure it's exactly 512D
            if len(arr) != self.target_dim:
                logger.warning(f"Loaded embedding has {len(arr)} dimensions, expected {self.target_dim}")
                if len(arr) > self.target_dim:
                    arr = arr[:self.target_dim]
                else:
                    padded = np.zeros(self.target_dim, dtype=np.float32)
                    padded[:len(arr)] = arr
                    arr = padded
            return arr
        except Exception as e:
            logger.error(f"Error converting bytes to numpy array: {e}")
            return None

# Create base class for our models
Base = declarative_base()

class User(Base):
    """User model for storing user information and face encodings"""
    __tablename__ = 'face_users'
    
    id = Column(Integer, primary_key=True)
    name = Column(String(100), unique=True, nullable=False, index=True)
    face_encoding = Column(NumpyArray(), nullable=False)  # This will ensure 512D
    image_data = Column(types.LargeBinary, nullable=False)
    image_format = Column(String(10), default='jpg')
    date_created = Column(DateTime, default=datetime.utcnow, index=True)
    last_accessed = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, index=True)
    
    # Relationship with access logs
    access_logs = relationship("AccessLog", back_populates="user", cascade="all, delete-orphan", lazy='dynamic')
    
    def __repr__(self):
        return f"<User(id={self.id}, name='{self.name}')>"
    
    def to_dict(self, include_encoding: bool = False) -> dict:
        """Convert user to dictionary"""
        data = {
            'id': self.id,
            'name': self.name,
            'image_format': self.image_format,
            'date_created': self.date_created.isoformat() if self.date_created else None,
            'last_accessed': self.last_accessed.isoformat() if self.last_accessed else None,
            'has_face_encoding': self.face_encoding is not None
        }
        
        if include_encoding and self.face_encoding is not None:
            data['face_encoding'] = self.face_encoding.tolist() if hasattr(self.face_encoding, 'tolist') else self.face_encoding
            data['encoding_length'] = len(self.face_encoding) if self.face_encoding is not None else 0
            
        return data

class AccessLog(Base):
    """Access log for tracking face recognition attempts"""
    __tablename__ = 'face_access_logs'
    
    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, ForeignKey('face_users.id', ondelete='CASCADE'), index=True, nullable=True)
    timestamp = Column(DateTime, default=datetime.utcnow, index=True)
    confidence = Column(Float)
    access_granted = Column(Boolean, default=False)
    face_encoding = Column(NumpyArray(), nullable=True)  # This will ensure 512D
    detection_metadata = Column(Text, nullable=True)  # JSON string for additional metadata
    
    # Relationship with user
    user = relationship("User", back_populates="access_logs")
    
    def __repr__(self):
        return f"<AccessLog(id={self.id}, user_id={self.user_id}, granted={self.access_granted})>"
    
    def set_detection_metadata(self, metadata: dict):
        """Store detection metadata as JSON"""
        if metadata:
            self.detection_metadata = json.dumps(metadata)
    
    def get_detection_metadata(self) -> Optional[dict]:
        """Retrieve detection metadata"""
        if self.detection_metadata:
            try:
                return json.loads(self.detection_metadata)
            except json.JSONDecodeError:
                logger.warning(f"Failed to parse detection metadata for log {self.id}")
        return None
    
    def to_dict(self) -> dict:
        """Convert access log to dictionary"""
        return {
            'id': self.id,
            'user_id': self.user_id,
            'timestamp': self.timestamp.isoformat() if self.timestamp else None,
            'confidence': self.confidence,
            'access_granted': self.access_granted,
            'detection_metadata': self.get_detection_metadata()
        }

def init_db(database_url: str, echo: bool = False, **kwargs) -> sessionmaker:
    """
    Initialize the database and create tables
    
    Args:
        database_url: Database connection URL
        echo: Whether to output SQL queries
        **kwargs: Additional arguments for create_engine
        
    Returns:
        Session maker instance
    """
    try:
        # Create engine with connection pooling
        engine_kwargs = {
            'echo': echo,
            'pool_pre_ping': True,  # Verify connections before using them
            'pool_recycle': 3600,   # Recycle connections after 1 hour
        }
        engine_kwargs.update(kwargs)
        
        # Create engine
        engine = create_engine(database_url, **engine_kwargs)
        # Create all tables
        Base.metadata.create_all(engine)
        logger.info(f"Database initialized successfully: {database_url}")
        # Create session factory
        Session = sessionmaker(bind=engine)
        return Session
        
    except Exception as e:
        logger.error(f"Failed to initialize database: {e}")
        raise

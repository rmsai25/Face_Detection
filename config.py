import os
import logging
from typing import Dict, Any, Optional
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

class Config:
    """Simplified Configuration class for Face Recognition System"""
    
    def __init__(self):
        # Application settings
        self.APP_NAME = "Face Recognition System"
        self.APP_VERSION = "1.0.0"
        
        # Camera settings
        self.CAMERA_SOURCE = os.getenv('CAMERA_SOURCE', 'webcam')
        self.CAMERA_INDEX = int(os.getenv('CAMERA_INDEX', 0))
        self.FRAME_WIDTH = int(os.getenv('FRAME_WIDTH', 640))
        self.FRAME_HEIGHT = int(os.getenv('FRAME_HEIGHT', 480))
        
        # Face Recognition settings
        self.FACE_DETECTION_MODEL = os.getenv('FACE_DETECTION_MODEL', 'retinaface')
        self.RECOGNITION_THRESHOLD = float(os.getenv('RECOGNITION_THRESHOLD', 0.6))
        
        # Feature flags
        self.ENABLE_DEPTH_ESTIMATION = False  # Hardcoded to False since not implemented
        self.ENABLE_FACE_DETECTION = True     # Always enabled
        
        # Database settings
        self.DATABASE_URL = self._get_required('DATABASE_URL')
        
        # Logging settings
        self.LOG_LEVEL = os.getenv('LOG_LEVEL', 'INFO')
        
        # Application modes
        self.REGISTRATION_MODE = self._get_bool('REGISTRATION_MODE', False)
        
        # Display settings
        self.SHOW_VIDEO_FEED = self._get_bool('SHOW_VIDEO_FEED', True)
        
        # Validate configuration
        self._validate()
    
    def _get_bool(self, key: str, default: bool) -> bool:
        """Get boolean value from environment variables"""
        value = os.getenv(key)
        if value is None:
            return default
        return value.lower() in ('true', '1', 't', 'yes', 'y')
    
    def _get_required(self, key: str) -> str:
        """Get required environment variable"""
        value = os.getenv(key)
        if not value:
            raise ValueError(f"Required environment variable {key} is not set")
        return value
    
    def _validate(self):
        """Validate configuration values"""
        if self.CAMERA_SOURCE not in ['webcam', 'phone', 'rtsp']:
            raise ValueError(f"Invalid CAMERA_SOURCE: {self.CAMERA_SOURCE}")
        
        if not 0 <= self.RECOGNITION_THRESHOLD <= 1:
            raise ValueError("RECOGNITION_THRESHOLD must be between 0 and 1")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary"""
        return {k: v for k, v in self.__dict__.items() if not k.startswith('_')}

# Global configuration instance
config = Config()

def print_config_summary():
    """Print configuration summary"""
    summary = config.to_dict()
    
    print("=" * 50)
    print(f"{config.APP_NAME} v{config.APP_VERSION} - Configuration")
    print("=" * 50)
    
    for key, value in summary.items():
        print(f"{key}: {value}")
    
    print("=" * 50)

if __name__ == "__main__":
    print_config_summary()




    
    
    
    






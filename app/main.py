import os 
import time
import cv2
import argparse
import logging
from dotenv import load_dotenv

'''from database.db_handler import DatabaseHandler
from face_processing.capture import VideoCapture
from face_processing.encoding import FaceEncoder
from face_processing.recognition import FaceRecognizer
from utils.logger import setup_logger
from config import config'''

from app.database.db_handler import DatabaseHandler
from app.face_processing.capture import VideoCapture
from app.face_processing.encoding import FaceEncoder
from app.face_processing.recognition import FaceRecognizer
from app.utils.logger import setup_logger
from app.config import config

# Suppress TensorFlow logging
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# Load environment variables
load_dotenv()

import torch

def check_gpu_status():
    """Check if GPU is available and display GPU information"""
    print("🔍 Checking GPU Status...")
    print(f"CUDA Available: {torch.cuda.is_available()}")

    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f}GB")
        print(f"CUDA Version: {torch.version.cuda}")
        print("✅ GPU acceleration is enabled!")
    else:
        print("❌ No GPU detected - running on CPU")
        print(f"PyTorch Version: {torch.__version__}")

    return torch.cuda.is_available() if torch.cuda.is_available() else False

class FaceDetectionSystem:
    def __init__(self, registration_mode=False):
        """
        Initialize the Face Detection System.
        
        Args:
            registration_mode (bool): If True, runs in face registration mode
        """
        # Disable file logging completely
        self.logger = setup_logger(
            level=config.LOG_LEVEL,
            enable_file=False,  # Disable file logging
            log_dir=None,       # Don't create log directory
            enable_console=True  # Keep console logging
        )
        
        # Clean up any existing log files if they exist
        log_dir = os.path.join(os.path.dirname(__file__), 'logs')
        if os.path.exists(log_dir):
            import shutil
            shutil.rmtree(log_dir, ignore_errors=True)
        self.registration_mode = registration_mode
        self.running = False
        
        try:
            # Initialize database connection
            self.db = DatabaseHandler(config.DATABASE_URL)
            self.logger.info("Database connection established")
            
            # Initialize face processing components
            self.face_encoder = FaceEncoder()
            self.face_recognizer = FaceRecognizer(
                face_encoder=self.face_encoder,
                registration_mode=self.registration_mode,
                create_dirs=False
            )
            
            # Load known faces from database (only in recognition mode)
            if not self.registration_mode:
                self.load_known_faces()
            
            # Initialize video capture with simplified parameters
            self.video_capture = VideoCapture(
                source=config.CAMERA_SOURCE,
                width=config.FRAME_WIDTH,
                height=config.FRAME_HEIGHT
            )
            
            if not self.video_capture.initialize():
                raise RuntimeError("Failed to initialize video capture")
                
            self.logger.info("Video capture initialized successfully")
            self.logger.info(f"System initialized in {'registration' if self.registration_mode else 'recognition'} mode")
            
        except Exception as e:
            self.logger.error(f"Error initializing system: {e}")
            raise

    def load_known_faces(self):
        """Load known faces from database"""
        try:
            users = self.db.get_all_users_with_encodings(include_encoding=True)
            if not users:
                self.logger.warning("No users with face encodings found in database")
                return
                
            success = self.face_recognizer.load_known_faces(users)
            if success:
                self.logger.info(f"Loaded {len(users)} known faces from database")
            else:
                self.logger.error("Failed to load known faces")
                
        except Exception as e:
            self.logger.error(f"Error loading known faces: {e}")

    def process_frame(self, frame):
        """
        Process a single frame using the FaceRecognizer.
        
        Args:
            frame: Input frame from camera
            
        Returns:
            Processed frame with annotations
        """
        try:
            # Use FaceRecognizer to process the frame
            processed_frame, face_data = self.face_recognizer.process_frame(frame, self.db)
            
            # Add mode indicator
            mode_text = "Mode: Registration" if self.registration_mode else "Mode: Recognition"
            cv2.putText(processed_frame, mode_text, (10, 25), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
            
            # Add performance info if available
            if hasattr(self.face_recognizer, 'get_performance_stats'):
                perf_stats = self.face_recognizer.get_performance_stats()
                fps_text = f"FPS: {perf_stats['fps']:.1f}"
                cv2.putText(processed_frame, fps_text, (10, processed_frame.shape[0] - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            return processed_frame
            
        except Exception as e:
            self.logger.error(f"Error processing frame: {e}")
            # Return original frame with error message
            cv2.putText(frame, "Processing error", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            return frame

    def toggle_registration_mode(self):
        """Toggle between registration and recognition modes"""
        self.registration_mode = not self.registration_mode
        self.face_recognizer.set_registration_mode(self.registration_mode)
        
        if not self.registration_mode:
            self.load_known_faces()
        
        mode = "Registration" if self.registration_mode else "Recognition"
        self.logger.info(f"Switched to {mode} mode")
        return mode

    def run(self):
        """Run the face detection system"""
        try:
            self.logger.info("Starting face detection system...")
            self.running = True   
            
            while self.running:
                # Capture frame
                frame = self.video_capture.capture_frame(interval=0.033)  # ~30 FPS
                if frame is None:
                    continue
                
                # Process the frame
                processed_frame = self.process_frame(frame)
                
                # Display the processed frame if GUI is enabled
                if config.SHOW_VIDEO_FEED:
                    cv2.imshow('Face Detection System', processed_frame)
                
                # Handle key presses
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):  # Quit
                    self.logger.info("System stopped by user")
                    break
                elif key == ord('r'):  # Toggle mode
                    self.toggle_registration_mode()
                elif key == ord(' '):  # Space for registration trigger
                    if self.registration_mode:
                        # The FaceRecognizer will handle this internally
                        pass
                    
        except KeyboardInterrupt:
            self.logger.info("System stopped by user interrupt")
        except Exception as e:
            self.logger.error(f"Error in main loop: {e}")
        finally:
            self.cleanup()
            
    '''def run(self):
        """Run the face detection system with optimized frame processing"""
        try:
            self.logger.info("Starting face detection system...")
            self.running = True
            frame_count = 0
            import time # Assuming 'time' needs to be imported for time.time()
            start_time = time.time()
        
            while self.running:
                # Capture frame (non-blocking)
                frame = self.video_capture.capture_frame(interval=0.033) # ~30 FPS target
                if frame is None:
                    continue
            
                # Process every 3rd frame for face detection (Optimization)
                frame_count += 1
                if frame_count % 3 == 0:
                    # This is the frame where face detection/recognition runs
                    processed_frame = self.process_frame(frame)
                else:
                    # Skip heavy processing, just display the raw frame
                    processed_frame = frame 
            
                # Display the frame
                if config.SHOW_VIDEO_FEED:
                    # NOTE: Assuming 'cv2' and 'config' are accessible/imported in the actual class file
                    cv2.imshow('Face Detection System', processed_frame)
            
                # Calculate and display FPS (for logging/debugging)
                # NOTE: Assuming 'logger' (or 'self.logger') is available. Using self.logger.info.
                if frame_count % 30 == 0: # Log FPS every 30 frames for a stable reading
                    elapsed_time = time.time() - start_time
                    fps = frame_count / elapsed_time
                    self.logger.info(f"Current FPS: {fps:.1f}")
            
                # Handle key presses
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):  # Quit
                    self.logger.info("System stopped by user")
                    break
                elif key == ord('r'):  # Toggle mode
                    self.toggle_registration_mode()
                elif key == ord(' '):  # Space for registration trigger
                    if self.registration_mode:
                    # The FaceRecognizer will handle this internally
                        pass
                
        except KeyboardInterrupt:
            self.logger.info("System stopped by user interrupt")
        except Exception as e:
            self.logger.error(f"Error in main loop: {e}")
        finally:
            self.cleanup()'''    

    def cleanup(self):
        """Clean up resources"""
        try:
            if hasattr(self, 'video_capture'):
                self.video_capture.release()
            cv2.destroyAllWindows()
            self.logger.info("System cleanup completed")
        except Exception as e:
            self.logger.error(f"Error during cleanup: {e}")

def main():
    """Main entry point for the application"""
    # Check GPU status first
    gpu_available = check_gpu_status()
    print()

    parser = argparse.ArgumentParser(description='Face Detection and Recognition System')
    parser.add_argument('--register', action='store_true',
                       help='Run in registration mode to add new faces')

    args = parser.parse_args()
    
    try:
        # Print system information
        print("\n" + "="*50)
        print("Face Detection & Recognition System")
        print("="*50)
        print(f"Mode: {'Registration' if args.register else 'Recognition'}")
        print(f"Camera: {config.CAMERA_SOURCE}")
        print(f"Resolution: {config.FRAME_WIDTH}x{config.FRAME_HEIGHT}")
        print(f"Face Detection: {config.FACE_DETECTION_MODEL}")
        print("="*50)
        print("Controls:")
        print("  'q' - Quit application")
        print("  'r' - Toggle registration/recognition mode")
        print("  SPACE - Trigger face registration (in registration mode)")
        print("="*50 + "\n")
        
        # Initialize and run the system
        system = FaceDetectionSystem(registration_mode=args.register)
        system.run()
        
    except Exception as e:
        print(f"Fatal error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()  
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    


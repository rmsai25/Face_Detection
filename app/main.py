import os
import cv2
import torch
import argparse
import logging
import uvicorn
from fastapi import FastAPI, Request, status
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.exceptions import RequestValidationError
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from dotenv import load_dotenv

# === Internal imports ===
from .api.endpoints import router as api_router
from .config import config
from .database.db_handler import DatabaseHandler
from .face_processing.encoding import FaceEncoder
from .face_processing.recognition import FaceRecognizer
from .face_processing.capture import VideoCapture
from .utils.logger import setup_logger

# Logging and Environment Setup
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
load_dotenv()


# FastAPI Application Setup
app = FastAPI(
    title="Face Recognition API",
    description="API for face recognition and registration using InsightFace + SCRFD",
    version="1.0.0"
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routes
app.include_router(api_router, prefix="/api")

# Static & Templates
os.makedirs("app/static", exist_ok=True)
os.makedirs("app/templates", exist_ok=True)
app.mount("/static", StaticFiles(directory="app/static"), name="static")
templates = Jinja2Templates(directory="app/templates")

# Error Handlers
@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content={"detail": exc.errors(), "body": str(exc.body) if exc.body else None},
    )

@app.exception_handler(500)
async def internal_exception_handler(request: Request, exc: Exception):
    logger.error(f"Internal server error: {str(exc)}")
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={"detail": "Internal server error"},
    )

# Root Page
@app.get("/", response_class=HTMLResponse)
async def root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

# Health Check
@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "model": "InsightFace + SCRFD",
        "gpu_available": torch.cuda.is_available()
    }

def check_gpu_status():
    """Check and display GPU availability for InsightFace."""
    print("🔍 Checking GPU Status for InsightFace...")
    print(f"CUDA Available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
        print(f"CUDA Version: {torch.version.cuda}")
        print("✅ GPU acceleration enabled for InsightFace!")
    else:
        print("❌ No GPU detected - running InsightFace on CPU")
        print("Note: InsightFace uses ONNX Runtime and will use CPU providers")
    print(f"PyTorch Version: {torch.__version__}")
    return torch.cuda.is_available()


# Face Detection System
class FaceDetectionSystem:
    def __init__(self, registration_mode=False):
        self.logger = setup_logger(
            level=config.LOG_LEVEL,
            enable_file=False,
            log_dir=None,
            enable_console=True
        )

        self.registration_mode = registration_mode
        self.running = False

        try:
            self.db = DatabaseHandler(config.DATABASE_URL)
            self.face_encoder = FaceEncoder()
            self.face_recognizer = FaceRecognizer(
                face_encoder=self.face_encoder,
                registration_mode=self.registration_mode,
                create_dirs=False
            )

            if not self.registration_mode:
                self.load_known_faces()

            self.video_capture = VideoCapture(
                source=config.CAMERA_SOURCE,
                width=config.FRAME_WIDTH,
                height=config.FRAME_HEIGHT
            )
            if not self.video_capture.initialize():
                raise RuntimeError("Failed to initialize camera")

            self.logger.info(f"Initialized in {'registration' if self.registration_mode else 'recognition'} mode using InsightFace + SCRFD")

        except Exception as e:
            self.logger.error(f"Initialization error: {e}")
            raise

    def load_known_faces(self):
        try:
            users = self.db.get_all_users_with_encodings(include_encoding=True)
            if not users:
                self.logger.warning("No users with encodings found")
                return
            self.face_recognizer.load_known_faces(users)
            self.logger.info(f"Loaded {len(users)} faces from DB using InsightFace embeddings")
        except Exception as e:
            self.logger.error(f"Error loading faces: {e}")

    def process_frame(self, frame):
        try:
            processed_frame, _ = self.face_recognizer.process_frame(frame, self.db)
            mode_text = "Mode: Registration (InsightFace)" if self.registration_mode else "Mode: Recognition (InsightFace)"
            cv2.putText(processed_frame, mode_text, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
            
            # Add model info to display
            model_text = "Model: InsightFace + SCRFD"
            cv2.putText(processed_frame, model_text, (10, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            return processed_frame
        except Exception as e:
            self.logger.error(f"Frame processing error: {e}")
            cv2.putText(frame, "Processing error", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            return frame

    def toggle_registration_mode(self):
        self.registration_mode = not self.registration_mode
        self.face_recognizer.set_registration_mode(self.registration_mode)
        if not self.registration_mode:
            self.load_known_faces()
        mode = "Registration" if self.registration_mode else "Recognition"
        self.logger.info(f"Toggled to {mode} mode using InsightFace")
        return mode

    def run(self):
        try:
            self.logger.info("🎥 Starting Face Detection System with InsightFace + SCRFD...")
            self.running = True
            while self.running:
                frame = self.video_capture.capture_frame(interval=0.033)
                if frame is None:
                    continue
                processed_frame = self.process_frame(frame)
                if config.SHOW_VIDEO_FEED:
                    cv2.imshow('Face Detection System - InsightFace + SCRFD', processed_frame)

                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    self.logger.info("Stopped by user")
                    break
                elif key == ord('r'):
                    self.toggle_registration_mode()
                elif key == ord('i'):
                    # Display model info
                    self.logger.info("Model: InsightFace with SCRFD detector and ArcFace recognizer")
            self.cleanup()

        except Exception as e:
            self.logger.error(f"Runtime error: {e}")
            self.cleanup()

    def cleanup(self):
        try:
            if hasattr(self, 'video_capture'):
                self.video_capture.release()
            cv2.destroyAllWindows()
            self.logger.info("Cleanup complete")
        except Exception as e:
            self.logger.error(f"Cleanup error: {e}")


# Model Information

def print_model_info():
    """Print information about the face recognition model."""
    print("\n" + "="*50)
    print("🤖 FACE RECOGNITION SYSTEM - MODEL INFORMATION")
    print("="*50)
    print("Model: InsightFace")
    print("="*50)
    print()

def main():
    parser = argparse.ArgumentParser(description="Face Recognition System using InsightFace + SCRFD")
    parser.add_argument("--mode", choices=["server", "realtime"], default="server",
                        help="Run mode: 'server' (FastAPI) or 'realtime' (Camera recognition)")
    parser.add_argument("--register", action="store_true", help="Enable registration mode in realtime system")
    args = parser.parse_args()

    if args.mode == "server":
        print("🚀 Starting FastAPI Server with InsightFace + SCRFD...")
        print_model_info()
        uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)
    else:
        check_gpu_status()
        print_model_info()
        print("\n🎥 Starting Real-Time Face Recognition with InsightFace + SCRFD")
        print("Controls:")
        print("  Press 'q' to quit")
        print("  Press 'r' to toggle between registration/recognition mode")
        print("  Press 'i' to display model information")
        print()
        system = FaceDetectionSystem(registration_mode=args.register)
        system.run()

if __name__ == "__main__":
    main()

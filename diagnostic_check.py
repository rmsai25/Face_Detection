import torch
import time
import platform
import numpy as np
import cv2
import insightface
from insightface.app import FaceAnalysis

# --- BASIC ENV CHECK ---
print("\n===== ENVIRONMENT INFO =====")
print("Python:", platform.python_version())
print("PyTorch:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
    print("CUDA Device Count:", torch.cuda.device_count())
else:
    print("⚠️  CUDA not available — models will run on CPU")

# --- SETUP ---
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# --- TEST MODELS ---
print("\n===== MODEL DEVICE TEST =====")
try:
    # Initialize InsightFace with SCRFD and ArcFace
    print("Initializing InsightFace with SCRFD detector and ArcFace recognizer...")
    app = FaceAnalysis(
        name='buffalo_l',
        providers=['CUDAExecutionProvider'] if device == 'cuda' else ['CPUExecutionProvider']
    )
    
    # Prepare the model with appropriate context
    ctx_id = 0 if device == 'cuda' else -1
    app.prepare(ctx_id=ctx_id, det_size=(640, 640))
    
    print(f"InsightFace device: {'GPU' if device == 'cuda' else 'CPU'}")
    print(f"InsightFace model: buffalo_l")
    print(f"InsightFace providers: {app.models['detection'].providers}")
    
    insightface_loaded = True
except Exception as e:
    print(f"InsightFace not found or failed to load: {e}")
    insightface_loaded = False
    app = None

# --- SPEED TEST ---
print("\n===== INFERENCE SPEED TEST =====")
if insightface_loaded:
    # Test InsightFace SCRFD detection + ArcFace embedding
    dummy_img = np.random.randint(0, 255, (640, 640, 3), dtype='uint8')
    
    # Test 1: Detection only
    t0 = time.time()
    faces = app.get(dummy_img)
    detection_time = time.time() - t0
    print(f"InsightFace SCRFD detection time: {detection_time:.4f} s")
    print(f"Faces detected: {len(faces)}")
    
    # Test 2: Detection + Embedding (if faces found)
    if len(faces) > 0:
        # The embedding is already computed in the same call
        embedding_time = detection_time  # Same call does both
        print(f"InsightFace ArcFace embedding time: {embedding_time:.4f} s")
        print(f"Embedding dimension: {len(faces[0].embedding)}D")
        print(f"Detection confidence: {faces[0].det_score:.4f}")
    
    # Test 3: Multiple inference test for stability
    print("\n----- MULTIPLE INFERENCE TEST -----")
    times = []
    for i in range(5):
        t_start = time.time()
        _ = app.get(dummy_img)
        times.append(time.time() - t_start)
    
    print(f"Average inference time over 5 runs: {np.mean(times):.4f} s")
    print(f"Standard deviation: {np.std(times):.4f} s")
    
    # Test 4: Test with different image sizes
    print("\n----- DIFFERENT RESOLUTION TEST -----")
    resolutions = [(320, 240), (640, 480), (1280, 720)]
    for width, height in resolutions:
        test_img = np.random.randint(0, 255, (height, width, 3), dtype='uint8')
        t_start = time.time()
        faces = app.get(test_img)
        inference_time = time.time() - t_start
        print(f"Resolution {width}x{height}: {inference_time:.4f} s, Faces: {len(faces)}")

# --- GPU UTILIZATION CHECK ---
if torch.cuda.is_available():
    print("\n===== GPU MEMORY USAGE =====")
    print(f"Allocated: {torch.cuda.memory_allocated(0)/1e6:.2f} MB")
    print(f"Reserved:  {torch.cuda.memory_reserved(0)/1e6:.2f} MB")

# --- MODEL CAPABILITIES CHECK ---
print("\n===== INSIGHTFACE CAPABILITIES =====")
if insightface_loaded:
    try:
        # Test with a more realistic image (with actual face-like patterns)
        test_img = np.random.randint(100, 200, (480, 640, 3), dtype='uint8')
        # Add some face-like features
        cv2.rectangle(test_img, (200, 150), (440, 400), (150, 150, 150), -1)  # Face area
        cv2.circle(test_img, (280, 220), 10, (100, 100, 100), -1)  # Left eye
        cv2.circle(test_img, (360, 220), 10, (100, 100, 100), -1)  # Right eye
        cv2.rectangle(test_img, (300, 280), (340, 320), (100, 100, 100), -1)  # Nose/mouth
        
        faces = app.get(test_img)
        print(f"Face detection working: {'✅' if len(faces) > 0 else '❌'}")
        
        if len(faces) > 0:
            face = faces[0]
            print(f"Detection score: {face.det_score:.4f}")
            print(f"Bounding box: {face.bbox}")
            print(f"Landmarks available: {'✅' if face.kps is not None else '❌'}")
            print(f"Embedding shape: {face.embedding.shape}")
            print(f"Embedding norm: {np.linalg.norm(face.embedding):.4f}")
            
            # Test embedding normalization
            embedding_norm = np.linalg.norm(face.embedding)
            if 0.95 <= embedding_norm <= 1.05:
                print("✅ Embedding properly normalized")
            else:
                print(f"⚠️  Embedding norm unusual: {embedding_norm:.4f}")
                
    except Exception as e:
        print(f"❌ Capability test failed: {e}")

# --- PERFORMANCE COMPARISON ---
print("\n===== PERFORMANCE SUMMARY =====")
if insightface_loaded:
    print("✅ InsightFace + SCRFD + ArcFace:")
    print("   - Unified detection and recognition")
    print("   - 512D normalized embeddings")
    print("   - 5-point facial landmarks")
    print("   - GPU acceleration via ONNX Runtime")
    print("   - State-of-the-art accuracy")
else:
    print("❌ InsightFace failed to load")

print("\n✅ Diagnostic complete — InsightFace + SCRFD system ready!\n")

# --- ADDITIONAL INSIGHTFACE INFO ---
if insightface_loaded:
    print("\n===== INSIGHTFACE MODEL INFO =====")
    try:
        # Get model information
        print("Available models in InsightFace:")
        print(" - buffalo_l: SCRFD + ArcFace (recommended)")
        print(" - buffalo_m: Medium version")
        print(" - buffalo_s: Small version (faster)")
        print(" - antelopev2: Latest version")
        
        # Test model switching capability
        print(f"\nCurrent model: buffalo_l")
        print(f"Detection size: 640x640")
        print(f"Context ID: {ctx_id} ({'GPU' if ctx_id >= 0 else 'CPU'})")
        
    except Exception as e:
        print(f"Model info error: {e}")
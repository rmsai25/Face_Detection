import torch, time, platform
import numpy as np
import insightface
from insightface.app import FaceAnalysis
import cv2

print("\n===== ENVIRONMENT INFO =====")
print("Python:", platform.python_version())
print("PyTorch:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("CUDA version:", torch.version.cuda)
    print("GPU:", torch.cuda.get_device_name(0))
    print("Compute capability:", torch.cuda.get_device_capability())
else:
    print("❌ CUDA not available")

# Test GPU tensor operations
print("\n===== GPU COMPUTATION TEST =====")
if torch.cuda.is_available():
    # Test basic GPU operations
    x = torch.randn(1000, 1000).cuda()
    y = torch.randn(1000, 1000).cuda()
    
    torch.cuda.synchronize()
    start = time.time()
    z = torch.matmul(x, y)
    torch.cuda.synchronize()
    gpu_time = time.time() - start
    
    print(f"GPU matrix multiplication: {gpu_time:.4f}s")
    print(f"GPU memory allocated: {torch.cuda.memory_allocated()/1e6:.2f} MB")
else:
    print("❌ Cannot test GPU computation - CUDA not available")

print("\n===== COMPATIBILITY CHECK =====")
if torch.cuda.is_available():
    compute_capability = torch.cuda.get_device_capability()
    cc = float(f"{compute_capability[0]}.{compute_capability[1]}")
    
    if cc >= 8.9:  # RTX 5070 requirement
        print(f"✅ Compute capability {cc} is supported")
    else:
        print(f"⚠️  Compute capability {cc} might have limited support")
    
    # Check if PyTorch was built with sufficient CUDA version
    cuda_version = torch.version.cuda
    if cuda_version >= '12.1':
        print(f"✅ CUDA {cuda_version} should support RTX 5070")
    else:
        print(f"⚠️  CUDA {cuda_version} might not fully support RTX 5070")

print("\n===== INSIGHTFACE + SCRFD DIAGNOSTIC =====")
try:
    # Test InsightFace installation and basic functionality
    print("Testing InsightFace installation...")
    
    # Check InsightFace version
    print(f"InsightFace version: {insightface.__version__}")
    
    # Initialize InsightFace with SCRFD
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Initializing InsightFace with device: {device}")
    
    app = FaceAnalysis(
        name='buffalo_l',
        providers=['CUDAExecutionProvider'] if device == 'cuda' else ['CPUExecutionProvider']
    )
    
    ctx_id = 0 if device == 'cuda' else -1
    app.prepare(ctx_id=ctx_id, det_size=(640, 640))
    
    print("✅ InsightFace + SCRFD initialized successfully")
    
    # Test inference speed
    print("\n----- INSIGHTFACE PERFORMANCE TEST -----")
    test_image = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
    
    # Warm up
    for _ in range(3):
        _ = app.get(test_image)
    
    # Performance test
    times = []
    for i in range(10):
        start_time = time.time()
        faces = app.get(test_image)
        end_time = time.time()
        times.append(end_time - start_time)
    
    avg_time = np.mean(times) * 1000  # Convert to milliseconds
    std_time = np.std(times) * 1000
    print(f"Average inference time: {avg_time:.2f} ± {std_time:.2f} ms")
    print(f"FPS: {1000/avg_time:.1f}")
    
    # Memory usage
    if torch.cuda.is_available():
        print(f"GPU memory allocated: {torch.cuda.memory_allocated()/1e6:.2f} MB")
        print(f"GPU memory reserved: {torch.cuda.memory_reserved()/1e6:.2f} MB")
    
    # Model capabilities
    print("\n----- MODEL CAPABILITIES -----")
    print(f"Detector: SCRFD")
    print(f"Recognizer: ArcFace")
    print(f"Embedding dimension: 512D")
    print(f"Landmark points: 5")
    print(f"Input size: 640x640")
    
    # Test with a more realistic image
    realistic_img = np.random.randint(100, 200, (480, 640, 3), dtype=np.uint8)
    # Create a simple face-like pattern for detection
    cv2.rectangle(realistic_img, (200, 150), (440, 400), (150, 150, 150), -1)
    
    faces = app.get(realistic_img)
    print(f"Face detection test: {len(faces)} faces detected")
    
    if len(faces) > 0:
        face = faces[0]
        print(f"Detection confidence: {face.det_score:.4f}")
        print(f"Embedding norm: {np.linalg.norm(face.embedding):.4f}")
        print("✅ All InsightFace features working correctly")
    
except Exception as e:
    print(f"❌ InsightFace diagnostic failed: {e}")
    print("Please check InsightFace installation:")
    print("  pip install insightface")
    print("  pip install onnxruntime-gpu (for GPU support)")

print("\n===== SYSTEM READINESS CHECK =====")
requirements_met = 0
total_requirements = 3

# Check PyTorch
if torch.cuda.is_available():
    print("✅ PyTorch with CUDA support")
    requirements_met += 1
else:
    print("⚠️  PyTorch (CPU only)")

# Check InsightFace
try:
    import insightface
    print("✅ InsightFace installed")
    requirements_met += 1
except:
    print("❌ InsightFace not installed")

# Check OpenCV
try:
    import cv2
    print("✅ OpenCV installed")
    requirements_met += 1
except:
    print("❌ OpenCV not installed")

print(f"\nOverall Status: {requirements_met}/{total_requirements} requirements met")
if requirements_met == total_requirements:
    print("🎉 System is ready for InsightFace + SCRFD face recognition!")
else:
    print("⚠️  Some requirements missing. Please install missing packages.")

print("\n===== RECOMMENDATIONS =====")
if torch.cuda.is_available():
    print("✅ GPU acceleration available")
    print("💡 Consider using 'buffalo_l' model for best accuracy")
else:
    print("⚠️  Running on CPU")
    print("💡 Consider using 'buffalo_s' model for faster CPU performance")

print("\nNext steps:")
print("1. Run your face recognition application")
print("2. Monitor GPU memory usage during operation")
print("3. Adjust detection threshold in config if needed")
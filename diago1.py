import torch, time, platform
import numpy as np

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
        
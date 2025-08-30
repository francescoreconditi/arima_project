from arima_forecaster.config import detect_gpu_capability, get_config

# Verifica capacità GPU
capability = detect_gpu_capability()
print(f"CUDA Available: {capability.has_cuda}")
print(f"GPU Name: {capability.gpu_name}")
if capability.gpu_memory is not None:
    print(f"GPU Memory: {capability.gpu_memory:.1f}GB")
else:
    print(f"GPU Memory: None")
print(f"CUDA Version: {capability.cuda_version}")

# Config
config = get_config()
print(f"Backend: {config.backend}")
print(f"CUDA Device: {config.cuda_device}")
print(f"Max Parallel: {config.max_gpu_models_parallel}")

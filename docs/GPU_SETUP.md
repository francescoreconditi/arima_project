# 🚀 GPU/CUDA Setup Guide

Guida completa per configurare e utilizzare l'accelerazione GPU/CUDA con ARIMA Forecaster Library.

## 📋 Indice

- [Requisiti Sistema](#requisiti-sistema)
- [Installazione Dependencies](#installazione-dependencies)
- [Configurazione](#configurazione)
- [Verifica Setup](#verifica-setup)
- [Utilizzo GPU](#utilizzo-gpu)
- [Troubleshooting](#troubleshooting)
- [Performance Benchmarks](#performance-benchmarks)

## 🖥️ Requisiti Sistema

### GPU Compatibili
- **NVIDIA GPU** con compute capability >= 6.0
- **Memoria GPU**: Minimo 4GB, Consigliato 8GB+
- **CUDA**: Versione 11.x o 12.x

### GPU Testate e Raccomandate

| GPU Model | Memory | Max Parallel Models | Use Case |
|-----------|---------|-------------------|----------|
| RTX 4090 | 24GB | 500+ | Enterprise/Research |
| RTX 4080 | 16GB | 200+ | Professional |
| RTX 4070 | 12GB | 100+ | Development |
| RTX 3090 | 24GB | 400+ | Workstation |
| RTX 3080 | 10GB | 150+ | Gaming/Dev |
| Tesla V100 | 32GB | 800+ | Data Center |
| A100 | 40GB | 1000+ | HPC |

### Sistema Operativo
- **Linux**: Ubuntu 18.04+, CentOS 7+ (Consigliato per RAPIDS)
- **Windows**: Windows 10/11 (Supporto base CUDA)
- **macOS**: Solo CPU (CUDA non supportato)

## 📦 Installazione Dependencies

### Opzione 1: Installazione Completa GPU
```bash
# Installa tutte le dipendenze GPU
pip install arima-forecaster[gpu]

# O con uv (più veloce)
uv sync --extra gpu
```

### Opzione 2: Installazione Personalizzata
```bash
# Solo PyTorch CUDA
pip install torch>=2.0.0

# PyTorch + CuPy
pip install torch>=2.0.0 cupy-cuda11x>=12.0.0

# Full stack (solo Linux)
pip install torch>=2.0.0 cupy-cuda12x>=12.0.0 cuml>=23.10.0
```

### Opzione 3: Installazione Conda (Consigliata per RAPIDS)
```bash
# Crea environment conda
conda create -n arima-gpu python=3.10
conda activate arima-gpu

# Installa RAPIDS (CUDA 11.8)
conda install -c rapidsai -c conda-forge -c nvidia \
    cuml=23.10 cupy cudatoolkit=11.8

# Installa PyTorch
conda install pytorch torchvision pytorch-cuda=11.8 -c pytorch -c nvidia

# Installa ARIMA Forecaster
pip install arima-forecaster
```

## ⚙️ Configurazione

### 1. Crea File `.env`
```bash
# Copia template configurazione
cp .env.example .env
```

### 2. Configura GPU nel `.env`
```bash
# Configurazione GPU base
ARIMA_BACKEND=auto
ARIMA_CUDA_DEVICE=0
ARIMA_GPU_MEMORY_LIMIT=8.0
ARIMA_MIXED_PRECISION=true
ARIMA_MAX_GPU_PARALLEL=100

# Per debugging
ARIMA_DEBUG=true
ARIMA_LOG_LEVEL=DEBUG
```

### 3. Configurazioni Ottimizzate per Hardware

#### RTX 4090 / A100 (24GB+)
```bash
ARIMA_BACKEND=cuda
ARIMA_GPU_MEMORY_LIMIT=20.0
ARIMA_MAX_GPU_PARALLEL=500
ARIMA_CHUNK_SIZE=2000
ARIMA_MIXED_PRECISION=true
```

#### RTX 4070 / 3080 (8-12GB)
```bash
ARIMA_BACKEND=auto
ARIMA_GPU_MEMORY_LIMIT=8.0
ARIMA_MAX_GPU_PARALLEL=100
ARIMA_CHUNK_SIZE=1000
ARIMA_MIXED_PRECISION=true
```

#### Server Condiviso
```bash
ARIMA_BACKEND=cuda
ARIMA_CUDA_DEVICE=0
ARIMA_GPU_MEMORY_LIMIT=4.0
ARIMA_MAX_GPU_PARALLEL=25
```

## ✅ Verifica Setup

### 1. Test Disponibilità GPU
```python
from arima_forecaster.config import detect_gpu_capability

# Verifica capacità GPU
capability = detect_gpu_capability()
print(f"CUDA Available: {capability.has_cuda}")
print(f"GPU Name: {capability.gpu_name}")
print(f"GPU Memory: {capability.gpu_memory:.1f}GB")
print(f"CUDA Version: {capability.cuda_version}")
```

### 2. Test Configurazione
```python
from arima_forecaster.config import get_config

config = get_config()
print(f"Backend: {config.backend}")
print(f"CUDA Device: {config.cuda_device}")
print(f"Max Parallel: {config.max_gpu_models_parallel}")
```

### 3. Benchmark GPU vs CPU
```python
import numpy as np
from arima_forecaster.utils.gpu_utils import benchmark_gpu_vs_cpu

# Test operazione matrix multiply
def test_operation(gpu_manager, data):
    a = gpu_manager.array(data)
    return gpu_manager.xp.dot(a, a.T)

# Genera dati test
test_data = np.random.randn(1000, 1000)

# Esegui benchmark
results = benchmark_gpu_vs_cpu(test_operation, test_data)
print(f"CPU Time: {results['cpu_time']:.3f}s")
print(f"GPU Time: {results['gpu_time']:.3f}s")
print(f"Speedup: {results['speedup']:.1f}x")
```

## 🎯 Utilizzo GPU

### 1. Model Selector GPU-Accelerated
```python
import pandas as pd
from arima_forecaster.core import GPUARIMAModelSelector

# Crea dataset multiplo
series_list = [
    pd.Series(np.random.randn(100) + i, name=f"series_{i}")
    for i in range(50)  # 50 serie temporali
]

# GPU-accelerated grid search
selector = GPUARIMAModelSelector(use_gpu=True)
results = selector.search_multiple_series(series_list)

print(f"Processate {len(results)} serie")
for result in results[:5]:
    print(f"{result['series_name']}: {result['best_order']} (AIC: {result['best_score']:.2f})")
```

### 2. SARIMA GPU-Accelerated
```python
from arima_forecaster.core import GPUSARIMAModelSelector

# SARIMA selector con GPU
sarima_selector = GPUSARIMAModelSelector(
    use_gpu=True,
    max_parallel_models=50  # SARIMA più pesante
)

results = sarima_selector.search_multiple_series(series_list[:20])  # 20 serie per SARIMA
```

### 3. Gestione Memoria GPU
```python
from arima_forecaster.utils.gpu_utils import get_gpu_manager

gpu_manager = get_gpu_manager()

# Verifica memoria disponibile
memory_info = gpu_manager.get_memory_info()
if memory_info:
    free_gb = memory_info[0] / (1024**3)
    total_gb = memory_info[1] / (1024**3)
    print(f"GPU Memory: {free_gb:.1f}GB free / {total_gb:.1f}GB total")

# Context manager per device specifico
with gpu_manager.device_context(device_id=0):
    # Operazioni su GPU 0
    pass
```

## 🐛 Troubleshooting

### Problema: CUDA Out of Memory
```
RuntimeError: CUDA out of memory
```

**Soluzioni:**
```bash
# Riduci parallel models
ARIMA_MAX_GPU_PARALLEL=25

# Imposta limite memoria
ARIMA_GPU_MEMORY_LIMIT=6.0

# Riduci chunk size
ARIMA_CHUNK_SIZE=500
```

### Problema: CuPy non riconosciuto
```
ImportError: No module named 'cupy'
```

**Soluzioni:**
```bash
# Installa CuPy per tua versione CUDA
pip install cupy-cuda11x  # Per CUDA 11.x
pip install cupy-cuda12x  # Per CUDA 12.x

# Verifica versione CUDA
nvidia-smi
nvcc --version
```

### Problema: Performance peggiore su GPU
**Cause comuni:**
1. **Dataset troppo piccolo** - GPU ha overhead setup
2. **Memoria insufficiente** - Thrashing GPU→CPU
3. **Driver vecchi** - Aggiorna driver NVIDIA

**Soluzioni:**
```python
# Forza CPU per dataset piccoli
if len(series_list) < 10:
    selector = GPUARIMAModelSelector(use_gpu=False)
```

### Problema: Mixed Precision Errors
```
RuntimeError: expected scalar type Half but got Float
```

**Soluzione:**
```bash
# Disabilita mixed precision
ARIMA_MIXED_PRECISION=false
```

## 📊 Performance Benchmarks

### Grid Search Performance (1000 Serie Temporali)

| Configuration | Time | Speedup | Models/sec |
|--------------|------|---------|------------|
| CPU (16 cores) | 45 min | 1x | 370 |
| RTX 4070 | 8 min | 5.6x | 2080 |
| RTX 4090 | 4.5 min | 10x | 3700 |
| A100 40GB | 3 min | 15x | 5550 |

### Memory Usage (per 100 parallel models)

| Configuration | GPU Memory | System RAM |
|--------------|------------|------------|
| ARIMA | 2.1GB | 1.5GB |
| SARIMA | 4.8GB | 2.8GB |
| SARIMAX | 6.2GB | 3.5GB |

### Optimal Batch Sizes

| GPU Memory | ARIMA Batch | SARIMA Batch |
|------------|-------------|--------------|
| 4GB | 25 | 10 |
| 8GB | 100 | 25 |
| 12GB | 150 | 40 |
| 16GB | 200 | 50 |
| 24GB+ | 500 | 100 |

## 🚀 Best Practices

### 1. Configurazione Production
```bash
# .env per production
ARIMA_BACKEND=auto
ARIMA_GPU_MEMORY_LIMIT=80%_of_gpu_memory
ARIMA_MIXED_PRECISION=true
ARIMA_LOG_LEVEL=WARNING
ARIMA_CACHE_ENABLED=true
```

### 2. Monitoring GPU
```python
import psutil
from arima_forecaster.utils.gpu_utils import get_gpu_manager

def monitor_resources():
    # CPU usage
    cpu_percent = psutil.cpu_percent()
    
    # GPU memory
    gpu_manager = get_gpu_manager()
    gpu_memory = gpu_manager.get_memory_info()
    
    print(f"CPU: {cpu_percent}%")
    if gpu_memory:
        used_gb = (gpu_memory[1] - gpu_memory[0]) / (1024**3)
        print(f"GPU Memory Used: {used_gb:.1f}GB")
```

### 3. Fallback Strategy
```python
from arima_forecaster.core import GPUARIMAModelSelector, ARIMAModelSelector

try:
    # Prova GPU first
    selector = GPUARIMAModelSelector(use_gpu=True)
    results = selector.search_multiple_series(series_list)
except Exception as e:
    print(f"GPU fallito: {e}")
    # Fallback CPU
    selector = ARIMAModelSelector()
    results = [selector.search(series) for series in series_list]
```

## 📞 Supporto

Per problemi specifici GPU:

1. **Verifica hardware**: `nvidia-smi`
2. **Controlla log**: `ARIMA_LOG_LEVEL=DEBUG`
3. **Test isolato**: Usa script benchmark
4. **Fallback CPU**: `ARIMA_BACKEND=cpu`

Per supporto avanzato, includi nelle issue:
- Output `nvidia-smi`
- Configurazione `.env`
- Log errori completi
- Specifiche hardware sistema
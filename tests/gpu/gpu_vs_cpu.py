import time
import numpy as np
import cupy as cp
from arima_forecaster.utils.gpu_utils import GPUArrayManager, benchmark_gpu_vs_cpu

print("=== Confronto Benchmark: Libreria vs Diretto ===\n")

# Test con diverse dimensioni di matrici
sizes = [500, 1000, 2000, 3000, 5000]

for size in sizes:
    print(f"Matrice {size}x{size}:")
    print("-" * 40)
    
    # Genera dati test
    test_data = np.random.randn(size, size).astype(np.float32)
    
    # === METODO 1: Usando la funzione della libreria ===
    print("  [Libreria] benchmark_gpu_vs_cpu:")
    
    def test_operation(gpu_manager, data):
        a = gpu_manager.array(data)
        return gpu_manager.xp.dot(a, a.T)
    
    results = benchmark_gpu_vs_cpu(test_operation, test_data, iterations=3)
    
    print(f"    CPU Time: {results['cpu_time']:.3f}s")
    print(f"    GPU Time: {results['gpu_time']:.3f}s") 
    print(f"    Speedup: {results['speedup']:.1f}x")
    
    # === METODO 2: Benchmark diretto ottimizzato ===
    print("\n  [Ottimizzato] Benchmark diretto:")
    
    # Pre-carica su GPU
    gpu_data = cp.asarray(test_data)
    
    # Warm-up GPU (importante!)
    _ = cp.dot(gpu_data, gpu_data.T)
    cp.cuda.Stream.null.synchronize()
    
    # CPU benchmark
    cpu_times = []
    for _ in range(3):
        start = time.perf_counter()
        _ = np.dot(test_data, test_data.T)
        cpu_times.append(time.perf_counter() - start)
    cpu_time = np.mean(cpu_times)
    
    # GPU benchmark (dati già su GPU, no transfer overhead)
    gpu_times = []
    for _ in range(3):
        start = time.perf_counter()
        _ = cp.dot(gpu_data, gpu_data.T)
        cp.cuda.Stream.null.synchronize()
        gpu_times.append(time.perf_counter() - start)
    gpu_time = np.mean(gpu_times)
    
    speedup = cpu_time / gpu_time
    
    print(f"    CPU Time: {cpu_time:.3f}s")
    print(f"    GPU Time: {gpu_time:.3f}s")
    print(f"    Speedup: {speedup:.1f}x")
    
    # === METODO 3: Usando GPUArrayManager correttamente ===
    print("\n  [Manager] GPUArrayManager pre-inizializzato:")
    
    # Inizializza manager UNA SOLA VOLTA fuori dal timing
    gpu_manager = GPUArrayManager(backend="cuda")
    cpu_manager = GPUArrayManager(backend="cpu")
    
    # Pre-carica dati e warm-up
    gpu_array = gpu_manager.array(test_data)
    _ = gpu_manager.xp.dot(gpu_array, gpu_array.T)
    gpu_manager.synchronize()
    
    # CPU benchmark con manager
    cpu_times = []
    cpu_array = cpu_manager.array(test_data)
    for _ in range(3):
        start = time.perf_counter()
        _ = cpu_manager.xp.dot(cpu_array, cpu_array.T)
        cpu_times.append(time.perf_counter() - start)
    cpu_time_mgr = np.mean(cpu_times)
    
    # GPU benchmark con manager (no re-init)
    gpu_times = []
    for _ in range(3):
        start = time.perf_counter()
        _ = gpu_manager.xp.dot(gpu_array, gpu_array.T)
        gpu_manager.synchronize()
        gpu_times.append(time.perf_counter() - start)
    gpu_time_mgr = np.mean(gpu_times)
    
    speedup_mgr = cpu_time_mgr / gpu_time_mgr
    
    print(f"    CPU Time: {cpu_time_mgr:.3f}s")
    print(f"    GPU Time: {gpu_time_mgr:.3f}s")
    print(f"    Speedup: {speedup_mgr:.1f}x")
    
    print("\n  [Analisi]:")
    if speedup > 1:
        print(f"    -> GPU realmente {speedup:.1f}x più veloce (metodo ottimizzato)")
    
    overhead = results['gpu_time'] - gpu_time if results['gpu_time'] else 0
    if overhead > 0:
        print(f"    -> Overhead libreria: {overhead:.3f}s ({overhead/results['gpu_time']*100:.0f}% del tempo totale)")
    
    print("=" * 40)
    print()
    
    # Cleanup memoria GPU
    del gpu_data, gpu_array
    cp.get_default_memory_pool().free_all_blocks()
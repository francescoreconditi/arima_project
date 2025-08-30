import time
import numpy as np
import cupy as cp

print("=== Benchmark Avanzato GPU vs CPU ===\n")

# Test con diverse dimensioni
sizes = [1000, 2000, 3000, 4000, 5000]

for size in sizes:
    print(f"Matrice {size}x{size}:")
    
    # Genera dati
    cpu_data = np.random.randn(size, size).astype(np.float32)
    
    # Pre-carica su GPU per evitare overhead trasferimento nel timing
    gpu_data = cp.asarray(cpu_data)
    
    # Warm-up GPU (prima esecuzione più lenta)
    _ = cp.dot(gpu_data, gpu_data.T)
    cp.cuda.Stream.null.synchronize()
    
    # Benchmark CPU
    start = time.perf_counter()
    cpu_result = np.dot(cpu_data, cpu_data.T)
    cpu_time = time.perf_counter() - start
    
    # Benchmark GPU (dati già su GPU)
    start = time.perf_counter()
    gpu_result = cp.dot(gpu_data, gpu_data.T)
    cp.cuda.Stream.null.synchronize()  # Aspetta che GPU finisca
    gpu_time = time.perf_counter() - start
    
    # Calcola speedup
    speedup = cpu_time / gpu_time
    
    print(f"  CPU Time: {cpu_time:.3f}s")
    print(f"  GPU Time: {gpu_time:.3f}s")
    print(f"  Speedup: {speedup:.1f}x")
    
    if speedup > 1:
        print(f"  [OK] GPU {speedup:.1f}x più veloce!")
    else:
        print(f"  [INFO] CPU più veloce per questa dimensione")
    
    # Verifica correttezza (primi 5 elementi)
    gpu_result_cpu = cp.asnumpy(gpu_result)
    max_diff = np.max(np.abs(cpu_result - gpu_result_cpu))
    print(f"  Max differenza: {max_diff:.2e}")
    
    print()
    
    # Libera memoria GPU
    del gpu_data, gpu_result
    cp.get_default_memory_pool().free_all_blocks()

print("\n=== Test con Operazioni Multiple ===\n")

# Test operazioni concatenate
size = 3000
print(f"Matrice {size}x{size} - 5 moltiplicazioni concatenate:")

cpu_data = np.random.randn(size, size).astype(np.float32)
gpu_data = cp.asarray(cpu_data)

# Warm-up
temp = gpu_data
for _ in range(5):
    temp = cp.dot(temp, gpu_data.T)
cp.cuda.Stream.null.synchronize()

# CPU benchmark
start = time.perf_counter()
temp = cpu_data
for _ in range(5):
    temp = np.dot(temp, cpu_data.T)
cpu_time = time.perf_counter() - start

# GPU benchmark  
start = time.perf_counter()
temp = gpu_data
for _ in range(5):
    temp = cp.dot(temp, gpu_data.T)
cp.cuda.Stream.null.synchronize()
gpu_time = time.perf_counter() - start

speedup = cpu_time / gpu_time
print(f"  CPU Time: {cpu_time:.3f}s")
print(f"  GPU Time: {gpu_time:.3f}s")
print(f"  Speedup: {speedup:.1f}x")
if speedup > 1:
    print(f"  [OK] GPU {speedup:.1f}x più veloce!")
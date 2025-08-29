"""
GPU/CUDA Configuration e Backend Detection.
Gestisce il rilevamento automatico delle capacità GPU e la configurazione dei backend.
"""

import os
import logging
from enum import Enum
from typing import Optional, Dict, Any, Tuple
from dataclasses import dataclass

from ..utils.logger import get_logger

logger = get_logger(__name__)

class GPUBackend(Enum):
    """Enum per i backend supportati."""
    CPU = "cpu"
    CUDA = "cuda"
    CUPY = "cupy"
    AUTO = "auto"

@dataclass
class GPUCapability:
    """Informazioni sulle capacità GPU del sistema."""
    has_cuda: bool = False
    has_cupy: bool = False
    has_torch_cuda: bool = False
    has_cuml: bool = False
    cuda_devices: int = 0
    gpu_memory: Optional[float] = None  # GB
    gpu_name: Optional[str] = None
    cuda_version: Optional[str] = None
    compute_capability: Optional[Tuple[int, int]] = None

def detect_gpu_capability() -> GPUCapability:
    """
    Rileva automaticamente le capacità GPU del sistema.
    
    Returns:
        GPUCapability: Informazioni dettagliate sulle capacità GPU
    """
    capability = GPUCapability()
    
    # Test CUDA availability with PyTorch
    try:
        import torch
        if torch.cuda.is_available():
            capability.has_torch_cuda = True
            capability.cuda_devices = torch.cuda.device_count()
            
            # Get primary GPU info
            if capability.cuda_devices > 0:
                props = torch.cuda.get_device_properties(0)
                capability.gpu_memory = props.total_memory / (1024**3)  # Convert to GB
                capability.gpu_name = props.name
                capability.compute_capability = (props.major, props.minor)
                
                # Get CUDA version
                capability.cuda_version = torch.version.cuda
                
            capability.has_cuda = True
            logger.info(f"PyTorch CUDA rilevato: {capability.cuda_devices} device(s)")
            logger.info(f"GPU primaria: {capability.gpu_name} ({capability.gpu_memory:.1f}GB)")
    except ImportError:
        logger.debug("PyTorch non disponibile")
    except Exception as e:
        logger.warning(f"Errore rilevamento PyTorch CUDA: {e}")
    
    # Test CuPy availability
    try:
        import cupy as cp
        if cp.cuda.is_available():
            capability.has_cupy = True
            if not capability.has_cuda:
                capability.cuda_devices = cp.cuda.runtime.getDeviceCount()
                capability.has_cuda = True
            logger.info("CuPy CUDA rilevato")
    except ImportError:
        logger.debug("CuPy non disponibile")
    except Exception as e:
        logger.warning(f"Errore rilevamento CuPy: {e}")
    
    # Test cuML availability (RAPIDS)
    try:
        import cuml
        capability.has_cuml = True
        logger.info("RAPIDS cuML rilevato")
    except ImportError:
        logger.debug("RAPIDS cuML non disponibile")
    except Exception as e:
        logger.warning(f"Errore rilevamento cuML: {e}")
    
    # Log summary
    if capability.has_cuda:
        logger.info("✅ CUDA support disponibile")
        logger.info(f"   - Dispositivi: {capability.cuda_devices}")
        logger.info(f"   - Memoria GPU: {capability.gpu_memory:.1f}GB" if capability.gpu_memory else "   - Memoria GPU: N/A")
        logger.info(f"   - CUDA Version: {capability.cuda_version}" if capability.cuda_version else "   - CUDA Version: N/A")
    else:
        logger.info("❌ CUDA support non disponibile - usando CPU")
    
    return capability

def get_optimal_backend(requested_backend: str = "auto") -> GPUBackend:
    """
    Determina il backend ottimale basato su richiesta utente e capacità sistema.
    
    Args:
        requested_backend: Backend richiesto ("cpu", "cuda", "auto")
    
    Returns:
        GPUBackend: Backend ottimale da utilizzare
    """
    capability = detect_gpu_capability()
    
    # Se esplicitamente richiesto CPU
    if requested_backend.lower() == "cpu":
        logger.info("Backend CPU richiesto esplicitamente")
        return GPUBackend.CPU
    
    # Se esplicitamente richiesto CUDA
    if requested_backend.lower() == "cuda":
        if capability.has_cuda:
            logger.info("Backend CUDA richiesto e disponibile")
            return GPUBackend.CUDA
        else:
            logger.warning("Backend CUDA richiesto ma non disponibile - fallback su CPU")
            return GPUBackend.CPU
    
    # Modalità automatica
    if requested_backend.lower() == "auto":
        if capability.has_cuda:
            # Controlla se abbiamo memoria GPU sufficiente
            if capability.gpu_memory and capability.gpu_memory < 2.0:
                logger.warning(f"GPU memory insufficiente ({capability.gpu_memory:.1f}GB < 2GB) - usando CPU")
                return GPUBackend.CPU
            
            logger.info("Auto-detection: usando backend CUDA")
            return GPUBackend.CUDA
        else:
            logger.info("Auto-detection: usando backend CPU")
            return GPUBackend.CPU
    
    # Fallback per valori non riconosciuti
    logger.warning(f"Backend '{requested_backend}' non riconosciuto - usando CPU")
    return GPUBackend.CPU

@dataclass 
class GPUConfig:
    """Configurazione GPU ottimizzata per il sistema."""
    backend: GPUBackend
    device_id: int = 0
    memory_limit: Optional[float] = None
    mixed_precision: bool = True
    max_parallel_models: int = 100
    chunk_size: int = 1000
    capability: Optional[GPUCapability] = None
    
    def __post_init__(self):
        """Ottimizza configurazione basata sulle capacità hardware."""
        if self.capability is None:
            self.capability = detect_gpu_capability()
        
        self._optimize_for_hardware()
    
    def _optimize_for_hardware(self):
        """Ottimizza parametri basati su hardware disponibile."""
        if not self.capability.has_cuda:
            return
        
        # Ottimizza max_parallel_models basato su memoria GPU
        if self.capability.gpu_memory:
            if self.capability.gpu_memory >= 24:  # RTX 4090, A100, etc.
                self.max_parallel_models = 500
                self.chunk_size = 2000
            elif self.capability.gpu_memory >= 16:  # RTX 4080, etc.
                self.max_parallel_models = 200
                self.chunk_size = 1500
            elif self.capability.gpu_memory >= 8:  # RTX 4070, etc.
                self.max_parallel_models = 100
                self.chunk_size = 1000
            else:  # Lower-end GPUs
                self.max_parallel_models = 50
                self.chunk_size = 500
                
        # Disabilita mixed precision per compute capability < 7.0
        if (self.capability.compute_capability and 
            self.capability.compute_capability[0] < 7):
            self.mixed_precision = False
            logger.info("Mixed precision disabilitata per compute capability < 7.0")
    
    def get_device_context(self):
        """Restituisce context manager per il device appropriato."""
        if self.backend == GPUBackend.CUDA:
            try:
                import torch
                return torch.cuda.device(self.device_id)
            except ImportError:
                logger.warning("PyTorch non disponibile - usando CPU context")
                return DummyContext()
        
        return DummyContext()
    
    def get_array_module(self):
        """Restituisce il modulo array appropriato (numpy vs cupy)."""
        if self.backend == GPUBackend.CUDA and self.capability.has_cupy:
            try:
                import cupy as cp
                cp.cuda.Device(self.device_id).use()
                return cp
            except ImportError:
                pass
        
        import numpy as np
        return np
    
    def to_dict(self) -> Dict[str, Any]:
        """Serializza configurazione."""
        return {
            'backend': self.backend.value,
            'device_id': self.device_id,
            'memory_limit': self.memory_limit,
            'mixed_precision': self.mixed_precision,
            'max_parallel_models': self.max_parallel_models,
            'chunk_size': self.chunk_size
        }

class DummyContext:
    """Context manager dummy per CPU fallback."""
    def __enter__(self):
        return self
    def __exit__(self, *args):
        pass

def get_gpu_config(backend: str = "auto", **kwargs) -> GPUConfig:
    """
    Crea configurazione GPU ottimizzata.
    
    Args:
        backend: Backend richiesto
        **kwargs: Override configurazione
    
    Returns:
        GPUConfig: Configurazione ottimizzata
    """
    optimal_backend = get_optimal_backend(backend)
    
    config = GPUConfig(
        backend=optimal_backend,
        **kwargs
    )
    
    logger.info(f"GPU Config creata: backend={config.backend.value}, "
                f"max_parallel={config.max_parallel_models}, "
                f"chunk_size={config.chunk_size}")
    
    return config

def set_gpu_memory_limit(limit_gb: float) -> bool:
    """
    Imposta limite memoria GPU se supportato.
    
    Args:
        limit_gb: Limite in GB
    
    Returns:
        bool: True se impostato con successo
    """
    try:
        import torch
        if torch.cuda.is_available():
            # PyTorch memory fraction
            memory_fraction = min(limit_gb / detect_gpu_capability().gpu_memory, 0.95)
            torch.cuda.set_per_process_memory_fraction(memory_fraction)
            logger.info(f"Limite memoria GPU impostato: {limit_gb}GB (fraction: {memory_fraction:.2f})")
            return True
    except Exception as e:
        logger.warning(f"Impossibile impostare limite memoria GPU: {e}")
    
    return False
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Any
import yaml
import hashlib
import os

@dataclass
class HHConfig:
    """Rigorous Hodgkin-Huxley Model Parameters"""
    N: int = 100
    density: float = 0.2
    conn_type: str = "dale"
    
    # Biologically Plausible Constants
    C: float = 1.0
    gNa: float = 120.0; ENa: float = 50.0
    gK: float = 36.0; EK: float = -77.0
    gL: float = 0.3; EL: float = -54.4
    gA: float = 20.0; EA: float = -80.0
    tauA: float = 20.0
    
    # Synaptic
    Eexc: float = 0.0; Einh: float = -80.0
    tau_syn: float = 5.0 
    tau_in: float = 10.0
    
    # Input Construction
    in_density: float = 0.2
    in_gain: float = 5.0

@dataclass
class TaskConfig:
    """Task-Specific Parameters"""
    poisson_rate_min: float = 10.0
    poisson_rate_max: float = 150.0
    
    # Timing
    dt: float = 0.05
    symbol_ms: float = 20.0 
    
    # Task Specifics
    narma_order: int = 10 
    xor_delay: int = 2
    mc_max_lag: int = 20
    
    # Lyapunov
    lyap_window: Tuple[int, int] = (50, 250)
    lyap_eps: float = 1e-6
    zscore_features: bool = True
    
    # Readout Filter
    tau_trace: float = 20.0

@dataclass
class ESNConfig:
    """Fair ESN Baseline Parameters"""
    N: int = 100
    spectral_radius: float = 0.95
    input_scale: float = 1.0
    density: float = 0.1
    leaking_rate: float = 1.0

@dataclass
class ExperimentConfig:
    """Experiment Execution Control"""
    # Sweep Grids - Coarse
    rho_grid_coarse: List[float] = field(default_factory=lambda: [0.01, 1.5]) # Range forROI
    bias_grid_coarse: List[float] = field(default_factory=lambda: [0.0, 8.0])
    seeds_coarse: int = 5
    seeds_fine: int = 20
    
    # Readout
    cv_folds: int = 5
    cv_gap: int = 10  # window steps
    ridge_alphas: List[float] = field(default_factory=lambda: [1e-6, 1e-4, 1e-2, 1.0, 10.0, 100.0])

    # Ablations
    ablation_mode: str = "none" # 'none', 'no_zscore', 'random_gates'
    
    # Paths
    cache_dir: str = "cache"
    results_dir: str = "results"
    sweep_mode: str = "coarse" # 'coarse', 'fine'

    hh: HHConfig = field(default_factory=HHConfig)
    task: TaskConfig = field(default_factory=TaskConfig)
    esn: ESNConfig = field(default_factory=ESNConfig)

    def get_task_input_id(self) -> str:
        """Determines cache compatibility. capture all input generation and filtering params."""
        id_str = f"{self.task.poisson_rate_min}_{self.task.poisson_rate_max}_"
        id_str += f"{self.task.dt}_{self.task.symbol_ms}_{self.task.zscore_features}_{self.ablation_mode}_"
        id_str += f"{self.task.narma_order}_{self.task.xor_delay}_{self.task.mc_max_lag}_"
        id_str += f"{self.hh.tau_in}_{self.hh.tau_syn}_{self.task.tau_trace}_{self.hh.in_density}_{self.hh.in_gain}"
        return hashlib.sha1(id_str.encode()).hexdigest()


def load_config(path: str) -> ExperimentConfig:
    with open(path, 'r') as f:
        data = yaml.safe_load(f)
    
    hh_data = data.pop('hh', {})
    task_data = data.pop('task', {})
    
    cfg = ExperimentConfig(**data)
    cfg.hh = HHConfig(**hh_data)
    cfg.task = TaskConfig(**task_data)
    return cfg

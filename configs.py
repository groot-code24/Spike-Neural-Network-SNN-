
from dataclasses import dataclass, field
from typing import List


@dataclass
class SNNConfig:
    input_dim:     int   = 20
    hidden_dim:    int   = 128
    n_layers:      int   = 2
    output_dim:    int   = 2
    T:             int   = 60
    dt:            float = 1.0
    tau_mem:       float = 20.0   # 9.9% retention over 45-step delay
    tau_syn:       float = 8.0
    v_th:          float = 0.5
    v_reset:       float = 0.0
    alpha_surr:    float = 1.0
    target_rate:   float = 0.15
    rate_penalty:  float = 0.005
    readout_len:   int   = 10     # classify from last 10 steps only
    all_recurrent: bool  = True   # both layers recurrent; required for 45-step delay


@dataclass
class TaskConfig:
    task_name:       str   = "delayed_xor"
    n_train:         int   = 4000
    n_val:           int   = 500
    n_test:          int   = 1000
    batch_size:      int   = 32
    input_dim:       int   = 20
    output_dim:      int   = 2
    T:               int   = 60
    cue_duration:    int   = 5      # brief 5-step cue
    delay:           int   = 45     # 45 steps of blank + distractors
    distractor_rate: float = 0.10   # noise during delay to prevent timing shortcuts
    noise_rate:      float = 0.02
    randomize_delay: bool  = False  # fixed delay for cleaner ablation


@dataclass
class BPTTConfig:
    lr:           float = 1e-3
    weight_decay: float = 1e-4
    grad_clip:    float = 1.0
    epochs:       int   = 60


@dataclass
class TruncBPTTConfig:
    lr:           float = 1e-3
    weight_decay: float = 1e-4
    grad_clip:    float = 1.0
    epochs:       int   = 60
    trunc_len:    int   = 10   # K=10 << delay=45 → must fail to bridge delay


@dataclass
class EpropConfig:
    lr:              float = 2e-3
    weight_decay:    float = 1e-4
    grad_clip:       float = 1.0
    epochs:          int   = 60
    tau_e:           float = 25.0  # eligibility trace time constant
    tau_delta:       int   = 0
    alpha_surr:      float = 1.0
    n_feedback_cols: int   = 16
    head_lr_mult:    float = 5.0


@dataclass
class CTCAConfig:
    lr:                float = 2e-3
    weight_decay:      float = 1e-4
    grad_clip:         float = 1.0
    epochs:            int   = 60
    trunc_len:         int   = 60   # full-sequence causal sweep
    alpha_surr:        float = 1.0
    n_forward_samples: int   = 0
    influence_decay:   float = 0.99  # 0.99^45 ≈ 0.64: meaningful credit over full delay
    head_lr_mult:      float = 5.0


@dataclass
class ExperimentConfig:
    exp_name:   str       = "ctca_v2_scientific"
    seed:       int       = 42
    device:     str       = "cpu"
    methods:    List[str] = field(default_factory=lambda: [
        "bptt", "tbptt", "eprop", "ctca"
    ])
    log_every:  int       = 10
    save_plots: bool      = True
    output_dir: str       = "results"
    snn:   SNNConfig       = field(default_factory=SNNConfig)
    task:  TaskConfig      = field(default_factory=TaskConfig)
    bptt:  BPTTConfig      = field(default_factory=BPTTConfig)
    tbptt: TruncBPTTConfig = field(default_factory=TruncBPTTConfig)
    eprop: EpropConfig     = field(default_factory=EpropConfig)
    ctca:  CTCAConfig      = field(default_factory=CTCAConfig)

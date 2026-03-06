"""Configuration for all experiments.

Supports two profiles:
  - 'local':  Small-scale (277K params, 500 train) for quick local iteration
  - 'server': Large-scale (10M+ params, 10K+ train) for GPU servers (H100 etc.)

Select via environment variable:  EXPERIMENT_SCALE=server python3 run_all.py
Default: auto-detect (server if CUDA available, else local)
"""
import os
import torch


def _detect_profile():
    env = os.environ.get('EXPERIMENT_SCALE', '').lower()
    if env in ('server', 'large'):
        return 'server'
    elif env in ('local', 'small'):
        return 'local'
    # Auto-detect: use server profile if CUDA is available
    return 'server' if torch.cuda.is_available() else 'local'


PROFILE = _detect_profile()


class Config:
    # ================================================================
    # Profile-dependent settings
    # ================================================================
    if PROFILE == 'server':
        # --- Model architecture (server: ~10.5M params) ---
        vocab_size = 35
        d_model = 512
        nhead = 8
        num_layers = 6
        d_ff = 2048
        max_seq_len = 24
        dropout = 0.0

        # --- Training ---
        batch_size = 128
        lr = 1e-3
        epochs = 50
        seed = 42
        weight_decay = 1e-4

        # --- Dataset sizes ---
        n_train = 10000
        n_val = 2000
        n_test = 3000
        shortcut_ratio = 0.70

        # --- Per-sample gradient computation ---
        score_max_samples = 2000   # compute ShortcutScores for this many samples
        score_batch_size = 16      # batched per-sample gradients for efficiency

        # --- Data Filtering ---
        df_warmup_epochs = 5
        df_confidence_threshold = 0.90

    else:
        # --- Model architecture (local: ~277K params) ---
        vocab_size = 35
        d_model = 128
        nhead = 4
        num_layers = 2
        d_ff = 256
        max_seq_len = 24
        dropout = 0.0

        # --- Training ---
        batch_size = 32
        lr = 3e-3
        epochs = 30
        seed = 42
        weight_decay = 1e-5

        # --- Dataset sizes ---
        n_train = 500
        n_val = 200
        n_test = 300
        shortcut_ratio = 0.70

        # --- Per-sample gradient computation ---
        score_max_samples = 200    # compute ShortcutScores for this many samples
        score_batch_size = 1       # one at a time on CPU/MPS

        # --- Data Filtering ---
        df_warmup_epochs = 3
        df_confidence_threshold = 0.90

    # ================================================================
    # Shared settings (profile-independent)
    # ================================================================

    # ShortcutScore hyperparameters
    alpha = 1.0       # weight for B(s) (alignment component)
    beta = 1.0        # weight for C(s) (concentration component)
    tau_A = 0.3        # alignment threshold
    tau_R = 0.5        # concentration threshold
    lambda_ = 2.0      # reweighting strength
    gamma = 0.8        # gradient projection strength
    rho = 0.7          # answer-gradient suppression strength
    val_grad_interval = 5  # recompute validation gradient every K steps

    # Self-Consistency Decoding
    sc_num_samples = 5
    sc_temperature = 0.8

    # Device
    device = 'cuda' if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() else 'cpu')

    # Token IDs
    PAD = 0
    BOS = 1
    EOS = 2
    SEP = 3      # separates reasoning from answer
    EQ = 4       # equals sign
    PLUS = 5
    MINUS = 6
    MULT = 7
    COLON = 8
    SAT = 9      # Satisfied (financial)
    VIO = 10     # Violated (financial)
    CAUS = 11    # Causal
    NCAUS = 12   # Not Causal
    DIGIT_OFFSET = 13  # digit d -> token d + DIGIT_OFFSET (0->13, 9->22)
    # Feature markers for financial/causal datasets
    FEAT_R = 23   # Revenue
    FEAT_C = 24   # Cost
    FEAT_M = 25   # Margin
    FEAT_D = 26   # Debt
    FEAT_X = 27   # Variable X
    FEAT_Y = 28   # Variable Y
    FEAT_Z = 29   # Variable Z
    FEAT_COR = 30  # Correlation

    @classmethod
    def digit_token(cls, d):
        return d + cls.DIGIT_OFFSET

    @classmethod
    def token_to_digit(cls, t):
        return t - cls.DIGIT_OFFSET

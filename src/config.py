"""Configuration for all experiments."""
import torch


class Config:
    # Model architecture
    vocab_size = 35
    d_model = 128
    nhead = 4
    num_layers = 2
    d_ff = 256
    max_seq_len = 24
    dropout = 0.0

    # Training
    batch_size = 32
    lr = 3e-3
    epochs = 30
    seed = 42
    weight_decay = 1e-5

    # Dataset sizes
    n_train = 500
    n_val = 200
    n_test = 300
    shortcut_ratio = 0.70  # fraction of training data with shortcut bias

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

    # Data Filtering
    df_warmup_epochs = 3
    df_confidence_threshold = 0.90

    # Device
    device = 'mps' if torch.backends.mps.is_available() else 'cpu'

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

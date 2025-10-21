# uos/presets.py
NONDUAL_DEFAULT = dict(
    graph="learnable_causal",
    epochs=20,
    batch_size=64,
    substrate_dim=128,
    num_phenomena=12,
    temporal_weight=0.30,
    gauge_weight=0.02,
    verbose=True,
    print_every=50,
    save_state=True,
    log_file="runs/train_full.log",
)

# pylint: skip-file

_base_ = "gdino_base_cfg.py"

model = dict(
    num_queries=20,
    # Disable grad checkpointing
    encoder=dict(num_cp=0),
    # This should be set automatically
    # bbox_head=dict(max_text_len=10),
)

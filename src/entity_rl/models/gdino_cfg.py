# pylint: skip-file

_base_ = "gdino_base_cfg.py"

chkp = "grounding_dino_swin-t_pretrain_obj365_goldg_grit9m_v3det_20231204_095047-b448804b.pth"

model = dict(
    # Number of entities
    # NOTE: this needs to stay that same as the original model, i.e. 900
    # num_queries=900,
    # Disable grad checkpointing
    encoder=dict(num_cp=0),
    # This should be set automatically
    # bbox_head=dict(max_text_len=10),
)

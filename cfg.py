import os.path as op
import albumentations as A

hf_cache_root = op.expanduser('~/.cache/huggingface')
hf_hub_cache = op.join(hf_cache_root, 'hub')
hf_transformers_cache = op.join(hf_cache_root, 'transformers')

gpus = '4,5'
device_n = len(gpus.split(','))
mode = 'val'  # 'train', 'val', 'infer_path_eval'
eval_mode = ['loc']
check_val = False
skip_val = True

# ------------------ MODEL CFG -------------------
ckpt = ''  # path to pretrained checkpoint (.pth)
finetune_mode = 'lora'
lora_rank = 32
moeffort_target_keywords = ("attention", "mlp")
num_q = 1
img_cls_loss_weight = 1.0

# -------------------- DATA ----------------------
data_root = ''  # path to DocTamperV1 dataset
forg_type_dir = ''  # path to tampering_types directory
path_pkl_dir = ''  # path to eval path-pkl directory
exp_out_dir = './exp_out'

infer_path_eval_pkl = ""  # path to test_image_path_list.pkl
infer_path_eval_save_dir = ""  # output dir for inference masks
infer_path_eval_patch_batch_size = 8
infer_path_eval_patch_fake_threshold = 0.5
infer_path_eval_overwrite = True

realtextv2_data_root = './data'
realtextv2_img_size = 512
realtextv2_train_crop_aug = A.CropNonEmptyMaskIfExists(
    height=realtextv2_img_size,
    width=realtextv2_img_size,
    p=1.0,
)

all_ds_name = ['T-SROIE_test', 'Tampered-IC13_test', 'RealTextManipulation_test', 'OSTF_test']
val_name_list = all_ds_name
val_sample_n = 1000

# ------------------- TRAINING -------------------
train_bs = 4
val_bs = 64
accum_step = 1
step_per_epoch = 500
val_step = step_per_epoch * 10
ds_len = sample_per_epoch = step_per_epoch * train_bs * device_n
print_log_step = 100
epochs = 10
lr = 3e-4
min_lr = 1e-5
weight_decay = 1e-4
dl_workers = 0

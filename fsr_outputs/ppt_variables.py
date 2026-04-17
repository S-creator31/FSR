# FSR PPT Variables — paste into your presentation notes or slides script
# Generated automatically by fsr_visualize.py

MODEL      = "resnet18"
DATASET    = "CIFAR10"
SAVE_NAME  = "cifar10_resnet18"

ACC_CLEAN  = 83.26   # Clean accuracy (%)
ACC_FGSM   = 57.26   # FGSM accuracy (%)
ACC_PGD20  = 50.54   # PGD-20 accuracy (%)
ACC_PGD50  = 49.84   # PGD-50 accuracy (%)

# Drop from CLEAN to PGD-20 (adversarial degradation):
ACC_DROP_PGD20 = 32.72

# Image paths for PPT insertion:
IMG_BAR_CHART    = "/teamspace/studios/this_studio/FSR/fsr_outputs/robustness_bar.png"
IMG_ATTN_MAPS    = "/teamspace/studios/this_studio/FSR/fsr_outputs/attention_maps.png"
IMG_FEAT_NORMS   = "/teamspace/studios/this_studio/FSR/fsr_outputs/feature_norms.png"

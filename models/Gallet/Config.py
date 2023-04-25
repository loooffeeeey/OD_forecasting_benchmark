# DEBUG FLAGS
TRAIN_JUST_ONE_BATCH = False
TRAIN_JUST_ONE_ROUND = False
PROFILE = False
CHECK_GRADS = False
# RAND_SEED = None
RAND_SEED = 66666

# Basic
LEARNING_RATE_DEFAULT = 1e-2    # 0.01
MAX_EPOCHS_DEFAULT = 200
EVAL_FREQ_DEFAULT = 5
BATCH_SIZE_DEFAULT = 32
OPTIMIZER_DEFAULT = 'ADAM'
WEIGHT_DECAY_DEFAULT = 0.01
DATA_DIR_DEFAULT = 'data/ny2016_0101to0331/'
LOG_DIR_DEFAULT = 'log/'
WORKERS_DEFAULT = 36
USE_GPU_DEFAULT = 1
GPU_ID_DEFAULT = 0
NETWORK_DEFAULT = 'GallatExt'
TAG_DEFAULT = None
NETWORKS = ['Gallat', 'GallatExt', 'GallatExtFull', 'AR', 'LSTNet', 'GCRN', 'GEML']
NETWORKS_TUNABLE = ['Gallat', 'GallatExt', 'GallatExtFull']
MULTI_HEAD_ATT_APPLICABLE = ['Gallat', 'GallatExt', 'GallatExtFull', 'GCRN', 'GEML']
REF_AR_DEFAULT = 'None'  # should be an AR model file name
MODE_DEFAULT = 'trainNeval'
EVAL_DEFAULT = 'eval.pth'   # should be a model file name
MODEL_SAVE_DIR_DEFAULT = 'model_save/'

UNIFY_FB_DEFAULT = 0
MIX_FB_DEFAULT = 0

MAX_NORM_DEFAULT = 10.0

FEAT_DIM_DEFAULT = 43
QUERY_DIM_DEFAULT = 41
HIDDEN_DIM_DEFAULT = 16

LOSS_FUNC_DEFAULT = 'SmoothL1Loss'
SMOOTH_L1_LOSS_BETA_DEFAULT = 10

NUM_HEADS_DEFAULT = 3

HISTORICAL_RECORDS_NUM_DEFAULT = 7
TIME_SLOT_ENDURANCE_DEFAULT = 1     # hour

TUNE_DEFAULT = 1

TEMP_FEAT_NAMES = ['St', 'Sp', 'Stpm', 'Stpp']
LSTNET_TEMP_FEAT = 'Stext'
ALL_TEMP_FEAT_NAMES = TEMP_FEAT_NAMES + [LSTNET_TEMP_FEAT]
HA_FEAT_DEFAULT = 'all'     # ['all', 'tendency', 'periodicity']

D_PERCENTAGE_DEFAULT = 0.5
G_PERCENTAGE_DEFAULT = 1 - D_PERCENTAGE_DEFAULT

REF_EXTENT = -1    # If -1, use scaling scheme; else if -2, use shifting scheme; Else, a simple leverage (normally 0.2)

TRAIN_TYPES = ['normal', 'pretrain', 'retrain']
TRAIN_TYPE_DEFAULT = 'normal'

RETRAIN_MODEL_PATH_DEFAULT = 'pretrained_model.pth'

DATA_TOTAL_H = -1
DATA_START_H = -1

# Customize: DIY
EVAL_METRICS_THRESHOLD_SET = [0, 3, 5]

METRICS_NAME = ['RMSE', 'MAPE', 'MAE']

METRICS_FOR_WHAT = ['Demand', 'OD']

OUR_MODEL = 'RefGaaRN'

MODELS_TO_EXAMINE = [
    ['HAplus', 'HAt', 'HAp', 'AR'],             # baseline
    ['LSTNet', 'GCRN', 'GEML', 'Gallat'],       # others
    [OUR_MODEL],                                # ours
    ['RefGaaRNNoTune', 'RefGaaRNConcat',        # variants
     'RefGaaRNWSum', 'RefGaaRNShift',
     'RefGaaRNARTune']
]

GALLAT_FINAL_ACTIVATION_USE_SIGMOID = True

GALLATEXT_TEMP_USE_ATT = False

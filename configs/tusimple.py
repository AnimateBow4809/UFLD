# DATA
dataset='Tusimple'
data_root = 'data_dump/TUSimple_flat'

# TRAIN
epoch = 100
batch_size = 32
optimizer = 'Adam'    #['SGD','Adam']
# learning_rate = 0.1
learning_rate = 4e-4
weight_decay = 1e-4
momentum = 0.9

scheduler = 'cos'     #['multi', 'cos']
# steps = [50,75]
gamma  = 0.1
warmup = 'linear'
warmup_iters = 100

# NETWORK
backbone = '18'
griding_num = 100
use_aux = True

# LOSS
sim_loss_w = 1.0
shp_loss_w = 0.0

# EXP
note = ''

log_path = 'log_dump'

# FINETUNE or RESUME MODEL PATH
finetune = None
resume = None

# TEST
test_model = "model_dump/model_fixed.pth"
test_work_dir = "test_workdir"

num_lanes = 4

## My CONFIGS
FWL=4
IWL=4
bitWidth=8
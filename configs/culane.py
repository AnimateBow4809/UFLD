# DATA
dataset='CULane'
data_root = 'data_dump/culane'

# TRAIN
epoch = 50
batch_size = 32
optimizer = 'SGD'  #['SGD','Adam']
learning_rate = 0.1
weight_decay = 1e-4
momentum = 0.9

scheduler = 'multi' #['multi', 'cos']
steps = [25,38]
gamma  = 0.1
warmup = 'linear'
warmup_iters = 695

# NETWORK
use_aux = True
griding_num = 200
backbone = '18'

# LOSS
sim_loss_w = 0.0
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


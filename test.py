import importlib


# module = 'tensorboardX'
module = 'torch.utils.tensorboard'
log_dir = 'experiments/debug_uncropping_custom_240414_172216/tb_logger'
writer = importlib.import_module(module).SummaryWriter(log_dir)
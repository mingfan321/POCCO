##########################################################################################
# Machine Environment Config
DEBUG_MODE = False
USE_CUDA = not DEBUG_MODE
##########################################################################################
# Path Config
import os
import sys
import argparse

os.chdir(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, "..")  # for problem_def
sys.path.insert(0, "../..")  # for utils

parser = argparse.ArgumentParser(description='verify_mocvrp_hyperparameter')
parser.add_argument('--beta', type=float, default=3.5, help='beta')
parser.add_argument('--cuda_device_num', type=int, default=1, help='CUDA_NUM')
args = parser.parse_args()

CUDA_DEVICE_NUM = args.cuda_device_num

##########################################################################################
# import
import logging
from utils.utils import create_logger, copy_all_src
from MOCVRPTrainer import CVRPTrainer as Trainer


##########################################################################################
# parameters
# env_params = {
#     'problem_size': 50,
#     'pomo_size': 50,
# }

model_params = {
    'embedding_dim': 128,
    'sqrt_embedding_dim': 128**(1/2),
    'encoder_layer_num': 6,
    'qkv_dim': 16,
    'head_num': 8,
    'logit_clipping': 50,
    'ff_hidden_dim': 512,
    'eval_type': 'softmax',
    "num_experts": 5,
    "topk": 2,
    "routing_level": "node",
    "routing_method": "input_choice"
}

optimizer_params = {
    'optimizer': {
        'lr': 3e-4,
        'weight_decay': 1e-6
    },
    'scheduler': {
        'milestones': [180,],
        'gamma': 0.1
    }
}

trainer_params = {
    'use_cuda': USE_CUDA,
    'cuda_device_num': CUDA_DEVICE_NUM,
    'dec_method': 'WS',
    'beta': args.beta,
    # 'evaluate': True,
    # "evaluate_interval": 10,
    'epochs': 200,
    'train_episodes': 100 * 1000,
    'train_batch_size': 64,
    # 'prev_model_path': None,
    'logging': {
        'model_save_interval': 50,
        'validation_interval': 10,
        'log_image_params_1': {
            'json_foldername': 'log_image_style',
            'filename': 'style_cvrp_50.json'
        },
        'log_image_params_2': {
            'json_foldername': 'log_image_style',
            'filename': 'style_loss_1.json'
        },
    },
    'model_load': {
        'enable': False,  # enable loading pre-trained model
        'path': './result/20240329_121100_train_cvrp_mix',  # directory path of pre-trained model and log files saved.
        'epoch': 1,  # epoch version of pre-trained model to laod.
    }
}

logger_params = {
    'log_file': {
        'desc': 'train_cvrp_mix',
        'filename': 'run_log'
    }
}

##########################################################################################
# main
def main():
    if DEBUG_MODE:
        _set_debug_mode()

    create_logger(**logger_params)
    _print_config()

    # trainer = Trainer(env_params=env_params,
    #                   model_params=model_params,
    #                   optimizer_params=optimizer_params,
    #                   trainer_params=trainer_params)
    trainer = Trainer(model_params=model_params,
                      optimizer_params=optimizer_params,
                      trainer_params=trainer_params)

    copy_all_src(trainer.result_folder)

    trainer.run()

def _set_debug_mode():
    global trainer_params
    trainer_params['epochs'] = 2
    trainer_params['train_episodes'] = 4
    trainer_params['train_batch_size'] = 2

def _print_config():
    logger = logging.getLogger('root')
    logger.info('DEBUG_MODE: {}'.format(DEBUG_MODE))
    logger.info('USE_CUDA: {}, CUDA_DEVICE_NUM: {}'.format(USE_CUDA, CUDA_DEVICE_NUM))
    [logger.info(g_key + "{}".format(globals()[g_key])) for g_key in globals().keys() if g_key.endswith('params')]

##########################################################################################


if __name__ == "__main__":
    # use 4000 EPOCH fixed_makespan UNIFORM sample
    # 4000 * 64 * 100
    # scheduler 200 times
    main()

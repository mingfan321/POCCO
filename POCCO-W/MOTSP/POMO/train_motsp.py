##########################################################################################
# Machine Environment Config
DEBUG_MODE = True
USE_CUDA = not DEBUG_MODE
# CUDA_DEVICE_NUM = 1

##########################################################################################
# Path Config
import os
import sys
import argparse

os.chdir(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, "..")  # for problem_def
sys.path.insert(0, "../..")  # for utils

parser = argparse.ArgumentParser(description='verify_motsp_hyperparameter')
parser.add_argument('--beta', type=float, default=3.5, help='beta')
parser.add_argument('--cuda_device_num', type=int, default=1, help='CUDA_NUM')
args = parser.parse_args()

CUDA_DEVICE_NUM = args.cuda_device_num

##########################################################################################
# import
import logging
from utils.utils import create_logger, copy_all_src

from MOTSPTrainer import TSPTrainer as Trainer

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
    # 'hyper_hidden_dim': 256,
    'eval_type': 'softmax',
    'MoD_layer_num': 1,
    'expert_loc': ['Dec'],
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
    'epochs': 200,
    'train_episodes': 100 * 1000,
    'train_batch_size': 64,
    'logging': {
        # 'model_save_interval': 20,
        'model_save_interval': 50,
        'validation_interval': 10,
        'log_image_params_1': {
            'json_foldername': 'log_image_style',
            'filename': 'style_tsp_50.json'
        },
        'log_image_params_2': {
            'json_foldername': 'log_image_style',
            'filename': 'style_loss_1.json'
        },
    },
    'model_load': {
        'enable': True,  # enable loading pre-trained model
        'path': './train__tsp_mix_topk3',  # directory path of pre-trained model and log files saved.
        'epoch': 200,  # epoch version of pre-trained model to laod.
    }
}

logger_params = {
    'log_file': {
        'desc': 'train__tsp_mix',
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
    trainer_params['train_episodes'] = 10
    trainer_params['train_batch_size'] = 4


def _print_config():
    logger = logging.getLogger('root')
    logger.info('DEBUG_MODE: {}'.format(DEBUG_MODE))
    logger.info('USE_CUDA: {}, CUDA_DEVICE_NUM: {}'.format(USE_CUDA, CUDA_DEVICE_NUM))
    [logger.info(g_key + "{}".format(globals()[g_key])) for g_key in globals().keys() if g_key.endswith('params')]

##########################################################################################

if __name__ == "__main__":
    main()
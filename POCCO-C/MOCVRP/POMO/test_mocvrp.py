##########################################################################################
# Machine Environment Config
DEBUG_MODE = False
USE_CUDA = not DEBUG_MODE
CUDA_DEVICE_NUM = 1

##########################################################################################
# Path Config
import os
import sys
import torch
import numpy as np


os.chdir(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, "..")  # for problem_def
sys.path.insert(0, "../..")  # for utils

##########################################################################################
# import
import logging
from utils.utils import create_logger, copy_all_src
from utils.cal_pareto_demo import Pareto_sols
from utils.cal_ps_hv import cal_ps_hv

from MOCVRPTester import CVRPTester as Tester
# from generate_test_dataset import load_dataset
##########################################################################################
import time
import hvwfg


from matplotlib import pyplot as plt
import matplotlib as mpl
mpl.style.use('default')
##########################################################################################
# parameters

model_params = {
    'embedding_dim': 128,
    'sqrt_embedding_dim': 128**(1/2),
    'encoder_layer_num': 6,
    'qkv_dim': 16,
    'head_num': 8,
    'logit_clipping': 10,
    'ff_hidden_dim': 512,
    'eval_type': 'argmax',
    "num_experts": 5,
    "topk": 2,
    "routing_level": "node",
    "routing_method": "input_choice"
}

tester_params = {
    'use_cuda': USE_CUDA,
    'cuda_device_num': CUDA_DEVICE_NUM,
    'model_load': {
        'path': './result/train_cvrp_n20',  # directory path of pre-trained model and log files saved.
        'epoch': 200,  # epoch version of pre-trained model to laod.
    },
    'test_episodes': 200,
    'test_batch_size': 200,
    'augmentation_enable': True,
    'aug_factor': 8,
    'aug_batch_size': 100
}
# if tester_params['augmentation_enable']:
#     tester_params['test_batch_size'] = tester_params['aug_batch_size']

logger_params = {
    'log_file': {
        'desc': 'test__cvrp_n20',
        'filename': 'run_log'
    }
}

##########################################################################################
def _set_debug_mode():
    global tester_params
    tester_params['test_episodes'] = 100


def _print_config():
    logger = logging.getLogger('root')
    logger.info('DEBUG_MODE: {}'.format(DEBUG_MODE))
    logger.info('USE_CUDA: {}, CUDA_DEVICE_NUM: {}'.format(USE_CUDA, CUDA_DEVICE_NUM))
    [logger.info(g_key + "{}".format(globals()[g_key])) for g_key in globals().keys() if g_key.endswith('params')]
##########################################################################################
def main(n_sols = 101):

    timer_start = time.time()
    logger_start = time.time()
    device = torch.device('cuda:0' if USE_CUDA is True else 'cpu')
    
    if DEBUG_MODE:
        _set_debug_mode()
    
    create_logger(**logger_params)
    _print_config()
    
    tester = Tester(model_params=model_params,
                    tester_params=tester_params)
    
    copy_all_src(tester.result_folder)
    # sols = np.zeros([n_sols, 2])

    problem_size = 100

    if tester_params['augmentation_enable']:
        test_name = 'Aug'
        tester_params['test_batch_size'] = tester_params['aug_batch_size']
    else:
        test_name = 'NoAug'
    # shared_depot_xy, shared_node_xy, shared_node_demand = get_random_problems(tester_params['test_episodes'], env_params['problem_size'])
    # loaded_problem = load_dataset('/home/qiang/Desktop/MOVP/PMOCO/test_data/movrp/movrp%d_test_seed1234.pkl'%(problem_size))
    # shared_depot_xy, shared_node_xy, shared_node_demand, capacity = [], [], [], []
    # for i in range(len(loaded_problem)):
    #     depot, loc, dem, cap = loaded_problem[i]
    #     shared_depot_xy.append(depot)
    #     shared_node_xy.append(loc)
    #     shared_node_demand.append(dem)
    #     capacity.append(cap)
    # shared_depot_xy = torch.FloatTensor(shared_depot_xy).to(device)[:, None, :]
    # shared_node_xy = torch.FloatTensor(shared_node_xy).to(device)
    # shared_node_demand = torch.FloatTensor(np.array(shared_node_demand) / np.array(capacity)[:, None]).to(device)

    test_path = f"./data/testdata_cvrp_size{problem_size}.pt"
    data = torch.load(test_path)
    shared_node_demand = data['demand_data'].squeeze(-1).to(device=CUDA_DEVICE_NUM)
    shared_depot_xy = data['node_data'][:, 0, :].unsqueeze(1).to(device=CUDA_DEVICE_NUM)
    shared_node_xy = data['node_data'][:, 1:, :].to(device=CUDA_DEVICE_NUM)

    # batch_size = shared_node_xy.shape[0]
    # sols = np.zeros([batch_size, n_sols, 2])
    # for i in range(n_sols):
    #     pref = torch.zeros(2).cuda()
    #     pref[0] = 1 - 0.01 * i
    #     pref[1] = 0.01 * i
    #     pref = pref / torch.sum(pref)
    #     pref = pref[None, :].expand(shared_depot_xy.size(0), 2)
    #
    #     aug_score = tester.run(shared_depot_xy, shared_node_xy, shared_node_demand, pref)
    #     # sols[i] = np.array(aug_score)
    #     sols[:, i, 0] = np.array(aug_score[0].flatten())
    #     sols[:, i, 1] = np.array(aug_score[1].flatten())
    batch_size = shared_node_xy.shape[0]
    sols = np.zeros([batch_size, n_sols, 2])

    mini_batch_size = tester_params['test_batch_size']
    b_cnt = tester_params['test_episodes'] / mini_batch_size
    b_cnt = int(b_cnt)
    # time_list = np.zeros((n_sols, 1))
    total_test_time = 0
    for bi in range(0, b_cnt):
        b_start = bi * mini_batch_size
        b_end = b_start + mini_batch_size
        for i in range(n_sols):
            pref = torch.zeros(2).cuda()
            pref[0] = 1 - 0.01 * i
            pref[1] = 0.01 * i
            # pref = pref / torch.sum(pref)
            pref = pref[None, :].expand(shared_depot_xy.size(0), 2)

            test_timer_start = time.time()
            aug_score = tester.run(shared_depot_xy, shared_node_xy, shared_node_demand, pref, episode=b_start)
            test_timer_end = time.time()
            total_test_time += test_timer_end - test_timer_start
            print('Ins{:d} Test Time(s): {:.4f}'.format(i, test_timer_end - test_timer_start))

            sols[b_start:b_end, i, 0] = np.array(aug_score[0].flatten())
            sols[b_start:b_end, i, 1] = np.array(aug_score[1].flatten())
        
    timer_end = time.time()
    total_time = timer_end - timer_start


    if problem_size == 20:
        ref = np.array([30,4])    #20
    elif problem_size == 50:
        ref = np.array([45,4])   #50
    elif problem_size == 100:
        ref = np.array([80,4])   #100
    elif problem_size == 150:
        ref = np.array([120,3])   #150
    elif problem_size == 200:
        ref = np.array([180,3])   #200
    elif problem_size == 250:
        ref = np.array([220,3])   #250
    elif problem_size == 300:
        ref = np.array([250,3])   #300
    else:
        print('Have yet define a reference point for this problem size!')

    # if problem_size == 20:
    #     ref = np.array([15,3])    #20
    # elif problem_size == 50:
    #     ref = np.array([40,3])   #50
    # elif problem_size == 100:
    #     ref = np.array([60,3])   #100
    # else:
    #     print('Have yet define a reference point for this problem size!')
    
    # hv = hvwfg.wfg(sols.astype(float), ref.astype(float))
    # hv_ratio =  hv / (ref[0] * ref[1])

    nd_sort = Pareto_sols(p_size=problem_size, pop_size=sols.shape[0], obj_num=sols.shape[2])
    sols_t = torch.Tensor(sols)
    nd_sort.update_PE(objs=sols_t)
    p_sols, p_sols_num, _ = nd_sort.show_PE()
    hvs = cal_ps_hv(pf=p_sols, pf_num=p_sols_num, ref=ref)

    print('Run Time(s): {:.4f}'.format(total_time))
    print('HV Ratio: {:.4f}'.format(hvs.mean()))

##########################################################################################
if __name__ == "__main__":
    main()
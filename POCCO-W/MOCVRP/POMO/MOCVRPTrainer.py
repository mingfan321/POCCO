import torch
import torch.nn.functional as F
import numpy as np
from logging import getLogger
from einops import rearrange

from MOCVRP.MOCVRProblemDef import get_random_problems
from MOCVRPEnv import CVRPEnv as Env
from MOCVRPModel import CVRPModel as Model

from torch.optim import Adam as Optimizer
from torch.optim.lr_scheduler import MultiStepLR as Scheduler

from utils.utils import *
from utils.cal_pareto_demo import Pareto_sols
from utils.cal_ps_hv import cal_ps_hv

class CVRPTrainer:
    def __init__(self,
                 # env_params,
                 model_params,
                 optimizer_params,
                 trainer_params):

        # save arguments
        # self.env_params = env_params
        self.model_params = model_params
        self.optimizer_params = optimizer_params
        self.trainer_params = trainer_params

        # result folder, logger
        self.logger = getLogger(name='trainer')
        self.result_folder = get_result_folder()
        self.result_log = LogData()

        self.validation_log = {"problem_size20_score": [],
                               "problem_size50_score": [],
                               "problem_size100_score": []}
        # self.evaluate = self.trainer_params['evaluate']
        # self.evaluate_interval = self.trainer_params['evaluate_interval']

        # cuda
        USE_CUDA = self.trainer_params['use_cuda']
        if USE_CUDA:
            cuda_device_num = self.trainer_params['cuda_device_num']
            torch.cuda.set_device(cuda_device_num)
            device = torch.device('cuda', cuda_device_num)
            torch.set_default_tensor_type('torch.cuda.FloatTensor')
        else:
            device = torch.device('cpu')
            torch.set_default_tensor_type('torch.FloatTensor')
        self.device = device

        # Main Components
        self.model = Model(**self.model_params)
        # self.env = Env(**self.env_params)
        self.env = Env()
        self.optimizer = Optimizer(self.model.parameters(), **self.optimizer_params['optimizer'])
        self.scheduler = Scheduler(self.optimizer, **self.optimizer_params['scheduler'])

        # Restore
        self.start_epoch = 1
        model_load = trainer_params['model_load']
        if model_load['enable']:
            checkpoint_fullname = '{path}/checkpoint_mocvrp-{epoch}.pt'.format(**model_load)
            checkpoint = torch.load(checkpoint_fullname, map_location=device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.start_epoch = 1 + model_load['epoch']
            self.result_log.set_raw_data(checkpoint['result_log'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.scheduler.last_epoch = model_load['epoch']-1
            self.logger.info('Saved Model Loaded !!')

        # utility
        self.time_estimator = TimeEstimator()

    def run(self):
        self.time_estimator.reset(self.start_epoch)
        for epoch in range(self.start_epoch, self.trainer_params['epochs']+1):
            self.logger.info('=================================================================')
            # LR Decay
            # if epoch % 20 == 0:
            #     self.scheduler.step()

            # Train
            train_score_obj1, train_score_obj2, train_loss = self._train_one_epoch(epoch)
            self.result_log.append('train_score_obj1', epoch, train_score_obj1)
            self.result_log.append('train_score_obj2', epoch, train_score_obj2)
            self.result_log.append('train_loss', epoch, train_loss)

            ############################
            # Logs & Checkpoint
            ############################
            elapsed_time_str, remain_time_str = self.time_estimator.get_est_string(epoch, self.trainer_params['epochs'])
            self.logger.info("Epoch {:3d}/{:3d}: Time Est.: Elapsed[{}], Remain[{}]".format(
                epoch, self.trainer_params['epochs'], elapsed_time_str, remain_time_str))

            all_done = (epoch == self.trainer_params['epochs'])
            model_save_interval = self.trainer_params['logging']['model_save_interval']
            validation_interval = self.trainer_params['logging']['validation_interval']
            
            # Save Model
            if epoch == self.start_epoch or epoch == 190 or all_done or (epoch % model_save_interval) == 0:
                self.logger.info("Saving trained_model")
                checkpoint_dict = {
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'scheduler_state_dict': self.scheduler.state_dict(),
                    'result_log': self.result_log.get_raw_data(),
                    'validation_log': self.validation_log,
                }
                torch.save(checkpoint_dict, '{}/checkpoint_mocvrp-{}.pt'.format(self.result_folder, epoch))

            # if epoch == 1 or epoch % self.evaluate_interval == 0:
            #     # Evaluate
            #     self.eval()
            # if epoch == 1 or (epoch % validation_interval == 0):
            #     # val_episodes = 200
            #     ps_set = [20, 50, 100]
            #     for ps in ps_set:
            #         # shared_problem = torch.load("./data/testdata_cvrp_size%d.pt" % (ps)).to(self.device)
            #         test_path = f"./data/testdata_cvrp_size{ps}.pt"
            #         data = torch.load(test_path)
            #         shared_node_demand = data['demand_data'].squeeze(-1).to(self.device)
            #         shared_depot_xy = data['node_data'][:, 0, :].unsqueeze(1).to(self.device)
            #         shared_node_xy = data['node_data'][:, 1:, :].to(self.device)
            #         if ps == 20:
            #             ref = np.array([30, 4])
            #         elif ps == 50:
            #             ref = np.array([45, 4])
            #         elif ps == 100:
            #             ref = np.array([80, 4])
            #
            #         n_sols = 101
            #         batch_size = shared_node_demand.size(0)
            #         sols = np.zeros([batch_size, n_sols, 2])
            #         # sols_aug = np.empty((n_sols, batch_size, 2))
            #         for i in range(n_sols):
            #             pref = torch.zeros(2)
            #             pref[0] = 1 - 0.01 * i
            #             pref[1] = 0.01 * i
            #
            #             score = self._val_one_batch(shared_depot_xy, shared_node_xy, shared_node_demand, pref, aug_factor=1, eval_type="argmax")
            #
            #             sols[:, i, 0] = np.array(score[:, 0].cpu().flatten())
            #             sols[:, i, 1] = np.array(score[:, 1].cpu().flatten())
            #
            #         nd_sort = Pareto_sols(p_size=ps, pop_size=sols.shape[0], obj_num=sols.shape[2])
            #         sols_t = torch.Tensor(sols)
            #         nd_sort.update_PE(objs=sols_t)
            #         p_sols, p_sols_num, _ = nd_sort.show_PE()
            #         hvs = cal_ps_hv(pf=p_sols, pf_num=p_sols_num, ref=ref)
            #
            #         print(
            #             ">> Epoch {:3d}/{:3d}, Problem Size{:3d}, HV_Score: {:.4f}".format(
            #                 epoch,
            #                 self.trainer_params[
            #                     'epochs'],
            #                 ps,
            #                 hvs.mean(),))
            #         self.validation_log["problem_size%d_score" % ps].append(round(float(hvs.mean()), 4))

            # All-done announcement
            if all_done:
                self.logger.info(" *** Training Done *** ")
                self.logger.info("Now, printing log array...")
                # util_print_log_array(self.logger, self.result_log)

            self.scheduler.step()

    def _train_one_epoch(self, epoch):

        score_AM_obj1 = AverageMeter()
        score_AM_obj2 = AverageMeter()
        
        loss_AM = AverageMeter()

        train_num_episode = self.trainer_params['train_episodes']
        episode = 0
        loop_cnt = 0
        while episode < train_num_episode:

            remaining = train_num_episode - episode
            batch_size = min(self.trainer_params['train_batch_size'], remaining)

            avg_score_obj1, avg_score_obj2, avg_loss = self._train_one_batch(batch_size)
            score_AM_obj1.update(avg_score_obj1, batch_size)
            score_AM_obj2.update(avg_score_obj2, batch_size)
            loss_AM.update(avg_loss, batch_size)

            episode += batch_size

            # Log First 10 Batch, only at the first epoch
            if epoch == self.start_epoch:
                loop_cnt += 1
                if loop_cnt <= 10:
                    self.logger.info('Epoch {:3d}: Train {:3d}/{:3d}({:1.1f}%)  Obj1 Score: {:.4f}, Obj2 Score: {:.4f},  Loss: {:.4f}'
                                     .format(epoch, episode, train_num_episode, 100. * episode / train_num_episode,
                                             score_AM_obj1.avg, score_AM_obj2.avg, loss_AM.avg))

        # Log Once, for each epoch
        self.logger.info('Epoch {:3d}: Train ({:3.0f}%)  Obj1 Score: {:.4f}, Obj2 Score: {:.4f},  Loss: {:.4f}'
                         .format(epoch, 100. * episode / train_num_episode,
                                 score_AM_obj1.avg, score_AM_obj2.avg, loss_AM.avg))

        return score_AM_obj1.avg, score_AM_obj2.avg, loss_AM.avg

    def _train_one_batch(self, batch_size):

        # Prep
        ###############################################
        self.model.train()
        self.model.set_eval_type(self.model_params["eval_type"])
        nums = torch.arange(20, 101, dtype=torch.float)
        weights = nums - 20 + 1
        probs = weights / weights.sum()
        size = nums[torch.multinomial(probs, 1)]
        size = int(size.item())

        self.env.problem_size = size
        self.env.pomo_size = size
        self.env.load_problems(batch_size)

        alpha = 1
        dirichlet_params = torch.tensor([alpha, alpha], dtype=torch.float)
        pref = torch.distributions.Dirichlet(dirichlet_params).sample((batch_size,))
        # pref = torch.Tensor(batch_size, 2).uniform_(1e-6, 1)

        reset_state, _, _ = self.env.reset()
        
        # self.model.decoder.assign(pref)
        self.model.pre_forward(reset_state, pref)
        #
        # self.model.pre_forward(reset_state)
        
        prob_list = torch.zeros(size=(batch_size, self.env.pomo_size, 0))
        selected_list = torch.zeros(size=(batch_size, self.env.pomo_size, 0))
        # shape: (batch, pomo, 0~problem)

        # POMO Rollout
        ###############################################
        state, reward, done = self.env.pre_step()
        
        while not done:
            selected, prob = self.model(state)
            # shape: (batch, pomo)
            
            state, reward, done = self.env.step(selected)
            prob_list = torch.cat((prob_list, prob[:, :, None]), dim=2)
            selected_list = torch.cat((selected_list, selected[:, :, None]), dim=2)
            
        # Loss
        ###############################################
        # reward was negative, here we set it to positive to calculate TCH
        reward = - reward
        pref = pref[:, None, :].expand_as(reward)

        if self.trainer_params['dec_method'] == "WS":
            tch_reward = (pref * reward).sum(dim=2)
        elif self.trainer_params['dec_method'] == "TCH":
            z = torch.ones(reward.shape).cuda() * 0.0
            tch_reward = pref * (reward - z)
            tch_reward, _ = tch_reward.max(dim=2)
        else:
            return NotImplementedError
        
        # set back reward and group_reward to negative
        reward = -reward
        tch_reward = -tch_reward
    
        log_prob = prob_list.log().sum(dim=2)
        ## shape = (batch, group)

        #################################################
        # calculate the preference optimization

        po_indicate = tch_reward[:, :, None] > tch_reward[:, None, :]
        log_pair_pro = log_prob[:, :, None] - log_prob[:, None, :]
        token_num = prob_list.shape[2]
        pf_log = torch.log(F.sigmoid(self.trainer_params['beta'] * log_pair_pro / token_num))
        loss_mean = -torch.mean(po_indicate * pf_log)

        # if hasattr(self.model, "aux_loss"):
        #     loss_mean = loss_mean + self.model.aux_loss
    
        # Score
        ###############################################
        _ , max_idx = tch_reward.max(dim=1)
        max_idx = max_idx.reshape(max_idx.shape[0],1)
        max_reward_obj1 = reward[:,:,0].gather(1, max_idx)
        max_reward_obj2 = reward[:,:,1].gather(1, max_idx)
        
        score_mean_obj1 = - max_reward_obj1.float().mean()
        score_mean_obj2 = - max_reward_obj2.float().mean()

        # Step & Return
        ################################################
        self.model.zero_grad()
        loss_mean.backward()
        self.optimizer.step()

        return score_mean_obj1.item(), score_mean_obj2.item(), loss_mean.item()

    def _val_one_batch(self, depot_xy, node_xy, node_demand, pref, aug_factor=1, eval_type="argmax"):

        batch_size, problem_size, _ = node_xy.size()

        self.env.problem_size = problem_size
        self.env.pomo_size = problem_size
        self.env.batch_size = batch_size

        self.env.reset_state.depot_xy = depot_xy
        self.env.reset_state.node_xy = node_xy
        self.env.reset_state.node_demand = node_demand

        self.env.depot_node_xy = torch.cat((depot_xy, node_xy), dim=1)
        # shape: (batch, problem+1, 2)
        depot_demand = torch.zeros(size=(self.env.batch_size, 1))
        # shape: (batch, 1)
        self.env.depot_node_demand = torch.cat((depot_demand, node_demand), dim=1)

        self.env.BATCH_IDX = torch.arange(self.env.batch_size)[:, None].expand(self.env.batch_size, self.env.pomo_size)
        self.env.POMO_IDX = torch.arange(self.env.pomo_size)[None, :].expand(self.env.batch_size, self.env.pomo_size)

        self.env.step_state.BATCH_IDX = self.env.BATCH_IDX
        self.env.step_state.POMO_IDX = self.env.POMO_IDX

        self.model.eval()
        self.model.set_eval_type(eval_type)

        with torch.no_grad():
            reset_state, _, _ = self.env.reset()
            self.model.pre_forward(reset_state, pref)
            state, reward, done = self.env.pre_step()

            while not done:
                selected, _ = self.model(state)
                # shape: (batch, pomo)
                state, reward, done = self.env.step(selected)

        reward = - reward
        tch_reward = (pref * reward).sum(dim=2)

        reward = -reward
        tch_reward = -tch_reward

        tch_reward = tch_reward.reshape(aug_factor, batch_size, self.env.pomo_size)

        tch_reward_aug = rearrange(tch_reward, 'c b h -> b (c h)')
        _, max_idx_aug = tch_reward_aug.max(dim=1)
        max_idx_aug = max_idx_aug.reshape(max_idx_aug.shape[0], 1)
        max_reward_obj1 = rearrange(reward[:, :, 0].reshape(aug_factor, batch_size, self.env.pomo_size),
                                    'c b h -> b (c h)').gather(1, max_idx_aug)
        max_reward_obj2 = rearrange(reward[:, :, 1].reshape(aug_factor, batch_size, self.env.pomo_size),
                                    'c b h -> b (c h)').gather(1, max_idx_aug)

        aug_score = []
        aug_score.append(-max_reward_obj1.float())
        aug_score.append(-max_reward_obj2.float())

        return torch.stack(aug_score, 0).transpose(1, 0).squeeze(2).contiguous()


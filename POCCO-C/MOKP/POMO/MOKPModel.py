import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from MOELayer import MoE

class KPModel(nn.Module):

    def __init__(self, **model_params):
        super().__init__()
        self.model_params = model_params

        self.encoder = KP_Encoder(**model_params)
        self.decoder = KP_Decoder(**model_params)
        self.encoded_nodes_and_dummy = None
        self.encoded_nodes_kv = None
        self.encoded_nodes_q = None
        self.encoded_graph = None
        # shape: (batch, problem, EMBEDDING_DIM)

        self.aux_loss = 0

        embedding_dim = self.model_params['embedding_dim']
        hyper_hidden_embd_dim = 256
        self.hyper_fc2 = nn.Linear(embedding_dim, hyper_hidden_embd_dim, bias=True)
        self.hyper_fc3 = nn.Linear(hyper_hidden_embd_dim, embedding_dim, bias=True)

    def set_eval_type(self, eval_type):

        self.eval_type = eval_type

    def pre_forward(self, reset_state, pref):
        #self.encoded_nodes = self.encoder(reset_state.problems)
        # shape: (batch, problem, EMBEDDING_DIM)
        embedding_dim = self.model_params['embedding_dim']
        
        batch_size = reset_state.problems.size(0)
        problem_size = reset_state.problems.size(1)
        self.encoded_nodes_and_dummy = torch.Tensor(np.zeros((batch_size, problem_size+1, self.model_params['embedding_dim'])))
        self.encoded_nodes_and_dummy[:, :problem_size, :] = self.encoder(reset_state.problems, pref)
        self.encoded_nodes_q = self.encoded_nodes_and_dummy[:, :problem_size, :]

        self.encoded_graph = self.encoded_nodes_q.mean(dim=1, keepdim=True)

        # hyper_embd = self.hyper_fc1(pref)
        encoded_ps = position_encoding_init(batch_size, problem_size, embedding_dim, pref.device)
        EP_embedding = self.hyper_fc2(encoded_ps)
        EP_embed = self.hyper_fc3(EP_embedding)
        self.encoded_nodes_kv = self.encoded_nodes_q + EP_embed
        # shape: (batch, problem, EMBEDDING_DIM)
        
        self.decoder.set_kv(self.encoded_nodes_kv)

        self.aux_loss = 0

    def forward(self, state):
        batch_size = state.BATCH_IDX.size(0)
        pomo_size = state.BATCH_IDX.size(1)

        if state.current_node is None:
            selected = torch.arange(pomo_size)[None, :].expand(batch_size, pomo_size)
            prob = torch.ones(size=(batch_size, pomo_size))
        else:
            # shape: (batch, pomo, embedding)
            probs, mod_loss = self.decoder(self.encoded_graph, capacity=state.capacity, ninf_mask=state.ninf_mask)
            # shape: (batch, pomo, problem)
            self.aux_loss += mod_loss

            if self.training or self.eval_type == 'softmax':
                selected = probs.reshape(batch_size * pomo_size, -1).multinomial(1) \
                    .squeeze(dim=1).reshape(batch_size, pomo_size)
                # shape: (batch, pomo)

                prob = probs[state.BATCH_IDX, state.POMO_IDX, selected] \
                    .reshape(batch_size, pomo_size)
                # shape: (batch, pomo)

            else:
                selected = probs.argmax(dim=2)
                # shape: (batch, pomo)
                prob = None

        return selected, prob


def _get_encoding(encoded_nodes, node_index_to_pick):
    # encoded_nodes.shape: (batch, problem, embedding)
    # node_index_to_pick.shape: (batch, pomo)

    batch_size = node_index_to_pick.size(0)
    pomo_size = node_index_to_pick.size(1)
    embedding_dim = encoded_nodes.size(2)

    gathering_index = node_index_to_pick[:, :, None].expand(batch_size, pomo_size, embedding_dim)
    # shape: (batch, pomo, embedding)

    picked_nodes = encoded_nodes.gather(dim=1, index=gathering_index)
    # shape: (batch, pomo, embedding)

    return picked_nodes


########################################
# ENCODER
########################################

class KP_Encoder(nn.Module):
    def __init__(self, **model_params):
        super().__init__()
        self.model_params = model_params
        embedding_dim = self.model_params['embedding_dim']
        encoder_layer_num = self.model_params['encoder_layer_num']

        #self.embedding = nn.Linear(2, embedding_dim)
        self.embedding = nn.Linear(3, embedding_dim)
        self.embedding_pref = nn.Linear(2, embedding_dim)
        self.layers = nn.ModuleList([EncoderLayer(**model_params) for _ in range(encoder_layer_num)])

    def forward(self, data, pref):
        # data.shape: (batch, problem, 2)

        embedded_input = self.embedding(data)
        # shape: (batch, problem, embedding)
        embedded_pref = self.embedding_pref(pref)
        # shape: (batch, embedding)

        out = torch.cat((embedded_input, embedded_pref[:, None, :]), -2)
        for layer in self.layers:
            out = layer(out)

        return out[:, :-1]


class EncoderLayer(nn.Module):
    def __init__(self, **model_params):
        super().__init__()
        self.model_params = model_params
        embedding_dim = self.model_params['embedding_dim']
        head_num = self.model_params['head_num']
        qkv_dim = self.model_params['qkv_dim']

        self.Wq1 = nn.Linear(embedding_dim, head_num * qkv_dim, bias=False)
        self.Wk1 = nn.Linear(embedding_dim, head_num * qkv_dim, bias=False)
        self.Wv1 = nn.Linear(embedding_dim, head_num * qkv_dim, bias=False)

        self.Wq2 = nn.Linear(embedding_dim, head_num * qkv_dim, bias=False)
        self.Wk2 = nn.Linear(embedding_dim, head_num * qkv_dim, bias=False)
        self.Wv2 = nn.Linear(embedding_dim, head_num * qkv_dim, bias=False)

        self.multi_head_combine = nn.Linear(head_num * qkv_dim, embedding_dim)

        self.addAndNormalization1 = Add_And_Normalization_Module(**model_params)
        self.feedForward = Feed_Forward_Module(**model_params)
        self.addAndNormalization2 = Add_And_Normalization_Module(**model_params)

    def forward(self, input1):
        # input.shape: (batch, problem, EMBEDDING_DIM)
        head_num = self.model_params['head_num']
        embed_nodes = input1[:, :-1, :]  # (batch, problem, embedding_dim)
        pref_node = input1[:, -1, :][:, None, :]  # (batch, 1, embedding_dim)

        q1 = reshape_by_heads(self.Wq1(input1), head_num=head_num)
        k1 = reshape_by_heads(self.Wk1(input1), head_num=head_num)
        v1 = reshape_by_heads(self.Wv1(input1), head_num=head_num)
        # q shape: (batch, HEAD_NUM, problem, KEY_DIM)

        q2 = reshape_by_heads(self.Wq2(embed_nodes), head_num=head_num)
        k2 = reshape_by_heads(self.Wk2(pref_node), head_num=head_num)
        v2 = reshape_by_heads(self.Wv2(pref_node), head_num=head_num)

        out_concat = multi_head_attention(q1, k1, v1)
        # shape: (batch, problem, HEAD_NUM*KEY_DIM)

        add_concat = multi_head_attention(q2, k2, v2)
        out_concat[:, :-1] = out_concat[:, :-1] + add_concat

        multi_head_out = self.multi_head_combine(out_concat)
        # shape: (batch, problem, EMBEDDING_DIM)

        out1 = self.addAndNormalization1(input1, multi_head_out)
        out2 = self.feedForward(out1)
        out3 = self.addAndNormalization2(out1, out2)

        return out3
        # shape: (batch, problem, EMBEDDING_DIM)


########################################
# DECODER
########################################

class KP_Decoder(nn.Module):
    def __init__(self, **model_params):
        super().__init__()
        self.model_params = model_params
        embedding_dim = self.model_params['embedding_dim']
        head_num = self.model_params['head_num']
        qkv_dim = self.model_params['qkv_dim']
        
        self.hyper_Wq = nn.Linear(1 + embedding_dim, head_num * qkv_dim, bias=False)
        self.hyper_Wk = nn.Linear(embedding_dim, head_num * qkv_dim, bias=False)
        self.hyper_Wv = nn.Linear(embedding_dim, head_num * qkv_dim, bias=False)
        self.hyper_multi_head_combine = nn.Linear(head_num * qkv_dim, embedding_dim, bias=False)

        self.moed_layer = MoE(input_size=embedding_dim, output_size=embedding_dim,
                              num_experts=self.model_params['num_experts'],
                              hidden_size=self.model_params['ff_hidden_dim'], k=self.model_params['topk'], T=1.0,
                              noisy_gating=True,
                              routing_level=self.model_params['routing_level'],
                              routing_method=self.model_params['routing_method'], moed_model="MLP")
        self.addAndNormalization = Add_And_Normalization_Module(**model_params)
        
        self.k = None  # saved key, for multi-head attention
        self.v = None  # saved value, for multi-head_attention
        self.single_head_key = None  # saved, for single-head attention
        
        
    def set_kv(self, encoded_nodes):
        # encoded_nodes.shape: (batch, problem, embedding)
        head_num = self.model_params['head_num']

        self.k = reshape_by_heads(self.hyper_Wk(encoded_nodes), head_num=head_num)
        self.v = reshape_by_heads(self.hyper_Wv(encoded_nodes), head_num=head_num)
        
        self.single_head_key = encoded_nodes.transpose(1, 2)
     
    def forward(self, graph, capacity, ninf_mask):
        # encoded_last_node.shape: (batch, pomo, embedding)
        # ninf_mask.shape: (batch, pomo, problem)
        embedding_dim = self.model_params['embedding_dim']
        head_num = self.model_params['head_num']
        qkv_dim = self.model_params['qkv_dim']
        
        batch_size = capacity.size(0)
        group_size = capacity.size(1)

        #  Multi-Head Attention
        #######################################################
        input1 = graph.expand(batch_size, group_size, embedding_dim)
        input2 = capacity[:, :, None]
        input_cat = torch.cat((input1, input2), dim=2)
        
        #  Multi-Head Attention
        #######################################################
        q = reshape_by_heads(self.hyper_Wq(input_cat), head_num = head_num)
       
        out_concat = multi_head_attention(q, self.k, self.v, ninf_mask=ninf_mask)
       
        mh_atten_out = self.hyper_multi_head_combine(out_concat)
        # shape: (batch, pomo, embedding)

        mh_atten, moed_loss = self.moed_layer(mh_atten_out)
        mh_atten_out = self.addAndNormalization(mh_atten_out, mh_atten)

        #  Single-Head Attention, for probability calculation
        #######################################################
        score = torch.matmul(mh_atten_out, self.single_head_key)
        # shape: (batch, pomo, problem)

        sqrt_embedding_dim = self.model_params['sqrt_embedding_dim']
        logit_clipping = self.model_params['logit_clipping']

        score_scaled = score / sqrt_embedding_dim
        # shape: (batch, pomo, problem)

        score_clipped = logit_clipping * torch.tanh(score_scaled)

        #score_masked = score_clipped + ninf_mask
        if ninf_mask is None:
            score_masked = score_clipped
        else:
            score_masked = score_clipped + ninf_mask

        probs = F.softmax(score_masked, dim=2)
        # shape: (batch, pomo, problem)

        return probs, moed_loss


########################################
# NN SUB CLASS / FUNCTIONS
########################################

def reshape_by_heads(qkv, head_num):
    # q.shape: (batch, n, head_num*key_dim)   : n can be either 1 or PROBLEM_SIZE

    batch_s = qkv.size(0)
    n = qkv.size(1)

    q_reshaped = qkv.reshape(batch_s, n, head_num, -1)
    # shape: (batch, n, head_num, key_dim)

    q_transposed = q_reshaped.transpose(1, 2)
    # shape: (batch, head_num, n, key_dim)

    return q_transposed

def multi_head_attention(q, k, v, ninf_mask=None):
    # q shape = (batch, head_num, n, key_dim)   : n can be either 1 or group
    # k,v shape = (batch, head_num, problem, key_dim)
    # ninf_mask.shape = (batch, group, problem)

    batch_s = q.size(0)
    head_num = q.size(1)
    n = q.size(2)
    key_dim = q.size(3)
    problem_s = k.size(2)

    score = torch.matmul(q, k.transpose(2, 3))
    # shape = (batch, head_num, n, TSP_SIZE)

    score_scaled = score / torch.sqrt(torch.tensor(key_dim, dtype=torch.float))
    if ninf_mask is not None:
        score_scaled = score_scaled + ninf_mask[:, None, :, :].expand(batch_s, head_num, n, problem_s)

    weights = nn.Softmax(dim=3)(score_scaled)
    # shape = (batch, head_num, n, problem)

    out = torch.matmul(weights, v)
    # shape = (batch, head_num, n, key_dim)

    out_transposed = out.transpose(1, 2)
    # shape = (batch, n, head_num, key_dim)

    out_concat = out_transposed.reshape(batch_s, n, head_num * key_dim)
    # shape = (batch, n, head_num*key_dim)

    return out_concat


class Add_And_Normalization_Module(nn.Module):
    def __init__(self, **model_params):
        super().__init__()
        embedding_dim = model_params['embedding_dim']
        self.norm = nn.InstanceNorm1d(embedding_dim, affine=True, track_running_stats=False)

    def forward(self, input1, input2):
        # input.shape: (batch, problem, embedding)

        added = input1 + input2
        # shape: (batch, problem, embedding)

        transposed = added.transpose(1, 2)
        # shape: (batch, embedding, problem)

        normalized = self.norm(transposed)
        # shape: (batch, embedding, problem)

        back_trans = normalized.transpose(1, 2)
        # shape: (batch, problem, embedding)

        return back_trans


class Feed_Forward_Module(nn.Module):
    def __init__(self, **model_params):
        super().__init__()
        embedding_dim = model_params['embedding_dim']
        ff_hidden_dim = model_params['ff_hidden_dim']

        self.W1 = nn.Linear(embedding_dim, ff_hidden_dim)
        self.W2 = nn.Linear(ff_hidden_dim, embedding_dim)

    def forward(self, input1):
        # input.shape: (batch, problem, embedding)

        return self.W2(F.relu(self.W1(input1)))


def position_encoding_init(batch_szie, n_position, emb_dim, device):
    ''' Init the sinusoid position encoding table '''

    # keep dim 0 for padding token position encoding zero vector
    position_enc = torch.FloatTensor(np.array([
        [pos / np.power(10000, 2 * (j // 2) / emb_dim) for j in range(emb_dim)]
        if pos != 0 else np.zeros(emb_dim) for pos in range(300)])).to(device)

    position_enc[1:, 0::2] = torch.sin(position_enc[1:, 0::2])  # dim 2i
    position_enc[1:, 1::2] = torch.cos(position_enc[1:, 1::2])  # dim 2i+1

    # n_size = n_position // 10
    # position_encoding = position_enc[n_size]
    position_encoding = position_enc[n_position - 1]
    return position_encoding[None, None, :].expand(batch_szie, 1, emb_dim)
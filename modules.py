import torch.nn as nn
import torch
import torch.nn.functional as F
from torch.optim.optimizer import Optimizer
import torch.optim as optim
import numpy as np


class Highway(nn.Module):
    def __init__(self, input_dim, layers):
        super(Highway, self).__init__()
        self.input_dim = input_dim
        self.layers = nn.ModuleList([nn.Linear(input_dim, input_dim * 2)
                                     for _ in range(layers)])
        self.reset_parameters()

    def reset_parameters(self):
        for layer in self.layers:
            nn.init.normal_(layer.weight, std=0.02)
            nn.init.constant_(layer.bias[self.input_dim:], 1)
            nn.init.constant_(layer.bias[:self.input_dim], 0)

    def forward(self, x):
        for layer in self.layers:
            new_x = layer(x)
            new_x, gate = new_x.chunk(2, dim=-1)
            new_x = F.relu(new_x)
            gate = torch.sigmoid(gate)
            x = gate * x + (1 - gate) * new_x
        return x


class CNNEncoder(nn.Module):
    def __init__(self, filters, input_dim, output_dim, highway_layers=1):
        super(CNNEncoder, self).__init__()
        self.convolutions = nn.ModuleList()
        for width, out_c in filters:
            self.convolutions.append(nn.Conv1d(input_dim, out_c, kernel_size=width))
        final_dim = sum(f[1] for f in filters)
        self.highway = Highway(final_dim, highway_layers)
        self.out_proj = nn.Linear(final_dim, output_dim)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.normal_(self.out_proj.weight, std=0.02)
        nn.init.constant_(self.out_proj.bias, 0.)

    def forward(self, input):
        # input: batch_size x seq_len x input_dim
        x  = input.transpose(1, 2)
        conv_result = []
        for i, conv in enumerate(self.convolutions):
            y = conv(x)
            y, _ = torch.max(y, -1)
            y = F.relu(y)
            conv_result.append(y)

        conv_result = torch.cat(conv_result, dim=-1)
        conv_result = self.highway(conv_result)
        return self.out_proj(conv_result) #  batch_size x output_dim


def linear(input_size, output_size):
    return FFNN(0, input_size, -1, output_size, None)


class FFNN(nn.Module):
    def __init__(self, num_hidden_layers, input_size, hidden_size, output_size, dropout, output_weights_initializer=None):
        super(FFNN, self).__init__()
        self.num_hidden_layers = num_hidden_layers
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.dropout = nn.Dropout(p=dropout) if dropout is not None and dropout > 0.0 else None
        self.linear = []
        for i in range(self.num_hidden_layers):
            if i == 0:
                self.linear.append(nn.Linear(input_size, hidden_size))
            else:
                self.linear.append(nn.Linear(hidden_size, hidden_size))
        last_input_size = hidden_size if self.num_hidden_layers > 0 else input_size
        self.linear.append(nn.Linear(last_input_size, output_size))
        self.linear = nn.ModuleList(self.linear)

    def forward(self, inputs):
        current_inputs = inputs
        for i in range(self.num_hidden_layers):
            current_outputs = F.relu(self.linear[i](current_inputs))
            if self.dropout is not None:
                current_outputs = self.dropout(current_outputs)
            current_inputs = current_outputs
        outputs = self.linear[self.num_hidden_layers](current_inputs)
        return outputs


class TokenEncoder(nn.Module):
    def __init__(self, token_vocab, char_vocab, char_dim, token_dim, embed_dim, filters, char2token_dim, dropout):
        super(TokenEncoder, self).__init__()
        self.char_embed = AMREmbedding(char_vocab, char_dim)
        self.token_embed = AMREmbedding(token_vocab, token_dim)
        self.char2token = CNNEncoder(filters, char_dim, char2token_dim)
        tot_dim = char2token_dim + token_dim
        self.out_proj = nn.Linear(tot_dim, embed_dim)
        self.char_dim = char_dim
        self.token_dim = token_dim
        self.dropout = nn.Dropout(dropout)

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.normal_(self.out_proj.weight, std=0.02)
        nn.init.constant_(self.out_proj.bias, 0.)

    def forward(self, token_input, char_input):
        seq_len, bsz, _ = char_input.size()
        char_repr = self.char_embed(char_input.view(seq_len * bsz, -1))
        char_repr = self.char2token(char_repr).view(seq_len, bsz, -1)
        token_repr = self.token_embed(token_input)

        token = self.dropout(torch.cat([char_repr, token_repr], -1))
        token = self.out_proj(token)
        return token


def AMREmbedding(vocab, embedding_dim):
    return Embedding(vocab.size, embedding_dim, vocab.padding_idx)


def Embedding(num_embeddings, embedding_dim, padding_idx):
    m = nn.Embedding(num_embeddings, embedding_dim, padding_idx=padding_idx)
    nn.init.normal_(m.weight, std=0.02)
    nn.init.constant_(m.weight[padding_idx], 0)
    return m

# emb: [batch, seqlen, emb]
# indices: [batch, x, index] or [batch, index]

def batch_gather(emb, indices, device):
    batch_size, seq_len = list(emb.size())[:2]
    if len(emb.size()) > 2:
        assert len(emb.size()) == 3
        emb_size = shape(emb, 2)
    else:
        emb_size = 1
    flattened_emb = emb.contiguous().view(batch_size * seq_len, emb_size) # [batch_size * seqlen, emb]

    ori_indices = indices
    assert len(indices.size()) > 1 and shape(indices, 0) == batch_size
    if len(indices.size()) == 2:
        x = 1
        num_indices = shape(indices, 1)
        indices = indices.view(batch_size, x, num_indices)
    elif len(indices.size()) == 3:
        x = shape(indices, 1)
        num_indices = shape(indices, 2)
    else:
        assert False

    offset = torch.arange(batch_size).view(batch_size, 1, 1).expand(-1, x, num_indices) * seq_len # [batch_size, x, index]

    offset = offset.to(device)

    indices = (indices + offset).view(batch_size * x * num_indices) # [batch_size * x * num_indices]
    gathered = torch.index_select(flattened_emb, 0, indices).view(batch_size, x, num_indices, emb_size)
    if len(ori_indices.size()) == 2:
        gathered = gathered.squeeze(dim=1)
    if len(emb.size()) == 2:
        gathered = gathered.squeeze(dim=-1)
    return gathered

def shape(x, dim):
    return list(x.size())[dim]

def mean(x):
    if len(x) == 0:
        return sum(x) / (len(x) + 1)
    return sum(x) / len(x)

def contain_nan(x):
    return (x != x).any().item()

def clip_and_normalize(word_probs, epsilon):
    word_probs = torch.clamp(word_probs, epsilon, 1.0 - epsilon)
    return word_probs / word_probs.sum(dim=-1, keepdim=True)

def get_batch_graph_data(data):
    a = 1
    # batch=1, node_num, others



    return data

class AdamWeightDecayOptimizer(Optimizer):
    """A basic Adam optimizer that includes "correct" L2 weight decay.
    https://github.com/google-research/bert/blob/master/optimization.py
    https://raw.githubusercontent.com/pytorch/pytorch/v1.0.0/torch/optim/adam.py"""
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8,
                 weight_decay=0, amsgrad=False):
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter at index 0: {}".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter at index 1: {}".format(betas[1]))
        defaults = dict(lr=lr, betas=betas, eps=eps,
                        weight_decay=weight_decay, amsgrad=amsgrad)
        super(AdamWeightDecayOptimizer, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(AdamWeightDecayOptimizer, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault('amsgrad', False)

    def step(self, closure=None):
        """Performs a single optimization step.

        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError('Adam does not support sparse gradients, please consider SparseAdam instead')
                amsgrad = group['amsgrad']

                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    # Exponential moving average of gradient values
                    state['exp_avg'] = torch.zeros_like(p.data)
                    # Exponential moving average of squared gradient values
                    state['exp_avg_sq'] = torch.zeros_like(p.data)
                    if amsgrad:
                        # Maintains max of all exp. moving avg. of sq. grad. values
                        state['max_exp_avg_sq'] = torch.zeros_like(p.data)

                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                if amsgrad:
                    max_exp_avg_sq = state['max_exp_avg_sq']
                beta1, beta2 = group['betas']

                state['step'] += 1

                # Decay the first and second moment running average coefficient
                exp_avg.mul_(beta1).add_(1 - beta1, grad)
                exp_avg_sq.mul_(beta2).addcmul_(1 - beta2, grad, grad)
                if amsgrad:
                    # Maintains the maximum of all 2nd moment running avg. till now
                    torch.max(max_exp_avg_sq, exp_avg_sq, out=max_exp_avg_sq)
                    # Use the max. for normalizing running avg. of gradient
                    denom = max_exp_avg_sq.sqrt().add_(group['eps'])
                else:
                    denom = exp_avg_sq.sqrt().add_(group['eps'])

                # Just adding the square of the weights to the loss function is *not*
                # the correct way of using L2 regularization/weight decay with Adam,
                # since that will interact with the m and v parameters in strange ways.
                #
                # Instead we want ot decay the weights in a manner that doesn't interact
                # with the m/v parameters. This is equivalent to adding the square
                # of the weights to the loss with plain (non-momentum) SGD.
                update = (exp_avg/denom).add_(group['weight_decay'], p.data)
                p.data.add_(-group['lr'], update)
        return loss


def get_aligment_embed(token_reps, alignment, device):

    align_reps = []
    temp = torch.zeros([1, 1, token_reps.shape[2]]).to(device)
    # use average reps or the first one
    for i in alignment:
        if i == -1:
            align_reps.append(temp)
        elif not isinstance(i, list):
            align_reps.append(token_reps[:, i, :].unsqueeze(1))
        else:
            temp2 = []
            for j in i:
                temp2.append(token_reps[:, j, :].unsqueeze(1))
            align_reps.append(torch.cat(temp2, 1).mean(1).unsqueeze(1))
    return torch.cat(align_reps, 1)


def optimizer(args, parameters):
    if args.optimizer.lower() == "adam":
        return optim.Adam(filter(lambda p: p.requires_grad, parameters), lr=args.learning_rate,
                          weight_decay=args.weight_decay)
    elif args.optimizer.lower() == "sgd":
        return optim.SGD(filter(lambda p: p.requires_grad, parameters), lr=args.learning_rate,
                         weight_decay=args.weight_decay)
    else:
        assert False, "no application for the optimizer"


class focal_loss(nn.Module):
    def __init__(self, alpha=0.5, gamma=2, num_classes=5, size_average=True):
        """
        focal_loss损失函数, -α(1-yi)**γ *ce_loss(xi,yi)
        步骤详细的实现了 focal_loss损失函数.
        :param alpha:   阿尔法α,类别权重.      当α是列表时,为各类别权重,当α为常数时,类别权重为[α, 1-α, 1-α, ....],常用于 目标检测算法中抑制背景类 , retainnet中设置为0.25
        :param gamma:   伽马γ,难易样本调节参数. retainnet中设置为2
        :param num_classes:     类别数量
        :param size_average:    损失计算方式,默认取均值
        """

        super(focal_loss, self).__init__()
        self.size_average = size_average
        if isinstance(alpha, list):
            assert len(alpha) == num_classes   # α可以以list方式输入,size:[num_classes] 用于对不同类别精细地赋予权重
            print("Focal_loss alpha = {}, 将对每一类权重进行精细化赋值".format(alpha))
            self.alpha = torch.Tensor(alpha)
        else:
            assert alpha < 1   # 如果α为一个常数,则降低第一类的影响,在目标检测中为第一类
            print(" --- Focal_loss alpha = {} , reduce first class --- ".format(alpha))
            self.alpha = torch.zeros(num_classes)
            self.alpha[0] += alpha
            self.alpha[1:] += (1-alpha)  # α 最终为 [ α, 1-α, 1-α, 1-α, 1-α, ...] size:[num_classes]
        self.gamma = gamma

    def forward(self, preds, labels):
        """
        focal_loss损失计算
        :param preds:   预测类别. size:[B,N,C] or [B,C]    分别对应与检测与分类任务, B 批次, N检测框数, C类别数
        :param labels:  实际类别. size:[B,N] or [B]
        :return:
        """
        # assert preds.dim()==2 and labels.dim()==1
        preds = preds.view(-1, preds.size(-1))
        self.alpha = self.alpha.to(preds.device)
        preds_softmax = F.softmax(preds, dim=1)   # 这里并没有直接使用log_softmax, 因为后面会用到softmax的结果(当然你也可以使用log_softmax,然后进行exp操作)
        preds_logsoft = torch.log(preds_softmax)
        preds_softmax = preds_softmax.gather(1, labels.view(-1, 1))   # 这部分实现nll_loss ( crossempty = log_softmax + nll )
        preds_logsoft = preds_logsoft.gather(1, labels.view(-1, 1))
        self.alpha = self.alpha.gather(0, labels.view(-1))
        loss = -torch.mul(torch.pow((1-preds_softmax), self.gamma), preds_logsoft)   # torch.pow((1-preds_softmax), self.gamma) 为focal loss中 (1-pt)**γ
        loss = torch.mul(self.alpha, loss.t())
        if self.size_average:
            loss = loss.mean()
        else:
            loss = loss.sum()
        return loss
import torch
import torch.nn as nn
from torch.nn import functional as F
from model import CausalSelfAttention


def reduce_rank_linear(full_rank_wts, new_rank):
    # The weight matrix is the transpose of the parameter
    full_rank_wts = torch.transpose(full_rank_wts, 0,1)

    # SVD only works when num rows > num cols
    assert full_rank_wts.shape[0] >= full_rank_wts.shape[1]

    U, S, V = torch.linalg.svd(full_rank_wts, full_matrices = False)
    low_rank_left = U[:,:new_rank] @ torch.diag(S[:new_rank]) 
    low_rank_right = V[:new_rank,:]

    # take the inverse to get back what nn.Linear needs
    tmp = torch.transpose(low_rank_left, 0,1)
    low_rank_left = torch.transpose(low_rank_right, 0,1)
    low_rank_right = tmp
    return low_rank_left, low_rank_right, None

def make_low_rank_nn_linear(original, new_rank):
    low_rank_weights_left, low_rank_weights_right = make_low_rank(original, new_rank)
    low_rank_weights = low_rank_weights_left @ low_rank_weights_right
    in_features = low_rank_weights.shape[1]
    out_features = low_rank_weights.shape[0]
    low_rank_module = torch.nn.Linear(in_features = in_features, out_features = out_features, bias = False)
    lr_state_dict = low_rank_module.state_dict()
    lr_state_dict['weight'] = low_rank_weights
    low_rank_module.load_state_dict(lr_state_dict)
    return low_rank_module

def make_low_rank(original, new_rank):
    assert isinstance(original, torch.nn.Linear)
    assert 'bias' not in original.state_dict()

    wts_shape = original.state_dict()['weight'].shape
    # The weights are the transpose of the matrix that multiplies the input,
    # per the torch documentation for nn.Linear.
    in_features = wts_shape[1]
    out_features = wts_shape[0]

    # Selection of largest singular values in S of SVD only works to
    # approximate the matrix when rows of matrix is > cols. 
    #
    # Weights of linear is transpose of multiplying matrix.
    must_flip = in_features < out_features
    full_rank_wts = original.state_dict()['weight']

    if must_flip:
        full_rank_wts = torch.transpose(full_rank_wts, 0, 1)

    low_rank_weights_left, low_rank_weights_right, low_rank_bias = reduce_rank_linear(full_rank_wts, new_rank = new_rank)

    if must_flip:
        low_rank_weights_right2 = torch.transpose(low_rank_weights_left, 0,1)
        low_rank_weights_left2 = torch.transpose(low_rank_weights_right, 0,1)
    else:
        low_rank_weights_left2 = low_rank_weights_left
        low_rank_weights_right2 = low_rank_weights_right

    return low_rank_weights_left2, low_rank_weights_right2

def select_diag_elements(original, block_rows, block_cols):
    num_diag_blocks = original.shape[0] // block_rows
    assert num_diag_blocks == original.shape[1] // block_cols

    source_row = 0
    source_col = 0
    diag = torch.zeros(size=(num_diag_blocks, block_rows, block_cols))
    for i in range(num_diag_blocks):
        diag[i,:,:] = original[source_row:source_row+block_rows, source_col:source_col+block_cols]
        source_row += block_rows
        source_col += block_cols

    return diag

def form_off_diag(original, block_rows, block_cols):
    diag_blocks = select_diag_elements(original, block_rows, block_cols)
    assert len(diag_blocks.shape) == 3
    block_diag = torch.block_diag(*diag_blocks)
    return original - block_diag

def LRPBD_iterated_soln(original, new_rank, block_size):
    assert isinstance(original, torch.Tensor)
    # original is used in expression input @ original
    # Rows must exceed cols for reduced SVD
    assert len(original.shape) == 2
    # assert original.shape[0] >= original.shape[1]

    row_count = original.shape[0]
    col_count = original.shape[1]

    assert row_count % block_size == 0
    assert col_count % block_size == 0

    if row_count > col_count:
        num_block_rows = (block_size * row_count) // col_count
        num_block_cols = block_size
    else:
        num_block_rows = block_size
        num_block_cols = (block_size * col_count) // row_count

    off_diag_mask = torch.ones_like(original)
    off_diag_mask = off_diag_mask - torch.block_diag(*select_diag_elements(off_diag_mask, num_block_rows, num_block_cols))

    off_diag_orig = off_diag_mask * original
    on_diag_mask = torch.ones_like(original)
    on_diag_mask = torch.block_diag(*select_diag_elements(on_diag_mask, num_block_rows, num_block_cols))

    #################################################################
    # Selection of largest singular values in S of SVD only works to
    # approximate the matrix when rows of matrix is > cols. 
    #
    # Weights of linear is transpose of multiplying matrix.
    must_flip = row_count < col_count

    if must_flip:
        off_diag_mask = torch.transpose(off_diag_mask, 0, 1)
        on_diag_mask = torch.transpose(on_diag_mask,0,1)
        original = torch.transpose(original,0,1)

    last = torch.zeros_like(original)
    last2 = last

    #Srebro, N., & Jaakkola, T. (2003, August). Weighted low-rank approximations. In ICML (Vol. 3, pp. 720-727).    
    for i in range(50):
        off_diag_orig = off_diag_mask * original
        # low_rank_weights_left, low_rank_weights_right, low_rank_bias = reduce_rank_linear(full_rank_wts, new_rank = new_rank)
        # SVD for low-rank approximation of the portions off the block diagonal.
        u,s,v = torch.linalg.svd(off_diag_orig + on_diag_mask * last, full_matrices=False)

        low_rank_weights_left = u[:,:new_rank] @ torch.diag(s[:new_rank])
        low_rank_weights_right = v[:new_rank,:]
        last = low_rank_weights_left @ low_rank_weights_right
        delta = last - last2
        last2 = last
        # print('delta:{0}'.format(torch.linalg.matrix_norm(delta)))


    if must_flip:
        low_rank_weights_right2 = torch.transpose(low_rank_weights_left, 0,1)
        low_rank_weights_left2 = torch.transpose(low_rank_weights_right, 0,1)
        original = torch.transpose(original,0,1)
    else:
        low_rank_weights_left2 = low_rank_weights_left
        low_rank_weights_right2 = low_rank_weights_right

    off_diag_estimate = low_rank_weights_left2 @ low_rank_weights_right2

    # Choose block diag parameters such that, when added to the off-diag low rank approx,
    # yields exactly the values in the original matrix.
    diag_elements = select_diag_elements(original, block_rows=num_block_rows, block_cols=num_block_cols)
    block_diag_parameters = diag_elements - select_diag_elements(off_diag_estimate,num_block_rows, num_block_cols)
    return low_rank_weights_left2, low_rank_weights_right2, block_diag_parameters


def LRPBD_from_matrix(original, new_rank, block_size):
    assert isinstance(original, torch.Tensor)
    # original is used in expression input @ original
    # Rows must exceed cols for reduced SVD
    assert len(original.shape) == 2
    # assert original.shape[0] >= original.shape[1]

    row_count = original.shape[0]
    col_count = original.shape[1]

    assert row_count % block_size == 0
    assert col_count % block_size == 0

    # print('original:')
    # print(original)

    if row_count > col_count:
        num_block_rows = block_size * row_count // col_count
        num_block_cols = block_size
    else:
        num_block_rows = block_size
        num_block_cols = block_size * col_count // row_count

    diag_elements = select_diag_elements(original, block_rows=num_block_rows, block_cols=num_block_cols)
    off_diag_mask = torch.ones_like(original)
    off_diag_mask = off_diag_mask - torch.block_diag(*select_diag_elements(off_diag_mask, num_block_rows, num_block_cols))

    off_diag_orig = off_diag_mask * original

    #################################################################
    # Selection of largest singular values in S of SVD only works to
    # approximate the matrix when rows of matrix is > cols. 
    #
    # Weights of linear is transpose of multiplying matrix.
    must_flip = row_count < col_count

    if must_flip:
        off_diag_orig = torch.transpose(off_diag_orig, 0, 1)

    # low_rank_weights_left, low_rank_weights_right, low_rank_bias = reduce_rank_linear(full_rank_wts, new_rank = new_rank)
    # SVD for low-rank approximation of the portions off the block diagonal.
    u,s,v = torch.linalg.svd(off_diag_orig, full_matrices=False)

    low_rank_weights_left = u[:,:new_rank] @ torch.diag(s[:new_rank])
    low_rank_weights_right = v[:new_rank,:]


    if must_flip:
        low_rank_weights_right2 = torch.transpose(low_rank_weights_left, 0,1)
        low_rank_weights_left2 = torch.transpose(low_rank_weights_right, 0,1)
    else:
        low_rank_weights_left2 = low_rank_weights_left
        low_rank_weights_right2 = low_rank_weights_right

    off_diag_estimate = low_rank_weights_left2 @ low_rank_weights_right2

    # Choose block diag parameters such that, when added to the off-diag low rank approx,
    # yields exactly the values in the original matrix.
    block_diag_parameters = diag_elements - select_diag_elements(off_diag_estimate,num_block_rows, num_block_cols)
    return low_rank_weights_left2, low_rank_weights_right2, block_diag_parameters

def make_LORI_FC_from_matrix(original_wts, new_rank, block_size, bias):
    assert bias is False
    wts_to_reduce = torch.transpose(original_wts,0,1)
    input_features = original_wts.shape[1]
    output_features = original_wts.shape[0]

    # left_matrix, right_matrix, block_diag_params = LRPBD_from_matrix(wts_to_reduce, 
    #                                                                     new_rank = new_rank, 
    #                                                                     block_size=block_size)
    left_matrix, right_matrix, block_diag_params = LRPBD_iterated_soln(wts_to_reduce, 
                                                                       new_rank = new_rank, 
                                                                       block_size=block_size)
    lori_fc = LORI_FC(input_features=input_features, output_features=output_features, rank=new_rank, diag_blocksize=block_size, bias=bias )
    sd = lori_fc.state_dict()
    sd['diag_params'] = block_diag_params
    sd['left.weight'] = torch.transpose(left_matrix,0,1)
    sd['right.weight'] = torch.transpose(right_matrix,0,1)
    lori_fc.load_state_dict(sd)
    return lori_fc


def make_LORI_FC_from_module(original, new_rank, block_size, bias):
    assert isinstance(original, torch.nn.Linear)
    assert 'bias' not in original.state_dict()
    original_wts = original.state_dict()['weight']
    return make_LORI_FC_from_matrix(original_wts=original_wts, new_rank=new_rank, block_size=block_size, bias=bias)

def set_state(target, source):
    assert isinstance(target, type(source))
    assert isinstance(source, type(target))
    target.load_state_dict(source.state_dict())

def reduce_rank_MLP(mlp, lori_mlp, lori_config):
    fc1 = make_LORI_FC_from_module(original=mlp.fc1, new_rank=lori_config.n_fc_bottleneck, )

def reduce_rank_causal_attn(caus_attn, lori_ca, lori_config):
    assert lori_config.bias == False
    c_wts = caus_attn.state_dict()['c_attn.weight']
    assert c_wts.shape[0] == 3*c_wts.shape[1]
    assert lori_config.n_embd == c_wts.shape[1]
    assert isinstance(caus_attn, CausalSelfAttention)
    assert isinstance(lori_ca, LORICausalSelfAttention)

    q_attn, k_attn, v_attn = c_wts.split(lori_config.n_embd, dim=0)

    # add bias as last column in matrix (see torch nn.Linear).
    if lori_config.bias:
        q_attn_bias, k_attn_bias, v_attn_bias = c_attn_bias.split(config.n_embd, dim=0)
        q_attn_aug = augment_with_bias(q_attn, q_attn_bias)
        k_attn_aug = augment_with_bias(k_attn, k_attn_bias)
        v_attn_aug = augment_with_bias(v_attn, v_attn_bias)
    else:
        q_attn_bias = None
        k_attn_bias = None
        v_attn_bias = None
        q_attn_aug = q_attn
        k_attn_aug = k_attn
        v_attn_aug = v_attn

    lori_q_attn = make_LORI_FC_from_matrix(original_wts=q_attn_aug, new_rank=lori_config.n_q, block_size=lori_config.n_fc_diagblock, bias=lori_config.bias)
    lori_k_attn = make_LORI_FC_from_matrix(original_wts=k_attn_aug, new_rank=lori_config.n_k, block_size=lori_config.n_fc_diagblock, bias=lori_config.bias)
    lori_v_attn = make_LORI_FC_from_matrix(original_wts=v_attn_aug, new_rank=lori_config.n_v, block_size=lori_config.n_fc_diagblock, bias=lori_config.bias)

    set_state(lori_ca.c_attn_q, lori_q_attn)
    set_state(lori_ca.c_attn_k, lori_k_attn)
    set_state(lori_ca.c_attn_v, lori_v_attn)

    lori_c_proj = make_LORI_FC_from_module(original=caus_attn.c_proj, new_rank=lori_config.n_fc_bottleneck, 
                                           block_size=lori_config.n_fc_diagblock, bias=lori_config.bias)

    set_state(lori_ca.c_proj, lori_c_proj)

    # q_U, q_S, q_V = torch.linalg.svd(q_attn_aug, full_matrices = False)
    # k_U, k_S, k_V = torch.linalg.svd(k_attn_aug, full_matrices = False)
    # # v_U, v_S, v_V = torch.linalg.svd(v_attn_aug, full_matrices = True)

    # # Bottlenecks are (per-head size) * (number of heads)
    # q_b = config.n_q * config.n_head
    # k_b = config.n_k * config.n_head

    # # Preserve the largest singular values in the approximation.
    # # q_attn_rr_dot = q_U[:,:q_b] @ torch.diag(q_S[:q_b]) @ q_V[:q_b,:] @ torch.transpose(k_V[:k_b,:], 0,1)
    # # q_attn_rr_dot_bias = None
    # # k_attn_rr_dot = k_U[:,:k_b] @ torch.diag(k_S[:k_b]) 
    # # k_attn_rr_dot_bias = None
    # q_attn_rr_dot = q_U[:,:q_b] @ torch.diag(q_S[:q_b]) @ q_V[:q_b,:]
    # q_attn_rr_dot_bias = None
    # k_attn_rr_dot = k_U[:,:k_b] @ torch.diag(k_S[:k_b]) @ k_V[:k_b,:]
    # k_attn_rr_dot_bias = None

class LORICausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        assert config.n_embd % config.n_k == 0
        assert config.n_embd % config.n_q == 0
        assert config.n_embd % config.n_v == 0
        assert config.n_q == config.n_k
        # self.c_attn_q = nn.Linear(config.n_embd, config.n_head * config.n_q, bias=config.bias)
        # self.c_attn_k = nn.Linear(config.n_embd, config.n_head * config.n_k, bias=config.bias)
        # self.c_attn_v = nn.Linear(config.n_embd, config.n_head * config.n_v, bias=config.bias)
        self.c_attn_q = LORI_FC(input_features=config.n_embd, output_features=config.n_embd, rank=config.n_q, diag_blocksize=config.n_fc_diagblock, bias=config.bias)
        self.c_attn_k = LORI_FC(input_features=config.n_embd, output_features=config.n_embd, rank=config.n_k, diag_blocksize=config.n_fc_diagblock, bias=config.bias)
        self.c_attn_v = LORI_FC(input_features=config.n_embd, output_features=config.n_embd, rank=config.n_v, diag_blocksize=config.n_fc_diagblock, bias=config.bias)
        # self.c_proj = nn.Linear(config.n_head*config.n_v, config.n_embd, bias=config.bias)
        self.c_proj = LORI_FC(input_features=config.n_embd, output_features=config.n_embd, rank=config.n_v, diag_blocksize=config.n_fc_diagblock, bias=config.bias)
        self.n_head = config.n_head

        # regularization
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.dropout = config.dropout
        self.n_q = config.n_q
        self.n_k = config.n_k
        self.n_v = config.n_v

        # flash attention make GPU go brrrrr but support is only in PyTorch >= 2.0
        self.flash = hasattr(torch.nn.functional, 'scaled_dot_product_attention')
        if not self.flash:
            print("WARNING: using slow attention. Flash Attention requires PyTorch >= 2.0")
            # causal mask to ensure that attention is only applied to the left in the input sequence
            self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size))
                                        .view(1, 1, config.block_size, config.block_size))

    def forward(self, x):
        B, T, C = x.size() # batch size, sequence length, embedding dimensionality (n_embd)
        nc0 = torch.isnan(x).any()
        q = self.c_attn_q(x)
        nc1 = torch.isnan(q).any()
        k = self.c_attn_k(x)
        nc2 = torch.isnan(k).any()
        v = self.c_attn_v(x)
        nc3 = torch.isnan(v).any()

        head_size  = C // self.n_head

        # k = k.view(B, T, self.n_head, self.n_k).transpose(1, 2) # (B, nh, T, hs)
        # q = q.view(B, T, self.n_head, self.n_q).transpose(1, 2) # (B, nh, T, hs)
        # k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        k = k.view(B, T, self.n_head, head_size).transpose(1, 2) # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, head_size).transpose(1, 2) # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, head_size).transpose(1, 2) # (B, nh, T, hs)

        # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        if self.flash:
            # efficient attention using Flash Attention CUDA kernels
            y = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=None, dropout_p=self.dropout if self.training else 0, is_causal=True)
        else:
            # manual implementation of attention
            att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
            att = att.masked_fill(self.bias[:,:,:T,:T] == 0, float('-inf'))
            att = F.softmax(att, dim=-1)
            att = self.attn_dropout(att)
            y = att @ v # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        nc4 = torch.isnan(y).any()
        y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side
        nc5 = torch.isnan(y).any()

        # output projection
        y = self.resid_dropout(self.c_proj(y))
        nc6 = torch.isnan(y).any()
        if (not nc0) and nc6:
            print('nc0:{0} nc1:{1} nc2:{2} nc3:{3} nc4:{4} nc5:{5} nc6:{6}'.format(nc0, nc1, nc2, nc3, nc4, nc5,nc6))

        return y

class LORIBlock(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.ln_1 = LayerNorm(config.n_embd, bias=config.bias)
        self.attn = LORICausalSelfAttention(config)
        self.ln_2 = LayerNorm(config.n_embd, bias=config.bias)
        self.mlp = LORIMLP(config)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class LORI_FC(nn.Module):
    def __init__(self, input_features, output_features, rank, diag_blocksize, bias):
        super().__init__()
        assert input_features % diag_blocksize == 0
        assert output_features % diag_blocksize == 0
        self.diag_blocksize = diag_blocksize
        self.input_features = input_features # self.left rows
        self.output_features = output_features # self.right cols
        if input_features < output_features:
            n_diag_blocks = input_features // diag_blocksize
        else:
            n_diag_blocks = output_features // diag_blocksize
        if bias:
            self.bias = nn.Parameter(torch.empty(size=(output_features,)))
        else:
            self.bias = None
        self.diag_params = nn.Parameter(torch.empty(size=(n_diag_blocks, 
                                                          input_features // n_diag_blocks,
                                                          output_features // n_diag_blocks )))
        self.left = nn.Linear(input_features, rank, bias = False)
        self.right = nn.Linear(rank, output_features, bias = bias)

    def forward(self, x):
        x1 = x @ torch.block_diag(*self.diag_params)
        x2 = self.right(self.left(x))
        if self.bias is not None:
            return x1 + x2 + self.bias
        else:
            return x1 + x2


# class LORI_FC1(nn.Module):
#     def __init__(self, config):
#         super().__init__()
#         # Number of blocks in the block-diagonal matrix
#         nblocks = config.n_embd // config.n_fc_diagblock
#         self.diag_params = nn.Parameter(torch.empty(
#             size=(nblocks, config.n_fc_diagblock, 4*config.n_fc_diagblock)))
#         self.left_ = nn.Linear(config.n_embd, config.n_fc_bottleneck)
#         self.right_ = nn.Linear(config.n_fc_bottleneck, 4*config.n_embd)
        
#     def forward(self, x):
#         x1 = x @ torch.block_diag(*self.diag_params)
#         x2 = self.right_(self.left_(x))
#         # print('FC1 x:{0} diag_params:{1} x1:{2} x2:{3}'.format(x.shape, self.diag_params.shape, x1.shape, x2.shape))
#         # print('FC1 left:{0} right:{1}'.format(self.left_.weight.shape, self.right_.weight.shape))
#         x = x1+x2
#         return x
    
# class LORI_FC2(nn.Module):
#     def __init__(self, config):
#         super().__init__()
#         # Number of blocks in the block-diagonal matrix
#         nblocks = config.n_embd // config.n_fc_diagblock

#         if 4*config.n_fc_bottleneck > config.n_embd:
#             bottleneck = config.n_embd
#         else:
#             bottleneck = 4*config.n_fc_bottleneck

#         self.diag_params_c_proj = nn.Parameter(torch.empty(
#             size=(nblocks, 4*config.n_fc_diagblock, config.n_fc_diagblock)))
#         self.left_ = nn.Linear(4*config.n_embd, bottleneck)
#         self.c_proj = nn.Linear(bottleneck, config.n_embd)
        
#     def forward(self, x):
#         block_diag_mat = torch.block_diag(*self.diag_params_c_proj)
#         x1 = x @ block_diag_mat
#         x2 = self.c_proj(self.left_(x))
#         # print('FC2 x:{0} diag_params:{1} x1:{2} x2:{3} block diag:{4}'.format(x.shape, self.diag_params_c_proj.shape, x1.shape, x2.shape, block_diag_mat.shape))
#         # print('FC2 left:{0} c_proj:{1}'.format(self.left_.weight.shape, self.c_proj.weight.shape))
#         x = x2 + x1
#         return x

class LORIMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.fc1 = LORI_FC(input_features=config.n_embd, 
                           output_features=4*config.n_embd, 
                           rank=config.n_fc_bottleneck, 
                           diag_blocksize=config.n_fc_diagblock, 
                           bias=config.bias)
        self.gelu = nn.GELU()
        self.fc2 = LORI_FC(input_features=4*config.n_embd, 
                           output_features=config.n_embd, 
                           rank=4*config.n_fc_bottleneck, 
                           diag_blocksize=4*config.n_fc_diagblock, 
                           bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        x = self.fc1(x)
        x = self.gelu(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x

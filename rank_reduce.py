
import torch
import numpy as np
from model import GPT, GPTConfig
import os

device = 'cuda'

# # Karpathy's wierd configurator thing requires network
# # config to be global variables.
# block_size = 1024
# # model
# n_layer = 12
# n_head = 12
# n_embd = 768
# dropout = 0.2 # for pretraining 0 is good, for finetuning try 0.1+
# bias = True # do we use bias inside LayerNorm and Linear layers?
lori=True
# Only for LORI:
n_q = 4
n_k = 4
n_v = 8
n_fc_bottleneck = 32
n_fc_diagblock = 16
compile = True # use PyTorch 2.0 to compile the model to be faster
# -----------------------------------------------------------------------------
config_keys = [k for k,v in globals().items() if not k.startswith('_') and isinstance(v, (int, float, bool, str))]
exec(open('configurator.py').read()) # overrides from command line or config file
config = {k: globals()[k] for k in config_keys} # will be useful for logging

def load_source_model(source_dir):
    ckpt_path = os.path.join(source_dir, 'ckpt.pt')
    checkpoint = torch.load(ckpt_path, map_location=device)
    checkpoint_model_args = checkpoint['model_args']
    # # force these config attributes to be equal otherwise we can't even resume training
    # # the rest of the attributes (e.g. dropout) can stay as desired from command line
    # assert checkpoint_model_args['n_layer'] == n_layer
    # assert checkpoint_model_args['n_head'] == n_head
    # assert checkpoint_model_args['n_embd'] == n_embd
    # assert checkpoint_model_args['block_size'] == block_size
    # assert checkpoint_model_args['bias'] == bias
    # assert checkpoint_model_args['vocab_size'] == vocab_size

    # create the model
    gptconf = GPTConfig(**checkpoint_model_args)
    model = GPT(gptconf)
    state_dict = checkpoint['model']
    # fix the keys of the state dictionary :(
    # honestly no idea how checkpoints sometimes get this prefix, have to debug more
    unwanted_prefix = '_orig_mod.'
    for k,v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
    model.load_state_dict(state_dict)
    return model, checkpoint_model_args

def augment_with_bias(wts, biases):
    biases = torch.unsqueeze(biases, dim=1)
    augmented_wts = torch.cat( (wts, biases), dim=1)
    # Number of cols should now be one greater than before cat
    assert augmented_wts.shape[0]+1 == augmented_wts.shape[1]
    return augmented_wts

def rank_reduce_attn_value(v_attn_weight, v_attn_bias, c_proj_weight, c_proj_bias, config):
    if config.bias:
        assert False
    else:
        prod = torch.transpose(v_attn_weight, 0,1) @ torch.transpose(c_proj_weight, 0,1)
        v_U, v_S, v_V = torch.linalg.svd(prod, full_matrices = True)
        v_attn_rr_left = v_U[:,:config.n_v] @ torch.diag(v_S[:config.n_v]) 
        v_attn_rr_right = v_V
        return v_attn_rr_left, v_attn_rr_right

#  attn_params = rank_reduce_attention(c_attn_bias, c_attn_weight, layer_prefix)
# c_attn_weight is weights of shape (3*n_embd x n_embd) = (output x input) dim
# c_attn_bias is weights of (3*n_embd) = output dim
def rank_reduce_attention(c_attn_bias, c_attn_weight, c_proj_weight, c_proj_bias, layer_prefix, config):
    assert config.n_q < target_config.n_embd
    assert config.n_k < target_config.n_embd
    assert config.n_v < target_config.n_embd

    assert len(c_attn_weight.shape) == 2
    assert c_attn_weight.shape[0] == 3*config.n_embd
    assert c_attn_weight.shape[1] == config.n_embd
    if target_config.bias:
        assert len(c_attn_bias.shape) == 1
        assert c_attn_bias.shape[0] == 3*config.n_embd
    else:
        assert c_attn_bias is None

    # Split weights into those of q, k, v:
    q_attn, k_attn, v_attn = c_attn_weight.split(config.n_embd, dim=0)

    # add bias as last column in matrix (see torch nn.Linear).
    if config.bias:
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

    q_U, q_S, q_V = torch.linalg.svd(q_attn_aug, full_matrices = True)
    k_U, k_S, k_V = torch.linalg.svd(k_attn_aug, full_matrices = True)
    # v_U, v_S, v_V = torch.linalg.svd(v_attn_aug, full_matrices = True)

    q_attn_rr_dot = q_U[:,:config.n_q] @ torch.diag(q_S[:config.n_q]) @ q_V[:config.n_q,:] @ torch.transpose(k_V[:config.n_k,:], 0,1)
    q_attn_rr_dot_bias = None
    k_attn_rr_dot = k_U[:,:config.n_k] @ torch.diag(k_S[:config.n_k]) 
    k_attn_rr_dot_bias = None

    v_attn_rr_left, v_attn_rr_right = rank_reduce_attn_value(v_attn, v_attn_bias,  c_proj_weight, c_proj_bias, config)

    new_attn_state_dict = {layer_prefix + 'attn.c_attn_q.weight': q_attn_rr_dot,
     layer_prefix + 'attn.c_attn_k.weight': k_attn_rr_dot,
     layer_prefix + 'attn.c_attn_v.weight': v_attn_rr_left,
     layer_prefix + 'attn.c_proj.weight': v_attn_rr_right}

    return new_attn_state_dict

# transformer.h.0.mlp.fc1.diag_params: torch.Size([96, 4, 16])
# transformer.h.0.mlp.fc1.left_.weight: torch.Size([64, 384])
# transformer.h.0.mlp.fc1.left_.bias: torch.Size([64])
# transformer.h.0.mlp.fc1.right_.weight: torch.Size([1536, 64])
# transformer.h.0.mlp.fc1.right_.bias: torch.Size([1536])
# transformer.h.0.mlp.fc2.diag_params_c_proj: torch.Size([96, 16, 4])
# transformer.h.0.mlp.fc2.left_.weight: torch.Size([256, 1536])
# transformer.h.0.mlp.fc2.left_.bias: torch.Size([256])
# transformer.h.0.mlp.fc2.c_proj.weight: torch.Size([384, 256])
# transformer.h.0.mlp.fc2.c_proj.bias: torch.Size([384])


def rank_reduce_mlp(mlp_fc_wt, mlp_fc_bias, mlp_c_proj_wt, mlp_c_proj_bias, target_config, prefix):
    # Just deal with the case of no bias for now.
    assert mlp_fc_bias is None
    assert mlp_c_proj_bias is None

    # FC 1.
    # --------------------
    # Set block diag params to match their counterparts in the source mlp_fc_wt.
    block_size = target_config.n_fc_diagblock
    assert target_config.n_embd % block_size == 0
    num_diag_blocks = target_config.n_embd // block_size
    fc1_diag = torch.zeros(size=(num_diag_blocks, block_size, 4*block_size))
    source_col = 0
    source_row = 0

    # print('fc1_diag size:{0}'.format(fc1_diag.shape))
    # print('block size:{0}'.format(block_size))
    mlp_fc_wt_transpose = torch.transpose(mlp_fc_wt,0,1)
    for i in range(num_diag_blocks):
        # print('source_col:{0}, source_col+4*block_size:{1}, mlp_fc_wt' shape:{2}'.format(source_col, source_col+4*block_size, mlp_fc_wt_transpose.shape))
        fc1_diag[i,:,:] = mlp_fc_wt_transpose[source_row:source_row+block_size,
                                    source_col:source_col+4*block_size]
        source_row += block_size
        source_col += 4*block_size

    # Find the residual between block diag and the fc_wt
    fc1_residual = mlp_fc_wt_transpose - torch.block_diag(*fc1_diag)

    # Find low rank approximation to the residual.
    U, S, V = torch.linalg.svd(fc1_residual, full_matrices = True)
    fc1_left_wt = torch.transpose( U[:,:target_config.n_fc_bottleneck] @ torch.diag(S[:target_config.n_fc_bottleneck]), 0,1)
    fc1_right_wt = torch.transpose(V[:target_config.n_fc_bottleneck,:], 0,1)

    # FC 2.
    # --------------------
    # Set block diag params to match their counterparts in the source mlp_c_proj_wt.
    fc2_diag = torch.zeros(size=(num_diag_blocks, 4*block_size, block_size))
    source_col = 0
    source_row = 0
    mlp_c_proj_wt_transpose = torch.transpose(mlp_c_proj_wt,0,1)
    for i in range(num_diag_blocks):
        fc2_diag[i,:,:] = mlp_c_proj_wt_transpose[source_row:source_row+4*block_size,
                                        source_col:source_col+block_size]
        source_row += 4*block_size
        source_col += block_size

    # Find the residual between block diag and mlp_c_proj_wt
    fc2_residual = mlp_c_proj_wt_transpose - torch.block_diag(*fc2_diag)

    # Low rank approximation to the residual
    U, S, V = torch.linalg.svd(fc2_residual, full_matrices=True)
    fc2_left_wt = torch.transpose( U[:,:4*target_config.n_fc_bottleneck] @ torch.diag(S[:4*target_config.n_fc_bottleneck]), 0,1)
    fc2_right_wt = torch.transpose(V[:4*target_config.n_fc_bottleneck,:], 0,1)

    mlp_params = {prefix + 'mlp.fc1.diag_params': fc1_diag,
                  prefix + 'mlp.fc1.left_.weight': fc1_left_wt,
                  prefix + 'mlp.fc1.right_.weight': fc1_right_wt,
                  prefix + 'mlp.fc2.diag_params_c_proj': fc2_diag,
                  prefix + 'mlp.fc2.left_.weight': fc2_left_wt,
                  prefix + 'mlp.fc2.c_proj.weight': fc2_right_wt }
    return mlp_params

    # v_attn_rr = v_U[:,:config.n_v] @ torch.diag( v_S[:config.n_v]) @ v_V[:config.n_v,:]
    # if config.bias:
    #     v_attn_rr_wts = v_attn_rr[:,:(config.n_v+1)]
    #     v_attn_rr_bias = v_attn_rr[:, (config.n_v+1)]
    # else:
    #     v_attn_rr_wts = v_attn_rr
    #     v_attn_rr_bias = None


# LORI shakespeare:
# number of parameters: 4.81M
# transformer.wte.weight: torch.Size([65, 384])
# transformer.wpe.weight: torch.Size([256, 384])
# transformer.h.0.ln_1.weight: torch.Size([384])
# transformer.h.0.ln_1.bias: torch.Size([384])
# transformer.h.0.attn.c_attn_q.weight: torch.Size([24, 384])
# transformer.h.0.attn.c_attn_q.bias: torch.Size([24])
# transformer.h.0.attn.c_attn_k.weight: torch.Size([24, 384])
# transformer.h.0.attn.c_attn_k.bias: torch.Size([24])
# transformer.h.0.attn.c_attn_v.weight: torch.Size([192, 384])
# transformer.h.0.attn.c_attn_v.bias: torch.Size([192])
# transformer.h.0.attn.c_proj.weight: torch.Size([384, 192])
# transformer.h.0.attn.c_proj.bias: torch.Size([384])
# transformer.h.0.ln_2.weight: torch.Size([384])
# transformer.h.0.ln_2.bias: torch.Size([384])
# transformer.h.0.mlp.fc1.diag_params: torch.Size([96, 4, 16])
# transformer.h.0.mlp.fc1.left_.weight: torch.Size([64, 384])
# transformer.h.0.mlp.fc1.left_.bias: torch.Size([64])
# transformer.h.0.mlp.fc1.right_.weight: torch.Size([1536, 64])
# transformer.h.0.mlp.fc1.right_.bias: torch.Size([1536])
# transformer.h.0.mlp.fc2.diag_params_c_proj: torch.Size([96, 16, 4])
# transformer.h.0.mlp.fc2.left_.weight: torch.Size([256, 1536])
# transformer.h.0.mlp.fc2.left_.bias: torch.Size([256])
# transformer.h.0.mlp.fc2.c_proj.weight: torch.Size([384, 256])
# transformer.h.0.mlp.fc2.c_proj.bias: torch.Size([384])
# transformer.h.1.ln_1.weight: torch.Size([384])
# transformer.h.1.ln_1.bias: torch.Size([384])


# def create_target_model():
#     model_args = dict(n_layer=n_layer, n_head=n_head, n_embd=n_embd, block_size=block_size,
#                     bias=bias, vocab_size=None, dropout=dropout, lori=lori, n_q = n_q, n_k = n_k, n_v = n_v, n_fc_bottleneck = n_fc_bottleneck, n_fc_diagblock = n_fc_diagblock) # start with model_args from command line
#     # determine the vocab size we'll use for from-scratch training
#     if meta_vocab_size is None:
#         print("defaulting to vocab_size of GPT-2 to 50304 (50257 rounded up for efficiency)")
#     model_args['vocab_size'] = meta_vocab_size if meta_vocab_size is not None else 50304
#     gptconf = GPTConfig(**model_args)
#     model = GPT(gptconf)    

# source_model = load_source_model('out_gpt_openai')
source_model, source_args = load_source_model('out-shakespeare-char-lori')

sd = source_model.state_dict()
for key in sd.keys():
    print('{0}: {1}'.format(key, sd[key].shape))

target_model_args = dict(source_args)
source_model.eval()

lori=True
# Only for LORI:
n_q = 6
n_k = 6
n_v = 6
n_fc_bottleneck = 32
n_fc_diagblock = 4

print('bias:{0}'.format(source_args['bias']))

# Set the LORI-related parameters, keeping all the other
# parameters the same as that of the source model.
target_model_args.update({'lori': True, 'n_q': n_q, 'n_k': n_k, 'n_v': n_v, \
                          'n_fc_bottleneck': n_fc_bottleneck, \
                          'n_fc_diagblock': n_fc_diagblock})
target_config = GPTConfig(**target_model_args)
print('target_model_args')
print(target_model_args)
print(target_config)
print('GPTConfig:')
print(target_config)
target_model = GPT(target_config)
target_model.eval()

td = target_model.state_dict()
for key in td.keys():
    print('{0}: {1}'.format(key, td[key].shape))

def maybe_assign(pname, td, sd, use_bias):
    if use_bias:
        td[pname] = sd[pname]

# Convert or copy each param in source state dict (sd) to 
# target state dict (td)
td['transformer.wte.weight'] = sd['transformer.wte.weight']
td['transformer.wpe.weight'] = sd['transformer.wpe.weight']
td['lm_head.weight'] = sd['lm_head.weight']
for i in range(0,target_config.n_layer):
    # Handle each layer in the net.
    sd_keys = sd.keys()
    layer_prefix = 'transformer.h.{0}.'.format(i)
    print('processing layer {0}'.format(layer_prefix))
    layer_i_params = [pn for pn in sd_keys if pn.startswith(layer_prefix)]
    biases = target_model_args['bias']

    # layer norm 1
    ln1_wt = layer_prefix + 'ln_1.weight'
    ln1_bias = layer_prefix + 'ln_1.bias'
    td[ln1_wt] = sd[ln1_wt]
    if target_model_args['bias']:
        td[ln1_bias] = sd[ln1_bias]

    # layer norm 2
    ln2_wt = layer_prefix + 'ln_2.weight'
    ln2_bias = layer_prefix + 'ln_2.bias'
    td[ln2_wt] = sd[ln2_wt]
    if biases:
        td[ln2_bias] = sd[ln2_bias]

    # Attention q,k,v
    c_attn_wt_name = 'attn.c_attn.weight'
    c_attn_bias_name = 'attn.c_attn.bias'
    if biases:
        assert layer_prefix + c_attn_bias_name in layer_i_params
        c_attn_bias = sd[layer_prefix + c_attn_bias_name]
    else:
        c_attn_bias = None

    print('layer prefix + c_attn_wt_name:{0}'.format(layer_prefix + c_attn_wt_name))
    print('layer i params:{0}'.format(layer_i_params))
    assert layer_prefix + c_attn_wt_name in layer_i_params
    c_attn_weight = sd[layer_prefix + c_attn_wt_name]

    c_proj_wt_name = 'attn.c_proj.weight'
    c_proj_bias_name = 'attn.c_proj.bias'
    c_proj_weight = sd[layer_prefix + c_proj_wt_name]
    if biases:
        c_proj_bias = sd[layer_prefix + c_proj_bias_name]
    else:
        c_proj_bias = None

    attn_params = rank_reduce_attention(c_attn_bias, c_attn_weight, c_proj_weight, c_proj_bias, layer_prefix, target_config)
    td.update(attn_params)

    # MLP params
    mlp_fc_wt_name = 'mlp.c_fc.weight'
    mlp_fc_bias_name = 'mlp.c_fc.bias'
    mlp_c_proj_wt_name = 'mlp.c_proj.weight'
    mlp_c_proj_bias_name = 'mlp.c_proj.bias'

    mlp_fc_wt = sd[layer_prefix + mlp_fc_wt_name]
    mlp_c_proj_wt = sd[layer_prefix + mlp_c_proj_wt_name]
    if biases:
        mlp_fc_bias = sd[layer_prefix + mlp_fc_bias_name]
        mlp_c_proj_bias = sd[layer_prefix + mlp_c_proj_bias_name]
    else:
        mlp_fc_bias = None
        mlp_c_proj_bias = None

    mlp_params = rank_reduce_mlp(mlp_fc_wt, mlp_fc_bias, mlp_c_proj_wt, mlp_c_proj_bias, target_config, layer_prefix)
    td.update(mlp_params)

out_dir = 'out-rr'
print(f"saving model to {out_dir}")
print('target model sd keys:')
default_keys = target_model.state_dict().keys()
for key in default_keys:
    print(key)


for key in td.keys():
    if key in default_keys:
        print('found {0}, default shape:{1}, new shape:{2}'.format(key, target_model.state_dict()[key].shape,
                                                                   td[key].shape))
    else:
        print('{0} not found'.format(key))
target_model.load_state_dict(td)
torch.save(target_model, os.path.join(out_dir, 'ckpt.pt'))








# Items in the state dict that require conversion in rank-reduction process.
# Other items are just copied over without changes.
# transformer.h.0.
# attn.c_attn.weight: torch.Size([1152, 384])
# attn.c_attn.bias: torch.Size([1152])
# attn.c_proj.weight: torch.Size([384, 384])
# attn.c_proj.bias: torch.Size([384])
# mlp.c_fc.weight: torch.Size([1536, 384])
# mlp.c_fc.bias: torch.Size([1536])
# mlp.c_proj.weight: torch.Size([384, 1536])
# mlp.c_proj.bias: torch.Size([384])

# conversion_items = { ('attn.c_attn.weight', ),
# transformer.h.0.attn.c_proj.weight: torch.Size([384, 384])
# transformer.h.0.ln_2.weight: torch.Size([384])
# transformer.h.0.mlp.c_fc.weight: torch.Size([1536, 384])
# transformer.h.0.mlp.c_proj.weight: torch.Size([384, 1536]))}

# target_model

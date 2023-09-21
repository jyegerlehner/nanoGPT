import torch
import numpy as np
from model import GPT, GPTConfig
import os
from LORI import set_state, reduce_rank_causal_attn, LORICausalSelfAttention, make_LORI_FC_from_matrix, make_LORI_FC_from_module, LORIMLP, reduce_rank_MLP, reduce_rank_Block



device = 'cuda'

# Karpathy's wierd configurator thing requires network
# config to be global variables.
# Let most of those come from the config of the full rank
# model that we are converting to low rank.
# block_size = 1024
# # model
# n_layer = 12
# n_head = 12
# n_embd = 768
# dropout = 0.2 # for pretraining 0 is good, for finetuning try 0.1+
# bias = True # do we use bias inside LayerNorm and Linear layers?
# # Only for LORI:
lori=True
n_q = 32
n_k = 32
n_v = 64
# n_q = 384
# n_k = 384
# n_v = 384
n_fc_bottleneck = 64
n_fc_diagblock = 4
compile = True # use PyTorch 2.0 to compile the model to be faster

# -----------------------------------------------------------------------------
config_keys = [k for k,v in globals().items() if not k.startswith('_') and isinstance(v, (int, float, bool, str))]
exec(open('configurator.py').read()) # overrides from command line or config file
config = {k: globals()[k] for k in config_keys} # will be useful for logging

def load_source_model(source_dir):
    ckpt_path = os.path.join(source_dir, 'ckpt.pt')
    checkpoint = torch.load(ckpt_path, map_location=device)
    checkpoint_model_args = checkpoint['model_args']
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
    source_config = checkpoint['config']
    print('-------------------------------')
    print('source_config:')
    print(source_config)
    return model, checkpoint_model_args, source_config

def augment_with_bias(wts, biases):
    biases = torch.unsqueeze(biases, dim=1)
    augmented_wts = torch.cat( (wts, biases), dim=1)
    # Number of cols should now be one greater than before cat
    assert augmented_wts.shape[0]+1 == augmented_wts.shape[1]
    return augmented_wts

# source_model = load_source_model('out_gpt_openai')
source_model, source_args, source_config = load_source_model('out-shakespeare-char')

print('source params:')
sd = source_model.state_dict()
for key in sd.keys():
    print('{0}: {1}'.format(key, sd[key].shape))
target_model_args = dict(source_args)
source_model.eval()
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

print('Target model state dict keys:')
print('------------------------')
td = target_model.state_dict()
for key in td.keys():
    print('{0}: {1}'.format(key, td[key].shape))

def maybe_assign(pname, td, sd, use_bias):
    if use_bias:
        td[pname] = sd[pname]

# Convert or copy each param in source state dict (sd) to 
# target state dict (td)
set_state(target_model.transformer.wte, source_model.transformer.wte)
set_state(target_model.transformer.wpe, source_model.transformer.wpe)
set_state(target_model.lm_head, source_model.lm_head)

# td['transformer.wte.weight'] = sd['transformer.wte.weight']
# td['transformer.wpe.weight'] = sd['transformer.wpe.weight']
# td['lm_head.weight'] = sd['lm_head.weight']
biases = target_model_args['bias']
for i in range(0,target_config.n_layer):
    source_block = source_model.transformer.h[i]
    target_block = target_model.transformer.h[i]
    reduce_rank_Block(source_block, target_block, target_config)


out_dir = 'out-rr'
print(f"saving model to {out_dir}")
# print('target model sd keys:')
# default_keys = target_model.state_dict().keys()
# for key in default_keys:
#     print(key)


# for key in td.keys():
#     if key in default_keys:
#         print('found {0}, default shape:{1}, new shape:{2}'.format(key, target_model.state_dict()[key].shape,
#                                                                    td[key].shape))
#     else:
#         print('{0} not found'.format(key))
target_model.load_state_dict(td)
config = dict( source_config, **config)
print('config:')
print(config)
checkpoint = {'model': target_model, 'config': config}
torch.save(checkpoint, os.path.join(out_dir, 'ckpt.pt'))








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

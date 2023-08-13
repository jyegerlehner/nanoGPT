out_dir = 'out_gpt_openai'

# evaluate the base gpt2
# n_layer=12, n_head=12, n_embd=768
# 124M parameters
eval_only = False
wandb_log = False
init_from = 'resume'
learning_rate = 3e-5


batch_size = 12
block_size = 1024
gradient_accumulation_steps = 5 * 8

# this makes total number of tokens be 300B
max_iters = 600000
lr_decay_iters = 600000

# eval stuff
eval_interval = 1000
eval_iters = 200
log_interval = 10

# weight decay
weight_decay = 1e-1

# train a miniature character-level shakespeare model
# good for debugging and playing on macbooks and such

out_dir = 'out-shakespeare-char-lori-big'
eval_interval = 250 # keep frequent because we'll overfit
eval_iters = 200
log_interval = 10 # don't print too too often

# we expect to overfit on this small dataset, so only save when val improves
always_save_checkpoint = False

wandb_log = False # override via command line if you like
wandb_project = 'shakespeare-char-lori-big'
wandb_run_name = 'mini-gpt-lori-big'

dataset = 'shakespeare_char'
gradient_accumulation_steps = 1
batch_size = 64
block_size = 256 # context of up to 256 previous characters

# baby GPT model :)
n_layer = 16
n_head = 6
n_embd = 384
dropout = 0.2
lori = True # low-rank implementation

learning_rate = 4e-4 # with baby networks can afford to go a bit higher
max_iters = 20000
lr_decay_iters = 15000 # make equal to max_iters usually
min_lr = 4e-5 # learning_rate / 10 usually
beta2 = 0.99 # make a bit smaller because number of params per iter is small

warmup_iters = 100 # not super necessary potentially
        
# on macbook also add
# device = 'cpu'  # run on cpu only
# compile = False # do not torch compile the model
compile = True
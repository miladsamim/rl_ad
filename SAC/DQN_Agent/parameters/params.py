import torch 
from parameters import dotdict
model_params = dotdict()
model_params.in_channels = 3 
model_params.h_size = 128 # dimensionality of latent vectors
model_params.dropout_cnn = 0.3
model_params.img_shape = (96,96)
model_params.act_dim = 5
model_params.n_frames = 16
model_params.buffer_sz = 15_000
model_params.n_head = 8 # number of attention heads
model_params.n_encs_f = 2 # number of encoder layers in the fusion transformer
model_params.n_decs_f = 1 # number of decoder layers in the fusion transformer
model_params.n_encs_t = 2 # number of encoder layers in the temporal transformer
model_params.n_decs_t = 1 # number of decoder layers in the temporal transformer
model_params.n_decs = 2 # number of decoder layers in decoder only arch (gpt)
model_params.norm_first = False # whether to normalize before applying attention layers
model_params.t_dropout = 0.1 # dropout rate in the transformers
model_params.c_dropout = 0.3 # dropout rate in sensor cnns
model_params.accel = True # whether to reshape all temporal states to single tensor, faster but requires enough gpu memory
model_params.device = 'cuda' if torch.cuda.is_available() else 'cpu'
model_params.residual = False
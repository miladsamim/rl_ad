import torch 
import torch.nn as nn
import torch.nn.functional as F
# from torchvision.transforms import Grayscale

from parameters.sensor_net import SensorModel
from parameters.positional_encoding import PositionalEncoding
from parameters.architectures_torch import Nature_Paper_Conv_Dropout_Torch2, Nature_Paper_Conv_Dropout_Torch

class DriveDQN(nn.Module):
    """Original Proposed Transformer Architecture"""
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.sensor_net = SensorModel(args.in_channels, args.h_size,
                                      p=args.c_dropout, image_shape=args.img_shape)
        self.fusion_net = nn.Transformer(d_model=args.h_size, nhead=args.n_head, num_encoder_layers=args.n_encs_f,
                                         num_decoder_layers=args.n_decs_f, dim_feedforward=4*args.h_size, 
                                         dropout=args.t_dropout, norm_first=args.device, device=args.device)
        self.temporal_net = nn.Transformer(d_model=args.h_size, nhead=args.n_head, num_encoder_layers=args.n_encs_t,
                                         num_decoder_layers=args.n_decs_t, dim_feedforward=4*args.h_size, 
                                         dropout=args.t_dropout, norm_first=args.device, device=args.device)
        self.positional_encoder = PositionalEncoding(d_model=args.h_size, max_len=64) # max number of time steps
        self.action_emb = nn.Embedding(args.act_dim, args.h_size)
        self.action_emb.weight.requires_grad = False
        self.act_dec_idx = torch.arange(args.act_dim).to(device=args.device) 
        self.out = nn.Linear(args.h_size, 1)

    def forward(self, X_img, X_sensor, X_act):
        n_frames, b_size = X_img.shape[0], X_img.shape[1]
        hidden_states = []
        dec_in = torch.ones(1, b_size, self.args.h_size, requires_grad=False, device=self.args.device)
        for i in range(n_frames):
            X_state_h = self.sensor_net(X_img[i], X_sensor[:,i])
            X_state_h.append(self.action_emb(X_act[i]))
            X_state_h = torch.stack(X_state_h, axis=0)
            X_state_h = self.fusion_net(X_state_h, dec_in)
            hidden_states.append(X_state_h)

        hidden_states = torch.concat(hidden_states) # seqLen X batchSize X h_size 
        hidden_states = self.positional_encoder(hidden_states)
        # dec_in = torch.rand(self.n_act_nets, b_size, self.args.h_size, requires_grad=False, device=self.args.device)
        dec_in = self.action_emb(self.act_dec_idx.repeat(b_size, 1)).transpose(0,1)
        hidden_state = self.temporal_net(hidden_states, dec_in) # seq_len X batchSize X h_size 
        hidden_state = hidden_state.transpose(0, 1) # seq,batch,... -> batch,seq,...
        return self.out(hidden_state).squeeze(2)


class DriveDQN_simple_fusion(nn.Module):
    """-Changing sensor fusion to simple concatenation"""
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.sensor_net = SensorModel(args.in_channels, args.h_size,
                                      p=args.c_dropout, image_shape=args.img_shape)
        self.temporal_net = nn.Transformer(d_model=args.h_size, nhead=args.n_head, num_encoder_layers=args.n_encs_t,
                                         num_decoder_layers=args.n_decs_t, dim_feedforward=4*args.h_size, 
                                         dropout=args.t_dropout, norm_first=args.device, device=args.device)
        self.positional_encoder = PositionalEncoding(d_model=args.h_size, max_len=64) # max number of time steps
        self.action_emb = nn.Embedding(args.act_dim, args.h_size)
        self.action_emb.weight.requires_grad = False
        self.act_dec_idx = torch.arange(args.act_dim).to(device=args.device) 
        self.out = nn.Linear(args.h_size, 1)
        # cnn1d out will be 248 from 128*8=1024 length 
        self.cnn_1d_sensor = nn.Conv1d(8, 8, kernel_size=8, stride=4, groups=8)
        # stacking im and cnn1d-out will be 128+248=376
        self.merge_sensors = nn.Linear(376, args.h_size)

    def forward(self, X_img, X_sensor, X_act):
        n_frames, b_size = X_img.shape[0], X_img.shape[1]
        hidden_states = []
        dec_in = torch.ones(1, b_size, self.args.h_size, requires_grad=False, device=self.args.device)
        for i in range(n_frames):
            X_state_h = self.sensor_net(X_img[i], X_sensor[:,i])
            X_state_h.append(self.action_emb(X_act[i]))
            X_sensor_h = torch.stack(X_state_h[1:], axis=1)
            X_sensor_h = self.cnn_1d_sensor(X_sensor_h).view(b_size, -1)
            X_state_h = torch.concat([X_state_h[0],X_sensor_h], axis=1)
            X_state_h = self.merge_sensors(X_state_h)
            hidden_states.append(X_state_h)

        hidden_states = torch.stack(hidden_states, axis=0) # seqLen X batchSize X h_size 
        hidden_states = self.positional_encoder(hidden_states)
        # dec_in = torch.rand(self.n_act_nets, b_size, self.args.h_size, requires_grad=False, device=self.args.device)
        dec_in = self.action_emb(self.act_dec_idx.repeat(b_size, 1)).transpose(0,1)
        hidden_state = self.temporal_net(hidden_states, dec_in) # seq_len X batchSize X h_size 
        hidden_state = hidden_state.transpose(0, 1) # seq,batch,... -> batch,seq,...
        return self.out(hidden_state).squeeze(2)

class DriveDQN_simple_fusion_learnable_act_embs(nn.Module):
    """-Changing sensor fusion to simple concatenation
       -Allow action embeddings to be learnable"""
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.sensor_net = SensorModel(args.in_channels, args.h_size,
                                      p=args.c_dropout, image_shape=args.img_shape)
        self.temporal_net = nn.Transformer(d_model=args.h_size, nhead=args.n_head, num_encoder_layers=args.n_encs_t,
                                         num_decoder_layers=args.n_decs_t, dim_feedforward=4*args.h_size, 
                                         dropout=args.t_dropout, norm_first=args.device, device=args.device)
        self.positional_encoder = PositionalEncoding(d_model=args.h_size, max_len=64) # max number of time steps
        self.action_emb = nn.Embedding(args.act_dim, args.h_size)
        self.act_dec_idx = torch.arange(args.act_dim).to(device=args.device) 
        self.out = nn.Linear(args.h_size, 1)
        # cnn1d out will be 248 from 128*8=1024 length 
        self.cnn_1d_sensor = nn.Conv1d(8, 8, kernel_size=8, stride=4, groups=8)
        # stacking im and cnn1d-out will be 128+248=376
        self.merge_sensors = nn.Linear(376, args.h_size)

    def forward(self, X_img, X_sensor, X_act):
        n_frames, b_size = X_img.shape[0], X_img.shape[1]
        hidden_states = []
        dec_in = torch.ones(1, b_size, self.args.h_size, requires_grad=False, device=self.args.device)
        for i in range(n_frames):
            X_state_h = self.sensor_net(X_img[i], X_sensor[:,i])
            X_state_h.append(self.action_emb(X_act[i]))
            X_sensor_h = torch.stack(X_state_h[1:], axis=1)
            X_sensor_h = self.cnn_1d_sensor(X_sensor_h).view(b_size, -1)
            X_state_h = torch.concat([X_state_h[0],X_sensor_h], axis=1)
            X_state_h = self.merge_sensors(X_state_h)
            hidden_states.append(X_state_h)

        hidden_states = torch.stack(hidden_states, axis=0) # seqLen X batchSize X h_size 
        hidden_states = self.positional_encoder(hidden_states)
        # dec_in = torch.rand(self.n_act_nets, b_size, self.args.h_size, requires_grad=False, device=self.args.device)
        dec_in = self.action_emb(self.act_dec_idx.repeat(b_size, 1)).transpose(0,1)
        hidden_state = self.temporal_net(hidden_states, dec_in) # seq_len X batchSize X h_size 
        hidden_state = hidden_state.transpose(0, 1) # seq,batch,... -> batch,seq,...
        return self.out(hidden_state).squeeze(2)

class DriveDQN_simple_fusion_single_act_dec(nn.Module):
    """-Changing sensor fusion to simple concatenation
       -Only decode to single action"""
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.sensor_d_size = sensor_d_size = 376
        self.sensor_net = SensorModel(args.in_channels, args.h_size,
                                      p=args.c_dropout, image_shape=args.img_shape)
        self.temporal_net = nn.Transformer(d_model=sensor_d_size, nhead=args.n_head, num_encoder_layers=args.n_encs_t,
                                         num_decoder_layers=args.n_decs_t, dim_feedforward=4*args.h_size, 
                                         dropout=args.t_dropout, norm_first=args.device, device=args.device)
        self.positional_encoder = PositionalEncoding(d_model=sensor_d_size, max_len=64) # max number of time steps
        self.action_emb = nn.Embedding(args.act_dim, args.h_size)
        self.action_emb.weight.requires_grad = False
        self.act_dec_idx = torch.arange(args.act_dim).to(device=args.device) 
        self.out = nn.Linear(self.sensor_d_size, args.act_dim)
        # cnn1d out will be 248 from 128*8=1024 length 
        self.cnn_1d_sensor = nn.Conv1d(8, 8, kernel_size=8, stride=4, groups=8)
        # stacking im and cnn1d-out will be 128+248=376
        self.merge_sensors = nn.Linear(sensor_d_size, args.h_size)

    def forward(self, X_img, X_sensor, X_act):
        n_frames, b_size = X_img.shape[0], X_img.shape[1]
        hidden_states = []
        dec_in = torch.ones(1, b_size, self.args.h_size, requires_grad=False, device=self.args.device)
        for i in range(n_frames):
            X_state_h = self.sensor_net(X_img[i], X_sensor[:,i])
            X_state_h.append(self.action_emb(X_act[i]))
            X_sensor_h = torch.stack(X_state_h[1:], axis=1)
            X_sensor_h = self.cnn_1d_sensor(X_sensor_h).view(b_size, -1)
            X_state_h = torch.concat([X_state_h[0],X_sensor_h], axis=1)
            hidden_states.append(X_state_h)

        hidden_states = torch.stack(hidden_states, axis=0) # seqLen X batchSize X h_size 
        hidden_states = self.positional_encoder(hidden_states)
        dec_in = torch.ones(1, b_size, self.sensor_d_size, requires_grad=False, device=self.args.device)
        # dec_in = self.action_emb(self.act_dec_idx.repeat(b_size, 1)).transpose(0,1)
        hidden_state = self.temporal_net(hidden_states, dec_in) # seq_len X batchSize X h_size 
        hidden_state = hidden_state.transpose(0, 1) # seq,batch,... -> batch,seq,...
        return self.out(F.relu(hidden_state)).squeeze(1)


class DriveDQN_simple_fusion_lstm(nn.Module):
    """-Changing sensor fusion to simple concatenation
       -Use GRU rnn to decode single action state for q values"""
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.sensor_net = SensorModel(args.in_channels, args.h_size,
                                      p=args.c_dropout, image_shape=args.img_shape)
        self.action_emb = nn.Embedding(args.act_dim, args.h_size)
        self.action_emb.weight.requires_grad = False
        # cnn1d out will be 248 from 128*8=1024 length 
        self.cnn_1d_sensor = nn.Conv1d(8, 8, kernel_size=8, stride=4, groups=8)
        # stacking im and cnn1d-out will be 128+248=376
        self.rnn = nn.GRU(376, args.h_size, num_layers=1)
        self.out = nn.Linear(args.h_size, args.act_dim)


    def forward(self, X_img, X_sensor, X_act):
        n_frames, b_size = X_img.shape[0], X_img.shape[1]
        hidden_states = []
        dec_in = torch.ones(1, b_size, self.args.h_size, requires_grad=False, device=self.args.device)
        for i in range(n_frames):
            X_state_h = self.sensor_net(X_img[i], X_sensor[:,i])
            X_state_h.append(self.action_emb(X_act[i]))
            X_sensor_h = torch.stack(X_state_h[1:], axis=1)
            X_sensor_h = self.cnn_1d_sensor(X_sensor_h).view(b_size, -1)
            X_state_h = torch.concat([X_state_h[0],X_sensor_h], axis=1)
            hidden_states.append(X_state_h)

        hidden_states = torch.stack(hidden_states, axis=0) # seqLen X batchSize X h_size 
        out, h = self.rnn(hidden_states)
        return self.out(F.relu(h.squeeze(0)))


class DriveDQN_simple_fusion2_single_act_dec(nn.Module):
    """-Changing sensor fusion to simple concatenation
       -Only decode to single action"""
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.cnn = Nature_Paper_Conv_Dropout_Torch(args.in_channels, args.h_size, p=args.dropout_cnn, image_shape=args.img_shape)
        self.internal_sensor_net = nn.Sequential(nn.Linear(7, args.h_size),
                                                 nn.ReLU(),
                                                 nn.Linear(args.h_size, args.h_size))        
        self.temporal_net = nn.Transformer(d_model=args.h_size*2, nhead=args.n_head, num_encoder_layers=args.n_encs_t,
                                         num_decoder_layers=args.n_decs_t, dim_feedforward=4*args.h_size, 
                                         dropout=args.t_dropout, norm_first=args.device, device=args.device)
        self.positional_encoder = PositionalEncoding(d_model=args.h_size*2, max_len=64) # max number of time steps
    
        self.out = nn.Linear(args.h_size*2, args.act_dim)
        # cnn1d out will be 248 from 128*8=1024 length 
        # stacking im and cnn1d-out will be 128+248=376

    def forward(self, X_img, X_sensor, X_act):
        n_frames, b_size = X_img.shape[0], X_img.shape[1]
        hidden_states = []
        for i in range(n_frames):
            X_img_h = self.cnn(X_img[i])
            X_sensor_h = X_sensor[:,i].transpose(0,2).squeeze(0)
            X_sensor_h = self.internal_sensor_net(X_sensor_h)
            X_state_h = torch.concat([X_img_h, X_sensor_h], axis=1)
            hidden_states.append(X_state_h)

        hidden_states = torch.stack(hidden_states, axis=0) # seqLen X batchSize X h_size 
        hidden_states = self.positional_encoder(hidden_states)
        dec_in = hidden_states[-1].unsqueeze(0)
        # dec_in = self.action_emb(self.act_dec_idx.repeat(b_size, 1)).transpose(0,1)
        hidden_state = self.temporal_net(hidden_states, dec_in) # seq_len X batchSize X h_size 
        hidden_state = hidden_state.transpose(0, 1) # seq,batch,... -> batch,seq,...
        return self.out(F.relu(hidden_state)).squeeze(1)


class DriveDQN_simple_fusion2_decoder(nn.Module):
    """-Changing sensor fusion to simple concatenation
       -Only decode to single action"""
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.cnn = Nature_Paper_Conv_Dropout_Torch(args.in_channels, args.h_size, p=args.dropout_cnn, image_shape=args.img_shape)
        self.internal_sensor_net = nn.Sequential(nn.Linear(7, args.h_size),
                                                 nn.ReLU(),
                                                 nn.Linear(args.h_size, args.h_size))
        d_size = 2*args.h_size 
        decoder_layer = nn.TransformerDecoderLayer(d_model=d_size, nhead=args.n_head, 
                                           dim_feedforward=4*d_size, dropout=args.t_dropout, norm_first=args.norm_first)
        self.temporal_decoder_net = nn.TransformerDecoder(decoder_layer, num_layers=args.n_decs)
        self.positional_encoder = PositionalEncoding(d_model=args.h_size*2, max_len=64) # max number of time steps
    
        self.out = nn.Linear(d_size, args.act_dim)

    def forward(self, X_img, X_sensor, X_act):
        n_frames, b_size = X_img.shape[0], X_img.shape[1]
        hidden_states = []
        for i in range(n_frames):
            X_img_h = self.cnn(X_img[i])
            X_sensor_h = X_sensor[:,i].transpose(0,2).squeeze(0)
            X_sensor_h = self.internal_sensor_net(X_sensor_h)
            X_state_h = torch.concat([X_img_h, X_sensor_h], axis=1)
            hidden_states.append(X_state_h)

        hidden_states = torch.stack(hidden_states, axis=0) # seqLen X batchSize X h_size 
        hidden_states = self.positional_encoder(hidden_states)
        mask = self._generate_square_subsequent_mask(self.args.n_frames)
        hidden_state = self.temporal_decoder_net(tgt=hidden_states, memory=hidden_states,
                                                 tgt_mask=mask, memory_mask=mask) # seq_len X batchSize X h_size 
        hidden_state = hidden_state[-1] # seq,batch,... -> batch,...
        return self.out(F.relu(hidden_state))

    def _generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float()
        mask = mask.masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

class DriveDQN_simple_fusion2_lstm(nn.Module):
    """-Changing sensor fusion to simple concatenation
       -Use GRU rnn to decode single action state for q values"""
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.cnn = Nature_Paper_Conv_Dropout_Torch(args.in_channels, args.h_size, p=args.dropout_cnn, image_shape=args.img_shape)
        # cnn1d out will be 248 from 128*8=1024 length
        self.internal_sensor_net = nn.Sequential(nn.Linear(7, args.h_size),
                                                 nn.ReLU(),
                                                 nn.Linear(args.h_size, args.h_size))
        # stacking im and cnn1d-out will be 128+248=376
        self.rnn = nn.LSTM(args.h_size*2, args.h_size, num_layers=2)
        self.out = nn.Linear(args.h_size, args.act_dim)

    def forward(self, X_img, X_sensor, X_act):
        n_frames, b_size = X_img.shape[0], X_img.shape[1]
        hidden_states = []
        for i in range(n_frames):
            X_img_h = self.cnn(X_img[i])
            X_sensor_h = X_sensor[:,i].transpose(0,2).squeeze(0)
            X_sensor_h = self.internal_sensor_net(X_sensor_h)
            X_state_h = torch.concat([X_img_h, X_sensor_h], axis=1)
            hidden_states.append(X_state_h)

        hidden_states = torch.stack(hidden_states, axis=0) # seqLen X batchSize X h_size 
        out, (h_n, c_n) = self.rnn(hidden_states)
        h_n = h_n[-1]
        return self.out(F.relu(h_n))


class DriveDQN_cnn_lstm(nn.Module):
    """-Changing sensor fusion to simple concatenation
       -Use GRU rnn to decode single action state for q values"""
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.cnn = Nature_Paper_Conv_Dropout_Torch2(args.in_channels, p=args.c_dropout, image_shape=args.img_shape)
        # stacking im and cnn1d-out will be 128+248=376
        self.rnn = nn.GRU(512, args.h_size, num_layers=1)
        self.out = nn.Linear(args.h_size, args.act_dim)


    def forward(self, X_img, X_sensor, X_act):
        n_frames, b_size = X_img.shape[0], X_img.shape[1]
        hidden_states = []
        for i in range(n_frames):
            X_img_h = self.cnn(X_img[i])
            hidden_states.append(X_img_h)

        hidden_states = torch.stack(hidden_states, axis=0) # seqLen X batchSize X h_size 
        out, h = self.rnn(hidden_states)
        return self.out(F.relu(h.squeeze(0)))


class DriveDQN_cnn(nn.Module):
    """-Changing sensor fusion to simple concatenation
       -Use GRU rnn to decode single action state for q values"""
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.cnn = Nature_Paper_Conv_Dropout_Torch2(args.in_channels+1, p=args.c_dropout, image_shape=args.img_shape)
        self.out = nn.Linear(512, args.act_dim)

    def forward(self, X_img, X_sensor, X_act):
        X = X_img.squeeze(0)
        X = self.cnn(X)
        return self.out(X)
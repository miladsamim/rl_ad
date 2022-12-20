import torch 
import torch.nn as nn
import torch.nn.functional as F

from typing import Dict, List, Tuple, TypeVar, Union

from parameters.architectures_torch import Nature_Paper_Conv_Dropout_Torch

class PACTPretrain(nn.Module):
    """Implements a PACT model for pretraining in the car racing env."""
    def __init__(self, args) -> None:
        super().__init__()
        self.model = PACTBase(args)
        self.head = PACTPretrainHead(args.h_size*4, args.h_size*2)
        self.state_criterion = nn.MSELoss()

    def forward(self, batch: Dict[str, torch.Tensor]):
        out_embd, state_embd = self.model(batch)
        state_pred = self.head(out_embd)
        return state_pred, state_embd

class PACTPretrainHead(nn.Module):
    """Implements a PACT Pretrain Head for the Car Racing env.
       Only for decoding s_t+1 (not a_t+1)"""
    def __init__(self, in_dim, out_dim) -> None:
        super().__init__()
        self.state_head = nn.Sequential(
            nn.Linear(in_dim, out_dim)
        )
    
    def forward(self, X:torch.tensor) -> torch.tensor:
        t, b, c = X.size()
        state_action_out_embd = X.reshape(t//2, b, -1)
        state_preds = self.state_head(state_action_out_embd)
        return state_preds

class PACTBase(nn.Module):
    """Implements a PACT Base class. Raw data -> Tokens ->
       Autoregressive State Decodings"""
    def __init__(self, args) -> None:
        super().__init__()
        self.args = args
        self.h_size = h_size = args.h_size*2 # concat of image and measures embds.
        t = args.n_frames # max seq len
        self.tokenzier = PACTTokenizer(args)
        encoder_layer = nn.TransformerEncoderLayer(d_model=h_size, nhead=args.n_head, 
                                           dim_feedforward=4*h_size, dropout=args.t_dropout, norm_first=args.norm_first)
        self.gpt = nn.TransformerEncoder(encoder_layer, num_layers=args.n_decs)
        self.pos_embd_local = nn.Embedding(t+1, h_size)
        self.pos_embd_global = nn.Embedding((t+1)*2, h_size)

    def forward(
        self, batch: Dict[str, torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        args = self.args
        t, b = batch['state']['X_img'].shape[:2]

        # local and global position embedding
        pos_local = torch.arange(t, dtype=torch.long, device=args.device).unsqueeze(-1)
        pos_global = torch.arange(2*t, dtype=torch.long, device=args.device).unsqueeze(-1)
        pos_local_h = self.pos_embd_local(pos_local).repeat_interleave(2, dim=0)
        pos_global_h = self.pos_embd_global(pos_global)

        tok_embd_dict = self.tokenzier(batch)
        embd_list = [tok_embd_dict[input_type] for input_type in ["state", "action"]]

        state_act_seq = torch.stack(embd_list, dim=1).reshape(t*2, b, self.h_size)
        mask = self._generate_square_subsequent_mask(t*2).to(args.device)
        out_embd = self.gpt(state_act_seq + pos_local_h + pos_global_h,
                            mask=mask)
        
        return out_embd, tok_embd_dict['state']

    def _generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float()
        mask = mask.masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        mask.requires_grad = False 
        return mask

class PACTTokenizer(nn.Module):
    """Implements a PACT Tokenizer for the Car Racing env."""
    def __init__(self, args) -> None:
        super().__init__()
        self.args = args
        self.state_tokenizer = PACTStateTokenizer(args)
        self.action_tokenizer = PACTActionTokenizer(args)
        self.h_size = args.h_size*2
    
    def forward(self, 
                batch: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        args = self.args
        X_img, X_sensor = batch['state'].values()
        X_act_idx, X_act = batch['action'].values()
        t, b = X_img.shape[:2]
        if args.accel: # if memory allows reshaping is faster 
            states_h = self.state_tokenizer(
                                X_img.reshape(-1, args.in_channels, *args.img_shape), 
                                X_sensor.reshape(7,-1,1)
                            ).reshape(t, b, self.h_size)
        else:
            states_h = []
            for i in range(t):
                states_h.append(self.state_tokenizer(X_img[i], 
                               X_sensor[:,i]))
            states_h = torch.stack(states_h, dim=0)
        actions_h = self.action_tokenizer(X_act_idx, X_act)
        
        return {'state': states_h, 'action': actions_h}

class PACTActionTokenizer(nn.Module):
    """Implements a PACT Action Tokenizer for the Car Racing env."""
    def __init__(self, args) -> None:
        super().__init__()
        self.h_size = 2*args.h_size 
        self.steer = self._build_real_tokenizer(1, self.h_size)
        self.throttle = self._build_real_tokenizer(1, self.h_size)
        self.brake = self._build_real_tokenizer(1, self.h_size)
        self.action_emb = nn.Embedding(args.act_dim, self.h_size)
        self.merge_actions = nn.Sequential(nn.Conv1d(in_channels=3, out_channels=9, kernel_size=9, padding='same'),
                                      nn.ReLU(),
                                      nn.Conv1d(in_channels=9, out_channels=1, kernel_size=9, padding='same'))

    def forward(self, X_act_idx, X_act):
        n_acts, t, b, in_h = X_act.size()
        # discrete 
        actions_d_h = self.action_emb(X_act_idx)
        # real
        actions_h_steer = self.steer(X_act[0])
        actions_h_throttle = self.throttle(X_act[1])
        actions_h_brake = self.brake(X_act[2])
        actions_r_h = torch.stack([actions_h_steer, actions_h_throttle, actions_h_brake], dim=2)
        actions_r_h = self.merge_actions(actions_r_h.view(-1,3,self.h_size)).reshape(t,b,self.h_size)
        return actions_d_h + actions_r_h


    def _build_real_tokenizer(self, in_dim, h_size):
        """Builds tokenizer for real valued input tensors.
           Assummes in_dim < 10"""
        return nn.Sequential(
                    nn.Linear(in_dim, 32),
                    nn.ReLU(),
                    nn.Linear(32, 64),
                    nn.ReLU(),
                    nn.Linear(64, h_size)
                )

class PACTStateTokenizer(nn.Module):
    """Implements a PACT State Tokenizer for the Car Racing env."""
    def __init__(self, args) -> None:
        super().__init__()
        self.cnn = Nature_Paper_Conv_Dropout_Torch(args.in_channels, args.h_size, 
                                                    p=args.c_dropout, image_shape=args.img_shape)
        self.steering = self._build_real_tokenizer(1, args.h_size)
        self.speed = self._build_real_tokenizer(1, args.h_size)
        self.gyro = self._build_real_tokenizer(1, args.h_size)
        self.abs1 = self._build_real_tokenizer(1, args.h_size)
        self.abs2 = self._build_real_tokenizer(1, args.h_size)
        self.abs3 = self._build_real_tokenizer(1, args.h_size)
        self.abs4 = self._build_real_tokenizer(1, args.h_size)

        n_measures = 7
        self.merge_measures = nn.Sequential(
                                    nn.Conv1d(in_channels=7, out_channels=n_measures*3, kernel_size=9, padding='same'),
                                    nn.ReLU(),
                                    nn.Conv1d(in_channels=n_measures*3, out_channels=1, kernel_size=9, padding='same'),
                                )

    def forward(self, X_img, X_sensors):
        X_img_h = self.cnn(X_img)
        X_steering_h = self.steering(X_sensors[0])
        X_speed_h = self.speed(X_sensors[1])
        X_gyro_h = self.gyro(X_sensors[2])
        X_abs1_h = self.abs1(X_sensors[3])
        X_abs2_h = self.abs2(X_sensors[4])
        X_abs3_h = self.abs3(X_sensors[5])
        X_abs4_h = self.abs4(X_sensors[6])
        X_sensors_h = torch.stack([X_steering_h, X_speed_h, X_gyro_h, X_abs1_h, X_abs2_h, X_abs3_h, X_abs4_h], dim=1)
        X_sensors_h = self.merge_measures(X_sensors_h).squeeze(1)
        return torch.cat([X_img_h, X_sensors_h], dim=1)        

    def _build_real_tokenizer(self, in_dim, h_size):
        """Builds tokenizer for real valued input tensors.
           Assummes in_dim < 10"""
        return nn.Sequential(
                    nn.Linear(in_dim, 32),
                    nn.ReLU(),
                    nn.Linear(32, 64),
                    nn.ReLU(),
                    nn.Linear(64, h_size)
                )
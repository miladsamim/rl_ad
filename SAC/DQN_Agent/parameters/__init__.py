from parameters.dotdict import dotdict
from parameters.params import model_params
from parameters.sensor_net import SensorModel
from parameters.positional_encoding import PositionalEncoding
from parameters.architectures_torch import Nature_Paper_Conv_Dropout_Torch
from parameters.memory_buffer import MemoryBufferSimple, MemoryBufferSeparated
from parameters.pact_drive_net import PDriveDQN
from parameters.drive_net import (DriveDQN, 
                                  DriveDQN_cnn,
                                  DriveDQN_cnn_lstm,
                                  DriveDQN_simple_fusion, 
                                  DriveDQN_simple_fusion_lstm,
                                  DriveDQN_simple_fusion2_gru,
                                  DriveDQN_simple_fusion2_lstm,
                                  DriveDQN_simple_fusion2_decoder,
                                  DriveDQN_simple_fusion_single_act_dec,
                                  DriveDQN_simple_fusion2_single_act_dec,
                                  DriveDQN_simple_fusion_learnable_act_embs,
                                  )
name_2_model = {
                'PDriveDQN': PDriveDQN,
                'DriveDQN': DriveDQN,
                'DriveDQN_cnn': DriveDQN_cnn,
                'DriveDQN_cnn_lstm': DriveDQN_cnn_lstm,
                'DriveDQN_simple_fusion': DriveDQN_simple_fusion,
                'DriveDQN_simple_fusion_lstm': DriveDQN_simple_fusion_lstm,
                'DriveDQN_simple_fusion2_lstm': DriveDQN_simple_fusion2_lstm,
                'DriveDQN_simple_fusion2_gru': DriveDQN_simple_fusion2_gru,
                'DriveDQN_simple_fusion2_decoder': DriveDQN_simple_fusion2_decoder,
                'DriveDQN_simple_fusion_single_act_dec': DriveDQN_simple_fusion_single_act_dec,
                'DriveDQN_simple_fusion2_single_act_dec': DriveDQN_simple_fusion2_single_act_dec,
                'DriveDQN_simple_fusion_learnable_act_embs': DriveDQN_simple_fusion_learnable_act_embs,
            }
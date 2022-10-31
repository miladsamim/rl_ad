import torch.nn as nn

def conv_block(c_input, c_output, *args, **kwargs):
    layers = []
    layers.append(nn.Conv2d(c_input,c_output, *args, **kwargs))
    layers.append(nn.BatchNorm2d(c_output))
    layers.append(nn.ReLU())
    return layers

def fc_block(c_input,c_output, *args, **kwargs):
    layers = []
    layers.append(nn.Linear(c_input,c_output, *args, **kwargs))
    layers.append(nn.ReLU())
    return layers

def cnn(output_dim, output_act):
    layers = []
    layers.extend(conv_block(3,24,kernel_size=5, stride=2))
    layers.extend(conv_block(24,36,kernel_size=5, stride=2))
    layers.extend(conv_block(36,48,kernel_size=5, stride=2))
    layers.extend(conv_block(48,64,kernel_size=3, stride=1))
    layers.append(nn.Flatten())
    layers.extend(fc_block(3136, 192))
    layers.extend(fc_block(192, 64))
    layers.append(nn.Linear(64, output_dim))
    layers.append(output_act)
    return nn.Sequential(*layers)

class Network(nn.Module):

    def __init__(self, input_dimension, output_dimension, output_activation=nn.Identity()):
        super(Network, self).__init__()
        self.cnn = cnn(output_dimension, output_activation)

    def forward(self, state):
        output = self.cnn(state)
        return output

# class Network(nn.Module):

#     def __init__(self, input_dimension, output_dimension, output_activation=nn.Identity()):
#         super(Network, self).__init__()
#         self.layer_1 = nn.Linear(in_features=input_dimension, out_features=64)
#         self.layer_2 = nn.Linear(in_features=64, out_features=64)
#         self.output_layer = nn.Linear(in_features=64, out_features=output_dimension)
#         self.output_activation = output_activation

#     def forward(self, inpt):
#         layer_1_output = nn.functional.relu(self.layer_1(inpt))
#         layer_2_output = nn.functional.relu(self.layer_2(layer_1_output))
#         output = self.output_activation(self.output_layer(layer_2_output))
#         return output

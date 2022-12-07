import torch.nn as nn

from parameters.architectures_torch import Nature_Paper_Conv_Dropout_Torch

class SensorModel(nn.Module):
    def __init__(self, in_channels, h_size, p=0.5, image_shape=(96,96)):
        super().__init__()
        self.cnn = Nature_Paper_Conv_Dropout_Torch(in_channels, h_size, p=p, image_shape=image_shape)
        self.steering = nn.Linear(1, h_size)
        self.speed = nn.Linear(1, h_size)
        self.gyro = nn.Linear(1, h_size)
        self.abs1 = nn.Linear(1, h_size)
        self.abs2 = nn.Linear(1, h_size)
        self.abs3 = nn.Linear(1, h_size)
        self.abs4 = nn.Linear(1, h_size)

    def forward(self, X_img, X_sensors):
        X_img_h = self.cnn(X_img)
        X_steering_h = self.steering(X_sensors[0])
        X_speed_h = self.speed(X_sensors[1])
        X_gyro_h = self.gyro(X_sensors[2])
        X_abs1_h = self.abs1(X_sensors[3])
        X_abs2_h = self.abs2(X_sensors[4])
        X_abs3_h = self.abs3(X_sensors[5])
        X_abs4_h = self.abs4(X_sensors[6])
        return [X_img_h, X_steering_h, X_speed_h, X_gyro_h, X_abs1_h, X_abs2_h, X_abs3_h, X_abs4_h]
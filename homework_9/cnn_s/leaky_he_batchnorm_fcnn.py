from torch import nn

class leakyFCNN_he_batchnorm(nn.Module):
    """
    Leaky Fully Connected Neural Network model with He Initialization and BatchNorm
    """

    def __init__(self, in_dim, out_dim):
        super(leakyFCNN_he_batchnorm, self).__init__()

        # 1st FC layer with BatchNorm
        self.fc_input = nn.Sequential(nn.Linear(in_dim, 512),
                                      nn.LeakyReLU(),
                                      nn.BatchNorm1d(512))
        # 2nd FC layer with BatchNorm
        self.fc_hidden = nn.Sequential(nn.Linear(512, 256),
                                       nn.LeakyReLU(),
                                       nn.BatchNorm1d(256))
        # Output FC layer
        self.fc_output = nn.Sequential(nn.Linear(256, out_dim))

        # Initialize weights
        self._initialize_weights()


    def _initialize_weights(self):

        for layer in self.modules():
            if isinstance(layer, nn.Linear):
                nn.init.kaiming_normal_(layer.weight, mode = 'fan_out', nonlinearity = 'leaky_relu')

                if layer.bias is not None:
                    nn.init.constant_(layer.bias, 0)


    def forward(self, x):

        x1 = self.fc_input(x)
        x2 = self.fc_hidden(x1)
        output = self.fc_output(x2)

        return output
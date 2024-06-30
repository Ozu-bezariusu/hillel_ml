from torch import nn

class primitiveFCNN(nn.Module):
    """
    Primitive Fully Connected Neural Network model
    """

    def __init__(self, in_dim, out_dim):
        super(primitiveFCNN, self).__init__()

        # 1st FC layer
        self.fc_input = nn.Sequential(nn.Linear(in_dim, 512),
                                      nn.Sigmoid())
        # 2nd FC layer
        self.fc_hidden = nn.Sequential(nn.Linear(512, 256),
                                       nn.Sigmoid())
        # Output FC layer
        self.fc_output = nn.Sequential(nn.Linear(256, out_dim))


    def forward(self, x):

        x1 = self.fc_input(x)
        x2 = self.fc_hidden(x1)
        output = self.fc_output(x2)

        return output
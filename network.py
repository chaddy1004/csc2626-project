from torch.nn import Module, Linear, ReLU, Sequential, Softmax, Conv2d, MaxPool2d, Flatten


class Actor(Module):
    def __init__(self, state_shape, n_actions):
        super(Actor, self).__init__()

        self.conv_block1 = Sequential(
            Conv2d(in_channels=state_shape[-1], out_channels=64, kernel_size=3, stride=1, padding=1),
            ReLU())
        self.conv_block2 = Sequential(Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1),
                                      ReLU())

        self.downsample = MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.flatten = Flatten()

    def forward(self, x):
        pass


class Critic(Module):
    def __init__(self, state_shape):
        super(Critic, self).__init__()
        self.conv_block1 = Sequential(
            Conv2d(in_channels=state_shape[-1], out_channels=64, kernel_size=3, stride=1, padding=1),
            ReLU())
        self.conv_block2 = Sequential(Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1),
                                      ReLU())
        self.downsample = MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.flatten = Flatten()

    def forward(self, x):
        pass

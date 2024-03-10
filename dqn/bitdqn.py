class BitDQN(nn.Module):

    def __init__(self, n_observations, n_actions):
        super(BitDQN, self).__init__()
        self.layer1 = BitLinear(n_observations, 128)
        self.layer2 = BitLinear(128, 128)
        self.layer3 = BitLinear(128, n_actions)

    # Called with either one element to determine next action, or a batch
    # during optimization. Returns tensor([[left0exp,right0exp]...]).
    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        return self.layer3(x)

from torch import nn

class Net(nn.Module):
    def __init__(self,input_size,hidden_size,output_size):
        super(Net,self).__init__()

        self.hidden = nn.Linear(input_size,hidden_size)
        self.relu = nn.ReLU()
        self.output = nn.Linear(hidden_size, output_size)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, X):
        feature = self.relu(self.hidden(X))
        output = self.softmax(self.output(feature))
        return output
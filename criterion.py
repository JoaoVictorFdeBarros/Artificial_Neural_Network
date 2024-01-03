from torch import nn
from device import set_device

criterion = nn.CrossEntropyLoss().to(set_device())
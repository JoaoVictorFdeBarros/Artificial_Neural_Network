from torch import optim

def set_optimizer(net,step):
    optmizer = optim.SGD(net.parameters(),lr=step, weight_decay=0.0000000000000000000000001)
    return optmizer
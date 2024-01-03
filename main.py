import torch
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from torchsummary import summary
from net import Net
from device import set_device
from criterion import criterion
from plot_net_boundary import plot_boundary
from optimizer import set_optimizer
from plot_error import plot_error

from sklearn.datasets import load_iris


hidden_layer_size = 1024
n_interations = 10000
step = 0.0001
n_prints = 75

scaler = StandardScaler()
dataset = load_iris()
data = scaler.fit_transform(dataset.data[:,[0,2]])

print(dataset.feature_names)

target = dataset.target
output_size = len(dataset.target_names)



input_size = data.shape[1]

net = Net(input_size, hidden_layer_size,output_size).to(set_device())
summary(net,input_size=(1,input_size))

Xtns = torch.from_numpy(data).float().to(set_device())
Ytns = torch.from_numpy(target).to(set_device())

error_array = []

fig, axs = plt.subplots(2, 1, figsize=[9, 9])
plt.subplots_adjust(left=0.1, right=0.9, top=0.95, bottom=0.05, hspace=0.3)


for i in range(n_interations):
    pred = net(Xtns).to(set_device())
    loss = criterion(pred,Ytns)
    optmizer = set_optimizer(net,step)

    error_array.append(loss.item())

    if i%(int(n_interations/n_prints)) == 0 or i==n_interations -1:
        plot_boundary(data, target, net,axs[0],"Comprimento do caule","Comprimento da pétala")
        plot_error([i for i in range(0, i + 1)], error_array,axs[1])

        plt.pause(0.0005)

    loss.backward()
    optmizer.step()

plt.show()


import torch
import numpy as np
import matplotlib.pyplot as plt
from device import set_device

def plot_boundary(X, y, model,subplot,y_name,x_name, class_labels):
  plt.sca(subplot)
  plt.title("Estado atual")
  plt.xlabel(x_name + " - Normalizado")
  plt.ylabel(y_name + " - Normalizado") 

  x_min, x_max = X[:, 0].min()-0.1, X[:, 0].max()+0.1
  y_min, y_max = X[:, 1].min()-0.1, X[:, 1].max()+0.1
  
  spacing = min(x_max - x_min, y_max - y_min) / 100
  
  XX, YY = np.meshgrid(np.arange(x_min, x_max, spacing),
                       np.arange(y_min, y_max, spacing))
  
  data = np.hstack((XX.ravel().reshape(-1,1), 
                    YY.ravel().reshape(-1,1)))
  
  db_prob = model(torch.Tensor(data).to(set_device()) )
  clf = np.argmax(db_prob.cpu().data.numpy(), axis=-1)
  
  Z = clf.reshape(XX.shape)
  
  plt.contourf(XX, YY, Z, cmap=plt.cm.brg, alpha=0.5)
  scatter = plt.scatter(X[:, 0], X[:, 1], c=y, edgecolors='k', s=25, cmap=plt.cm.brg)
  legend_labels = [f'Iris-{label}' for label in class_labels]
  legend = plt.legend(handles=scatter.legend_elements()[0], labels=legend_labels)
  legend.get_frame().set_facecolor('yellow')
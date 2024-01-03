import matplotlib.pyplot as plt

def plot_loss(n_interations,error,subplot):
    plt.sca(subplot)
    plt.plot(n_interations,error,color='black')
    plt.xlabel("Épocas")
    plt.ylabel("Loss")
    plt.title(str("Cross Entropy Loss"))
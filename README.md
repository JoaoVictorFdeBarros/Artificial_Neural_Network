# Objetivo

Atividade desenvolvida durante a disciplina de inteligência computacional. O objetivo foi implementar uma RNA (Rede Neural Artificial) de classificação.
A base de dados escolhida para ser classificada foi a [Iris Flower Dataset](https://scikit-learn.org/stable/auto_examples/datasets/plot_iris_dataset.html), uma base de dados bastante conhecida na literatura.
Para criar uma visualização bidimensional do processo de treinamento da rede, foram selecionados apenas 2 dos 4 atributos disponíveis na base de dados, o comprimento do caule e das pétalas.

# Implementação

- Desenvolvido na versão 3.10.12 do Python
- Validado no Ubuntu 22.04
- As principais bibliotecas usadas foram a [Torch](https://pytorch.org/), o [SciKitLearn](https://scikit-learn.org/stable/) e o [MatPlotLib](https://matplotlib.org/)

# Hiper parâmetros

Foram definidos:
- Os 2 parâmetros de entrada, já descritos, 
- Na camada oculta foram criados 1024 neurônios
- 3 neurônios na camada de saída para classificar entre as 3 possíveis classes - Iris-setosa, Iris-Versicolor, Iris-Virginica
- 10.000 épocas de treinamento
- Taxa de aprendizagem de 0,0001
- A função de ativação escolhida para os neurônios da camada oculta foi a ReLU e para a saída foi o SoftMax
- A função usada para ajustar o erro foi a SGD (Stochastic Gradient Descent)

![Hiper parâmetros](https://github.com/JoaoVictorFdeBarros/Artificial_Neural_Network/blob/master/images/HyperParams.png)

*Torch Summary*

# Resultado

![Resultado](https://github.com/JoaoVictorFdeBarros/Artificial_Neural_Network/blob/master/images/result.png)
*Resultado*

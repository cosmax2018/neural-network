
# test-neuron.py : test delal classe neuron.py
#
# CopyRight 2021 by Lumachina Software - @_°° 
# Massimiliano Cosmelli (massimiliano.cosmelli@gmail.com)

from neuron import Neuron

import random

iterazioni = 1000000
learning_rate = 0.01

# gli accoppiamenti inputs-outputs per addestrare la rete  (dataset)
# dato che per es. la mia rete neurale ha 2 inputs e 1 output, allora saranno terne
# ad es: (4,5.2,1) significa che quando come input ho la coppia i1 = 4  e i2 = 5.2
#                  allora la rete neurale deve sparare fuori 1
#                  ricordo che la rete neurale funziona in modo binario 0/1
#
inputs  = ((0,0,0),(0,0,1),(0,1,0),(0,1,1),(1,0,0),(1,0,1),(1,1,0),(1,1,1))
outputs = (0,1,0,1,0,1,0,1)
dataset = (inputs,outputs)

print('\nTest Neuron class')

#				how many inputs
neuron = Neuron(len(dataset[0][0]),	learning_rate, dataset, iterazioni)

neuron.reset()

neuron.training()

#
# adesso prova come si comporta il neurone addestrato
#

neuron.test(True)	# fa il test sull'intero dataset, se il neurone è stato ben addestrato lo deve passare al 100%

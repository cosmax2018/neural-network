
# neurons.py : implementazione di uno strato di neuroni in Python
#
# CopyRight 2021 by Lumachina Software - @_°° 
# Massimiliano Cosmelli (massimiliano.cosmelli@gmail.com)

from neurons import Neurons

import random,time

iterazioni = 100000
learning_rate = 0.1

# tutti i neuroni del layer hanno lo stesso input, ma vengono addestrati per fornire outputs diversi.
# ogni neurone del layer ha il suo sub-dataset (che corrisponde a una riga diversa del dataset)

inputs  = (((0,0,0),(0,0,1),(0,1,0),(0,1,1),(1,0,0),(1,0,1),(1,1,0),(1,1,1)),\
		   ((0,0,0),(0,0,1),(0,1,0),(0,1,1),(1,0,0),(1,0,1),(1,1,0),(1,1,1)),\
		   ((0,0,0),(0,0,1),(0,1,0),(0,1,1),(1,0,0),(1,0,1),(1,1,0),(1,1,1)),\
		   ((0,0,0),(0,0,1),(0,1,0),(0,1,1),(1,0,0),(1,0,1),(1,1,0),(1,1,1)),\
		   ((0,0,0),(0,0,1),(0,1,0),(0,1,1),(1,0,0),(1,0,1),(1,1,0),(1,1,1)),\
		   ((0,0,0),(0,0,1),(0,1,0),(0,1,1),(1,0,0),(1,0,1),(1,1,0),(1,1,1)))
outputs = ((0,1,0,1,0,1,0,1),(1,0,1,0,1,0,1,0),(0,1,0,1,0,1,0,1),(1,0,1,0,1,0,1,0),(0,1,0,1,0,1,0,1),(1,0,1,0,1,0,1,0))
dataset = (inputs,outputs)

print('\nTest Neurons class')

#					index 	how many neurons per layer		how many inputs per neuron		name	 	
neurons = Neurons(	0,		len(dataset[0]),				len(dataset[0][0][0]),			'output',	learning_rate, dataset, iterazioni)

neurons.reset()

t = time.time()

neurons.training()		# esegue l'addestramento della rete usando il dataset

print(f'\nElapsed {time.time()-t} sec.\n\n')
time.sleep(3)

#
# adesso prova come si comporta la rete addestrata
#

neurons.test(True)	# fa il test sull'intero dataset, se ogni neurone del layer è stato ben addestrato lo deve passare al 100%


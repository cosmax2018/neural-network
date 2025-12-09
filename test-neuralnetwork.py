
# test-neuralnetwork.py : implementazione di una rete neurale a strati in Python
#
# CopyRight 2021 by Lumachina Software - @_°° 
# Massimiliano Cosmelli (massimiliano.cosmelli@gmail.com)

from neuralnetwork import NeuralNetwork

import random
import numpy as np
import matplotlib.pyplot as plt

INPUT,OUTPUT = 0,1

iterazioni = 100000
learning_rate = 0.01

# tutti i neuroni di un determinato layer hanno lo stesso input,ma vengono addestrati per fornire outputs diversi.

inputs  = ((0,0,0),(0,0,1),(0,1,0),(0,1,1),(1,0,0),(1,0,1),(1,1,0),(1,1,1))
outputs = ((0,),(1,),(1,),(1,),(1,),(1,),(1,),(1,))	# xor a 3 ingressi

# inputs  = ((0,0,0),(0,0,1),(0,1,0),(0,1,1),(1,0,0),(1,0,1),(1,1,0),(1,1,1))
# outputs = ((0,0),(0,1),(0,1),(1,1),(0,0),(1,0),(0,1),(1,1))

# inputs =   ((1, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0),\
			# (0, 0, 1, 0, 0, 1, 1, 1, 0, 0, 0),\
			# (1, 0, 1, 0, 1, 1, 1, 1, 1, 1, 0),\
			# (0, 0, 1, 0, 0, 1, 1, 1, 0, 0, 0))
			
# outputs = ((1,),(0,),(1,),(0,))
	
dataset = (inputs,outputs)

number_of_hidden_neurons_layer_1 = 11					# how many hidden neurons at layer #1
number_of_hidden_neurons_layer_2 = 3					# how many hidden neurons at layer #2
number_of_output_neurons_layer = len(outputs[0])		# how many output neurons

# how layers are made (2 x hiddens + 1 x output layers)
number_of_neurons_per_layer = (number_of_output_neurons_layer,	\
							   number_of_hidden_neurons_layer_2,\
							   number_of_hidden_neurons_layer_1)
							   
layers_names = ('output',	\
				'hidden#2',	\
				'hidden#1')
				
neuralnet = NeuralNetwork(dataset,						\
						  iterazioni,					\
						  learning_rate,				\
						  number_of_neurons_per_layer,	\
						  layers_names,					\
						  True)
						  
mostra_menu,not_trained = True,True
						  
while mostra_menu:
	print('\n')
	print('[R]eset')	
	print('[G]o Training')
	print('[T]est')
	print('[P]lot Deltas')
	print('[S]ave')
	print('[L]oad')
	print('[F]orecast')
	print('[E]xit')
	if not_trained:
		print('\n *** rete non addestrata ***')
	else:
		print('\n')	
		
	risposta = input('>')
		
	# esegue l'addestramento della rete usando il dataset
	if risposta.upper() == 'G':
		neuralnet.training()
		not_trained = False
		
	# reset della rete
	if risposta.upper() == 'R':
		neuralnet.reset()
		not_trained = True
		
	# grafica l'andamento dei delta per ciascun layer escluso quello di input (layer 3) - grafici uniti
	if risposta.upper() == 'P':
		try:
			num_curves = 3
			x = np.linspace(0,1,iterazioni)
			y = np.zeros((x.size,num_curves))

			for i in range(len(number_of_neurons_per_layer)-1):
				y[:,i] = neuralnet.get_deltas(i)
				
			plt.title("layers deltas",size=24,weight='bold')
			plt.xlabel("ITERAZIONI")
			plt.ylabel("DELTA")
			plt.plot(x,y)
			plt.show()
		except ValueError:
			print("Plot unavailable. Calculate deltas values before..")
		
	# adesso prova come si comporta la rete addestrata col dataset usato per l'addestramento
	if risposta.upper() == 'T':
		neuralnet.test(True)
		
	# produce quindi una previsione
	if risposta.upper() == 'F':
		str_input = input(f'Dammi una tupla di input lunga {len(inputs[0])} separata da virgole (es. 1,0,0,..) : ')
		_input_ = tuple(map(int, str_input.split(',')))
		neuralnet.fire(_input_)
		_output_ = neuralnet.get_sigmoid_outputs(True)	# output with normalization = true
		print(f"La previsione per l'input: {_input_} e' {_output_}")
		
	# salvataggio dei pesi e dei bias della rete
	if risposta.upper() == 'S':
		filename = input("Se vuoi salvare i pesi e i bias della rete dammi il nome del file, altrimenti premi invio:")
		if filename != None and filename != '':
			neuralnet.save(filename)
		
	# caricamento dei pesi e dei bias della rete
	if risposta.upper() == 'L':
		filename = input("Se vuoi caricare i pesi e i bias della rete dammi il nome del file, altrimenti premi invio:")
		if filename != None and filename != '':
			neuralnet.load(filename)
			not_trained = False
			
	if risposta.upper() == 'E':
		mostra_menu = False

from Hyperspectra_grids import sliding_window, extract_grids, WindowSize
from models import LoadModel
import torch.utils.data as utils
import torch.nn as nn
import torch
import numpy as np
import ast
import random
import math

import os
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

from Dataloaders.Pavia_dataloader import Dataloader as pavia
from Dataloaders.Botswana_dataloader import Dataloader as botswana
from Dataloaders.indian_pines_dataloader import Dataloader as indian
from Dataloaders.KSC_dataloader import Dataloader as ksc
#from Dataloaders.Salinas_dataloader import Dataloader as salinas
#from Dataloaders.SalinasA_dataloader import Dataloader as salinasa
#from Dataloaders.Samson_dataloader import Dataloader as samson

'''
Aby rozpocząć TEST nalży wybrac odpowiedni zestaw danych:
        
    Data Sets: PaviaU | IndianPines | Botswana | KSC | Salinas | Samson

Przypisać zmiennej #name = odpowiedni dataset: 

'''

name = 'IndianPines'


if name == 'PaviaU':
    my_dataloader = pavia()
elif name == 'Botswana':
    my_dataloader = botswana()
elif name == 'IndianPines':
    my_dataloader = indian()
elif name == 'KSC':
    my_dataloader = ksc()
#elif name == 'Salinas':
#    my_dataloader = salinas()
#elif name == 'Samson':
#    my_dataloader = samson()
#elif name == 'SalinasA':
#    my_dataloader = salinasa()

'''
param: Height, Width, Bands: Size of patches 
param: D, H, W: Size of 3D image
param: samples: How many patches will be extracted (2400 = 20>x>10)
param: calsses: Labels in HSI_gt image
'''
Width, Height, Bands, samples, D, H, W, classes, patch = LoadModel(name)
#------------------------------------------------------------------------------
image = my_dataloader.get_image()
np.save('./Loadset/'+ name + '/image_uint8.npy',image)
image_uint8 = './Loadset/'+ name +'/image_uint8.npy'
labels = my_dataloader.get_labels()
np.save('./Loadset/'+ name + '/labels_uint8.npy',labels)
labels_uint8 = './Loadset/'+ name + '/labels_uint8.npy'
#------------------------------------------------------------------------------
window_size = WindowSize(Width, Height)
#------------------------------------------------------------------------------
window_size = (Width,Height)
#extract_grids on (uint8 numpy image(image_np_uint8), numpy gt, window_size, number of samples (1200))
data_extract = extract_grids(image_uint8, labels_uint8, window_size, patch)
#using torch.device to decide witch subassembly of PC to use
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
#------------------------------------------------------------------------------
# Hyperparameter
learning_rate = 0.001
#------------------------------------------------------------------------------
'''DATA EXTRACT'''
my_data = list(map(lambda x: torch.Tensor(x), data_extract[0][0]))
list_of_training_items = my_data[max(0,int(math.floor(len(my_data) *0.6))):]
#Treningowy
Training_data = list_of_training_items[max(0,int(math.ceil(len(list_of_training_items) *0.6))):]
Training = torch.stack(Training_data)
#Walidacyjny
Validation_data = list_of_training_items[max(0,int(math.floor(len(list_of_training_items) *0.4))):]
Validation = torch.stack(Validation_data) 
#------------------------------------------------------------------------------
'''LABELS EXTRACT'''
my_data_labels = list(map(lambda x: torch.Tensor(x), data_extract[0][1]))

list_of_training_labels = my_data_labels[max(0,int(math.floor(len(my_data_labels) *0.6))):]

Training_data_labels = list_of_training_labels[max(0,int(math.ceil(len(list_of_training_labels) *0.6))):]
Training_labels = torch.stack(Training_data_labels)

Validation_data_labels = list_of_training_labels[max(0,int(math.floor(len(list_of_training_labels) *0.4))):]
Validation_labels = torch.stack(Validation_data_labels)
#------------------------------------------------------------------------------
'''TEST, który dla funkcji ostatecznej pobiera cały obraz '''
my_TEST_data = my_data[max(0,int(math.ceil(len(my_data) *0.4))):]
my_TEST_tensor = torch.stack(my_TEST_data)

my_TEST_data_labels = my_data_labels[max(0,int(math.ceil(len(my_data_labels) *0.4))):]
my_TEST_tensor_labels = torch.stack(my_TEST_data_labels)
#------------------------------------------------------------------------------
'''DATALOADERS'''
my_training_training_tensor_dataset = utils.TensorDataset(Training, Training_labels)
Training_loader = utils.DataLoader(my_training_training_tensor_dataset)

my_validation_tensor_dataset = utils.TensorDataset(Validation, Validation_labels)
Validation_Loader = utils.DataLoader(my_validation_tensor_dataset)

my_TEST_tensor_dataset = utils.TensorDataset(my_TEST_tensor, my_TEST_tensor_labels)
my_TEST_dataloader = utils.DataLoader(my_TEST_tensor_dataset)
#------------------------------------------------------------------------------

class C3D(nn.Module):
    def __init__(self, output, _stride, kernel_size, D_out, H_out, W_out, output2, kernel_size2, _stride2):
        super(C3D, self).__init__()     
        self.layer1 = nn.Sequential(
            nn.Conv3d(1, output, kernel_size, stride=_stride),
            nn.ReLU(),
            nn.Conv3d(output, output2, kernel_size2, stride=_stride2),
            nn.Softmax(dim=1))
        self.fc = nn.Linear(D_out*H_out*output2*W_out, D_out*H_out)   
        
    def forward(self, x, D_out, H_out):
        out = self.layer1(x)
        out = out.reshape(out.size(0), -1)
        out = self.fc(out)
        out = out.reshape(D_out, H_out)

        return out 

# Convolutional neural network (CNN)
def MODEL(model, num_epochs, o1, o2, D_out, H_out):
    #------------------------------------------------------------------------------  
    show = False
    if show:
        print('o1: ', o1)
        print('o2: ', o2)
        print('num_epochs: ', num_epochs)
        print('D_out: ', D_out)
        print('H_out: ', H_out)
    
    if show:
        print(model)
    #------------------------------------------------------------------------------
    #Model loss
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr = learning_rate)
    if show:
        print(optimizer)
    #------------------------------------------------------------------------------    
    # Train the model           
    for epoch in range(num_epochs):
        for i, data in enumerate(Training_loader):
            img, labels = data
            img = img.reshape(1, 1, Height, Width, Bands)
            labels = labels.reshape(1, 1, Height, Width)
            img = img.to(device)
            labels = labels.to(device=device, dtype=torch.int64)
            
            # ===================forward=====================
            output = model(img, D_out, H_out)
            labels = labels[0,0,o1:-o2, 0]  
            loss = criterion(output, labels)          
            
            # ===================backward====================
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()  
    #------------------------------------------------------------------------------            
        model.eval()
        correct = 0
        total = 0                 
        for img, labels in Validation_Loader:
            img = img.reshape(1, 1, Height, Width, Bands)
            labels = labels.reshape(1, 1, Height, Width)
            img = img.to(device)
            labels = labels.to(device=device, dtype=torch.int64)
            outputs = model(img, D_out, H_out)   
            labels = labels[0,0,o1:-o2, 0] 
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
#------------------------------------------------------------------------------
# Test the model
def Validate(model, D_out, H_out, o1, o2):
    model.eval()
    accuracy = -1
    with torch.no_grad():
        correct = 0
        total = 0
        for img, labels in Validation_Loader:
            img = img.reshape(1, 1, Height, Width, Bands)
            labels = labels.reshape(1, 1, Height, Width)
            img = img.to(device)
            labels = labels.to(device=device, dtype=torch.int64)
            outputs = model(img, D_out, H_out)
            labels = labels[0,0,o1:-o2, 0] 
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        accuracy = 100 * correct / total
        print('\tVALIDATION Accuracy: {} %'.format(accuracy)) 
    return accuracy

def TEST(model, D_out, H_out, o1, o2):
    model.eval()
    accuracy = -1
    with torch.no_grad():
        correct = 0
        total = 0
        for img, labels in my_TEST_dataloader:
            img = img.reshape(1, 1, Height, Width, Bands)
            labels = labels.reshape(1, 1, Height, Width)
            img = img.to(device)
            labels = labels.to(device=device, dtype=torch.int64)
            outputs = model(img, D_out, H_out)
            labels = labels[0,0,o1:-o2, 0] 
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        accuracy = 100 * correct / total
        print('TEST Accuracy: {} %'.format(accuracy)) 
    return accuracy

'''GENETIC ALGORITHM'''
population = 10
generations = 10
threshold = 80.0

if name == 'IndianPines' or 'Samson' or "PaviaU" or "Salinas":
    kernel = [3, 5]
    _stride2 = [1]
    _stride3 = [0]
    l = 2
    s_stride = 2
    alpha = 65
else:
    kernel = [3, 5, 7]
    _stride2 = [1, 2]
    _stride3 = [1, 2]
    l = 4
    s_stride = 6
    alpha = 65
    
class GeneticAlgorithm():
    def __init__(self):
        self.num_epochs = np.random.randint(1, 20)
        self.output = random.choice([32])
        self.output2 = random.choice([16])
        self._stride = random.randint(1, s_stride)
        self._stride2 = random.choice(_stride2)
        self.N = random.randint(5, 20)
        self.N2 = random.randint(1, 5)      
        self.k = random.choice(kernel)
        self.k2 = random.choice([1])
        self._accuracy = 0

    def hyperparameters(self):  
        hyperparameters = {
            '_stride':self._stride,
            '_stride2':self._stride2,
            'N':self.N,
            'N2':self.N2,
            'k':self.k,
            'k2':self.k2,
            'output':self.output,
            'output2':self.output2,
            'num_epochs':self.num_epochs
            }
        return hyperparameters

def _networks(population):
    return [GeneticAlgorithm() for _ in range(population)]

def fitness(networks, verbose = False):           
    for network in networks:
        hyperparams = network.hyperparameters()
        if verbose:
            print('!!!!!FITNESS!!!!!')
#------------------------------------------------------------------------------        
        _stride = hyperparams['_stride'] 
        _stride2 = hyperparams['_stride2'] 
        if verbose:
            print('_stride: ',_stride)
#------------------------------------------------------------------------------        
        k = hyperparams['k']
        k2 = hyperparams['k2']
        if verbose:
            print('k: ',k)
#------------------------------------------------------------------------------        
        N = hyperparams['N']
        N2 = hyperparams['N2']
        if verbose:
            print('N: ',N)
#------------------------------------------------------------------------------
        kernel_size = (k, k, N)    
        kernel_size2 = (k2, k2, N2) 
        if verbose:
            print('kernel_size: ',kernel_size)
#------------------------------------------------------------------------------
#D_out = ((Weight + 2 * padding[0] - dilation[0] * (kernel_size[0] - 1) - 1)/stride[0]) + 1
        D_out1 = math.floor(((Width + 2 * 0 - 1 * (k - 1)-1)/_stride)+1)
        D_out = math.floor(((D_out1 + 2 * 0 - 1 * (k2 - 1)-1)/_stride2)+1)
#------------------------------------------------------------------------------        
#H_out = ((Weight + 2 * padding[1] - dilation[1] * (kernel_size[1] - 1) - 1)/stride[1]) + 1
        H_out1 = math.floor(((Height + 2 * 0 - 1 * (k - 1)-1)/_stride)+1)
        H_out = math.floor(((H_out1 + 2 * 0 - 1 * (k2 - 1)-1)/_stride2)+1)
#------------------------------------------------------------------------------        
#W_out = ((Weight + 2 * padding[2] - dilation[2] * (kernel_size[2] - 1) - 1)/stride[2]) + 1
        W_out1 = math.floor(((Bands + 2 * 0 - 1 * (N - 1)-1)/_stride)+1)
        W_out = math.floor(((W_out1 + 2 * 0 - 1 * (N2 - 1)-1)/_stride2)+1)
#------------------------------------------------------------------------------          
#Calculation just to match labels size to img in training                            
        if name == 'IndianPines' or 'Samson'or 'SalinasA':
            if D_out % 2 == 0:
                x = Height - D_out 
                o1 = math.floor(x/2)
                o2 = o1 + 1
            else:
                x = Height - D_out
                x1= math.floor(x/2)
                o1 = x1
                o2 = x1 
        elif name == 'PaviaU'or 'Botswana' or 'KSC' or 'Salinas':    
            if D_out % 2 == 0:
                x = Height - D_out
                o2 = math.floor(x/2)
                o1 = o2
            else:
                x = Height - D_out
                x1= math.floor(x/2)
                o1 = x1
                o2 = x1 + 1                
#------------------------------------------------------------------------------        
        output = hyperparams['output'] 
        output2 = hyperparams['output2']
        if verbose:
            print('output: ',output)               
#------------------------------------------------------------------------------        
        num_epochs = hyperparams['num_epochs']
        if verbose:
            print('num_epochs: ',num_epochs) 
            
        print('in features: ', D_out*H_out*output2*W_out,'out features: ', D_out*H_out)
#        if D_out*H_out < 151 and D_out*H_out > 13: KSC
        if D_out*H_out > alpha:
            try:
                model = C3D(output, _stride, kernel_size, D_out, H_out, W_out, output2, kernel_size2, _stride2).to(device)
                print(network.hyperparameters())
                MODEL(model, num_epochs, o1, o2, D_out, H_out)
                network._accuracy = TEST(model, D_out, H_out, o1, o2)
            except Exception as e:                          
                if 'out of memory' in str(e):
                    print('| WARNING: ran out of memory, retrying batch')
                    for p in model.parameters():
                        if p.grad is not None:
                            del p.grad  # free some memory
                            torch.cuda.empty_cache()          
                elif 'device-side assert triggered':
                    continue
                else:
                    network._accuracy = 0
                    print(e)
                    print(network.hyperparameters())
                    print('*************************************************')
    return networks

def selection(networks):
    networks = sorted(networks, key=lambda network: network._accuracy, reverse=True)
    number_of_parents = int(0.2 * len(networks))
    networks = networks[:max(number_of_parents, 8)]
    print('Długosc listy networks: ', len(networks))
    return networks

def crossover(networks):
    offspring = []
    param1 = 4
    param2 = 2
    for _ in range(int((population - len(networks)) / 2 )):
        parent1 = random.choice(networks)
        parent2 = random.choice(networks)
               
        child1 = GeneticAlgorithm()
        child2 = GeneticAlgorithm()
        
        child1.num_epochs = int(parent1.num_epochs/param1) + int(parent2.num_epochs/param2)
        child2.num_epochs = int(parent1.num_epochs/param2) + int(parent2.num_epochs/param1)
        
        child1._stride = int(math.ceil(parent1._stride/param1) + int(parent2._stride/param2))
        child2._stride = int(math.ceil(parent1._stride/param2) + int(parent2._stride/param1))
        
        child1._stride2 = int(math.ceil(parent1._stride2/param1) + int(parent2._stride2/param2))
        child2._stride2 = int(math.ceil(parent1._stride2/param2) + int(parent2._stride2/param1))

        child1.N = int(parent1.N/param1) + int(parent2.N/param2)
        child2.N = int(parent1.N/param2) + int(parent2.N/param1)
        
        child1.k = int(parent1.k/param1) + int(parent2.k/param2)
        child2.k = int(parent1.k/param2) + int(parent2.k/param1) 
            
        offspring.append(child1)
        offspring.append(child2)

    networks.extend(offspring)

    return networks

def mutate(networks):
    for network in networks:
        if np.random.uniform(0, 1) <= 0.01:           
            network.num_epochs += np.random.randint(1,10)
            network._stride += np.random.randint(1,2)
            network._stride2 += np.random.choice(_stride3)
            network.N += np.random.randint(1,l)
            network.k += np.random.choice([1, 2, 3])
    return networks

def main():
    
    networks = _networks(population)
    epsilon = 0.1
    old_accuracy = 0
    count = 0
    while count < 20:
        wyjscie1 = open('Precyzja_' + name + '.txt', 'a')     
        print("URUCHOMIENIE: ", count+1, file = wyjscie1)       
        for gen in range(generations):
            print('****Generation {}****'.format(gen+1))
            networks = fitness(networks)
            networks = selection(networks)
            networks = crossover(networks)
            networks = mutate(networks) 

            max_accuracy = -1
            
            for network in networks:
                
                wyjscie1 = open('Precyzja3_' + name + '.txt', 'a')
                wyjscie2 = open('Precyzja3_params_' + name + '.txt', 'a')                
                print(network._accuracy, file = wyjscie1)
                print(network._accuracy, file = wyjscie2)
                print(network.hyperparameters(), file = wyjscie2)
                
                if max_accuracy < network._accuracy:
                    max_accuracy = network._accuracy 

                    wyjscie1 = open('Precyzja4_' + name + '.txt', 'a')
                    wyjscie2 = open('Precyzja4_params_' + name + '.txt', 'a')     
                    print("Generacja: ", count+1, file = wyjscie1) 
                    print(max_accuracy, file = wyjscie1)
                    print(max_accuracy, file = wyjscie2)
                    print(network.hyperparameters(), file = wyjscie2)

            print(old_accuracy)
            print(max_accuracy)
            if  abs(max_accuracy - old_accuracy) < epsilon:
                print("Genetic algorithjm, finished")
                print("Accuracy doesnt changed in loop")
                print("Number of iterations ", gen+1)
                print("Max accuracy", max_accuracy)
                break
            old_accuracy = max_accuracy
        count += 1
              
if __name__ == '__main__':
        main()

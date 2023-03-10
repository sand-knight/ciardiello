import numpy as np
from algorithms import choi as ch, choe, brueser, pino
import matplotlib.pyplot as plt
from algorithms import heartrate as hr
from algorithms import segmenter as sg

data = np.genfromtxt('data/raw_2022-03-14_bcg.csv', delimiter=';')
data = data[14000:16000]
# data = np.genfromtxt('data/experiments/400.csv', delimiter=';')


indices=ch.choi(data,50.)
meanhr=hr.heartrate_from_indices(indices,50.)
print('choi', meanhr)

indices=choe.choe(data,50.)
meanhr=hr.heartrate_from_indices(indices,50.)
print('choe', meanhr)

indices=pino.pino(data,50.,24, levels=6)
meanhr=hr.heartrate_from_indices(indices,50.)
print('pino', meanhr)

print(brueser.brueser(data,50.))





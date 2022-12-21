# -*- coding: utf-8 -*-
"""

@author: eep5pt1 edited by Hafsa Maryam
"""

import pandas as pd
import numpy as np
from numpy import savetxt
from tempfile import TemporaryFile


reader = pd.read_csv ('output_abilene.csv')  # ('output_geant.csv') for Geant
print (reader)

x=np.array(reader)
y=x[:,1]
k=int(max(y))
z=x[:,2] # bit rate in kbps
rate=(z*10e-3)# divide by 10e-6 for geant -  bit rate in Gbps, multiplying by 100 to increase origianl bit-rates

index = np.where(y == float(1))
source_dat=np.empty((len(index[0]), k+1)) # each column corresponds to the bit-rates of a source node (bit-rates are sequential in time)


outfile = TemporaryFile()

for i in range(0, k+2): # range(1, k+1): for Geant
    index = np.where(y == i)
    for j in range(0, len(index[0])):
        source_dat[j,i-1] = float(rate[index[0][j]])

savetxt('source_data_abilene.csv', source_dat, delimiter=',')  # ('source_data_geant.csv') for Geant

np.save(outfile, source_dat)    
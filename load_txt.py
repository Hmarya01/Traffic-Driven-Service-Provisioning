# -*- coding: utf-8 -*-
"""
@author: eep5pt1, edited by Hafsa Maryam 

creates the abilene csv file

Note: this is specifically edited for Abilene network data and in the comments instructions are also given for the geant network as well.
"""
import os
import pandas as pd  
import numpy as np




path = 'data/abilene'  # data/geant for Geant

# For Abilene nodes
nodes_all=[ 'ATLAM5', 'ATLAng' , 'CHINng', 'DNVRng' , 'HSTNng', 'IPLSng', 'KSCYng', 'LOSAng' , 'NYCMng', 'SNVAng', 'STTLng', 'WASHng'] 

# For Geant nodes

# nodes_all= [ 'at1.at','be1.be' , 'ch1.ch', 'cz1.cz', 'de1.de', 'es1.es', 'fr1.fr', 'gr1.gr', 'hr1.hr', 'hu1.hu', 'ie1.ie' ,'il1.il', 'it1.it', 'lu1.lu','nl1.nl' ,'ny1.ny','pl1.pl' ,'pt1.pt','se1.se' ,'si1.si' , 'sk1.sk','uk1.uk' ]

nodes=np.array(nodes_all)
data_txt=os.listdir(path)

cols = ["source", "bit_rate"]
rows = []



for filename in data_txt:
     fullname = os.path.join(path, filename)
     with open(fullname) as f:
         lines = f.readlines()
         rates = np.zeros(len(nodes))
         for i in range(45,len(lines)):  # (55,len(lines)) for Geant
             arr=lines[i].split()
             if len(arr)==1:
                 break
             else:
                 source_node=arr[2]
                 source_index = np.where(nodes == source_node)
                 rates[source_index]=rates[source_index]+float(arr[6])
               
             
       
         for i in range(0,len(nodes)):
             rows.append({'source': i,
                     'bit_rate': rates[i]})      
         
         
     df = pd.DataFrame(rows, columns=cols)
    # Writing dataframe to csv
     df.to_csv('output_abilene.csv')   #('output_geant.csv')  for Geant  

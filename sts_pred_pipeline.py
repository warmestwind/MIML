from sts_read_h5 import read_h5 #class
#from sts_get_sample_set import get_sample_set #func
from sts_compute_suv import  suv_calcu #class
from sts_pandas_csv import csv_file #class
from sts_haar import haar_center, haar_lr
from sts_homo import homo_non
from sts_single_call import infer_sts

import tensorflow as tf
from tensorflow.contrib.tensor_forest.python import tensor_forest
import numpy as np
import csv


def sts_main():
    'this main func is used for construct data files (csv)'
    #read all h5data
    reader = read_h5('lab_petct_vox_5.00mm.h5')
    reader.get_patient_id()
    data_dict = reader.get_data_dict()

    #get non/labeled index sample list from every 
    # namedtuple('PatientData', ['ct', 'pet', 'label', 'valid'])
    single = data_dict['STS_031']
    pet = single.pet
    ct = single.ct
    features = []
    nan_index = []
    s_z, s_x, s_y = pet.shape 
    prediction = np.zeros(pet.shape)
    #s_z, s_x, s_y = 168,54,72
    print(s_z, s_x, s_y)
    #f = open("features.txt", "w")
    for z in range(s_z):
        for x in range(s_x):
            for y in range(s_y):
                temp = [] # save one voxel features
                #compute suv
                computer = suv_calcu(pet, (z,x,y))
                suvs = computer.get_result()
                temp.append(suvs['max'])
                temp.append(suvs['min'])
                temp.append(suvs['mean'])
                temp.append(suvs['std'])
                temp.append(suvs['self'])

                temp.append(haar_center(pet,(z,x,y),9))
                temp.append(homo_non(pet,(z,x,y),3))
                temp.append(homo_non(pet,(z,x,y),5))
                temp.append(ct[z,x,y])
                temp.append(haar_lr(pet,(z,x,y),10))
                temp.append(z)
                temp.append(x)
                temp.append(y)
           
                #print(str(temp), file = f)
                if np.nan in temp:
                    nan_index.append(z*s_x*s_y+x*s_y+ y)
                    
                else :
                    features.append(temp)
                    #tf.reset_default_graph()
                    #result,  _,  _ = infer_sts([temp])
                #prediction[z,x,y] = result
    
    result,  confidence,  _ = infer_sts(features)
    pre_num  = len(result)
    non_num = len(nan_index)
    print("num:",pre_num," ", non_num)
    assert pre_num+non_num == 1680000 ,'num error'
    print("done assert ")
    print(result.shape)
    print(confidence.shape)
    for i in nan_index:
       result = np.insert(result, i, 0)
       confidence = np.insert(confidence, i, 0)
    print("max value:",np.max(confidence))
    confidence = confidence * result
    
    #pred = np.array(result,dtype = np.int32).reshape((175,100,100))
    prob = np.array(confidence,dtype = np.float32).reshape((168,100,100))
    #np.save('031devn2.npy', pred)
    np.save('031testn3prob.npy', prob)
    #f.close()



if __name__ == '__main__' :
    print('start Pediction pipeline')
    sts_main()

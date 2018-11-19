import os

import numpy as np
import h5py
#import pandas
from collections import namedtuple

# read single patient h5 file

class read_h5:
    'return a dict which keys from STS_002/3/5/12/21/23/31 , values is a namedtuple()'
    def __init__(self, filepath):
        #'lab_petct_vox_5.00mm.h5'
        #protected , _ just a convention
        self._path = os.path.join('F:\MIML\segmenting-soft-tissue-sarcomas', filepath) 
        self.pdata = h5py.File(self._path, 'r')
        self.patient_id = []
        self.data_dict = {}

    def get_patient_id(self):
        for id in self.pdata['ct_data']: # ['ct_data'] ['pet_data'] ['label_data']
            self.patient_id.append(id)
        return self.patient_id

    def get_data_dict(self):
        pdata = namedtuple('PatientData', ['ct', 'pet', 'label', 'valid'])
        #print(len(self.patient_id))
        for id in self.patient_id:
            try:
                ct_data = self.pdata['ct_data'][id]
                pet_data = self.pdata['pet_data'][id]
                label_data = self.pdata['label_data'][id]
                #print('success')
            except KeyError as ke:
                #print('false')
                single = pdata(None, None, None, False)
            single = pdata(np.array(ct_data), np.array(pet_data), np.array(label_data), True)
            self.data_dict[id] = single
        return self.data_dict

if __name__ == '__main__':
    print('only test read_h5 class')
    reader = read_h5('lab_petct_vox_5.00mm.h5')
    reader.get_patient_id()
    data_dict = reader.get_data_dict()
    single = data_dict['STS_003']
    print(single.pet.shape)
    # protected , _ just a convention ,all python class memember is public
    reader._path = 'lalala'
    print(reader._path)

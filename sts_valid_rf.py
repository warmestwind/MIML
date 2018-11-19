import csv
import numpy as np
#import matplotlib.pyplot as plt
from sts_rf_valid_model import random_forest_sts
import tensorflow as tf
import pandas as pd 

#5维向量：[max min mean std self]+ 1维[haar_center]
def load_series(filename):
    data = []
    label = []
    try:
        with open(filename) as csvfile:
            csvreader = csv.reader(csvfile)
            #data = [row[series_idx] for row in csvreader if len(row) > 0]
            num = 0
            for row in csvreader:
                if num== 0: 
                    #print(row[6])
                    num+=1 
                    continue
                else:
                    label.append(row[5])
                    row.pop(5) #label
                    row.pop(5) #coord
                    #row.pop(5) #homo3
                    #row.pop(5) #homo5
                    #pop(8) #ct
                    data.append(row)                   
                    num+=1
            #normalized_data = (data - np.mean(data)) / np.std(data)
        #return normalized_data
        return data, num-1, label
    except IOError:
        return None

def load_series_pd(filename):
    data = []
    label = []
    try:
        df = pd.read_csv(filename)
        num = 0
        for i in range(df.shape[0]):
            if num==0: 
                #print(row[6])
                row = df.loc[i]
                print(row)
                print(row[0])
                num+=1 
                break             
            else:
                label.append(row[5])
                row.pop(5) #label
                row.pop(5) #coord
                #row.pop(5) #homo3
                #row.pop(5) #homo5
                #row.pop(8) #ct
                data.append(row)                   
                num+=1
        #normalized_data = (data - np.mean(data)) / np.std(data)
        #return normalized_data
        return data, num-1, label
    except IOError:
        return None

    
def split_data(data, percent_train=0.80):
    num_rows = len(data)
    train_data, test_data = [], []
    for idx, row in enumerate(data):
        if idx < num_rows * percent_train:
            train_data.append(row)
        else:
            test_data.append(row)
    return train_data, test_data

def main():
    #dataset, data_size, label = load_series("concate_data_ct/new_test_set.csv")
    dataset, data_size, label = load_series("dataset/1023/new_dev_set.csv")
    print(dataset[13])
    #dataset, data_size, label = load_series("concat_023.csv")
    #dataset, data_size, label = load_series("012_lab_filtered.csv")
    print(data_size)           
    print((dataset[1])) # 0 1 ...175436
    print((label[1]))

    #003
 #   valid_set = dataset#[0:2488]
 #   valid_label = label#[0:2488]  

    #002
 #  valid_set = dataset#[0:967]
 #   valid_label = label#[0:967] 

    #005
 #   valid_set = dataset[0:9886]
 #   valid_label = label[0:9886] 
   
    #031
 #   valid_set = dataset[0:8098]
 #   valid_label = label[0:8098]

    #012
#    valid_set = dataset#[0:1724]
#    valid_label = label#[0:1724]
    
    #021      
 #   valid_set = dataset[0:16300]
 #   valid_label = label[0:16300]

    # test_set 031:
 #   valid_set = dataset#[0:8098] #(row_p_no. -1)*2
 #   valid_label = label#[0:8098]

    # dev_set 023 
    valid_set = dataset#[0:6050] #(row_p_no. -1)*2
    valid_label = label#[0:6050]

  
    sess=tf.Session()

    random_forest_sts(valid_set, valid_label)

if __name__ == '__main__':
    print("start valid random forest")
    main()

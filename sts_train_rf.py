import csv
import numpy as np
import matplotlib.pyplot as plt
from sts_rf_model_cv import random_forest_sts
import tensorflow as tf

#5维向量：[max min mean std self]+ 1维[haar_center]+[homo3]
#max,min,mean,std,self,class,coord,haar_center,homo_3,homo_5
def load_series(filename):
    data = []
    label = []
    try:
        with open(filename) as csvfile:
            csvreader = csv.reader(csvfile)
            #data = [row[series_idx] for row in csvreader if len(row) > 0]
            num = 0
            for row in csvreader:
                if num==0: 
                    #print(row[6])
                    print(type(row))
                    num+=1 
                    continue
                else:
                    label.append(row[5])
                    row.pop(5) #class
                    row.pop(5) #coord
                    #row.pop(6)
                    #row.pop(6)
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
    #dataset, data_size, label = load_series("sorted2_3.csv")
    #dataset, data_size, label = load_series("concate_data_ct/new_train_set.csv")
    dataset, data_size, label = load_series("dataset/1023/new_train_set.csv")
    print(data_size)
    print((dataset[1])) # 0 1 ...175436
    print((label[1]))
 #   train_set, test_set = split_data(dataset)
 #   train_label, test_label = split_data(label)
 #   print(len(train_set))
 #   print(len(test_set))
    train_set = dataset#[0:3456] #keep number of  pos and neg sample balance
    train_label = label#[0:3456]

    #train_set = tf.convert_to_tensor(train_set)
    #test_set = tf.convert_to_tensor(test_set)
    #train_label = tf.convert_to_tensor(train_label)
    #test_label = tf.convert_to_tensor(test_label)

    #sess=tf.Session()
    #print(sess.run(tf.shape(train_set)))
    #print(sess.run(tf.shape(test_set)))      
    #random_forest_sts(train_set, train_label, test_set, test_label)
    #random_forest_sts(train_set[:37414], train_label[:37414], None, None)
    random_forest_sts(train_set, train_label, None, None)
    
if __name__ == '__main__':
    print("start train random forest")
    main()

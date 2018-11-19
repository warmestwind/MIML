#reference: https://github.com/tensorflow/tensorflow/tree/r1.4/tensorflow/contrib/tensor_forest
import tensorflow as tf
from tensorflow.contrib.tensor_forest.python import tensor_forest
import numpy as np

def infer_sts(infer_x):
    if np.nan in infer_x:
        return 0, _ , _
    # Parameters
    #num_steps = 200 # Total steps to train
    num_classes = 2 
    num_features = 13 
    num_trees = 20
    max_nodes = 200

    # Random Forest Parameters
    # fill():intelligently sets any non-specific parameters
    hparams = tensor_forest.ForestHParams(num_classes=num_classes,
                                        num_features=num_features,
                                        regression=False,
                                        num_trees=num_trees,
                                        max_nodes=max_nodes).fill()
    
    #Input and Target data
    X = tf.placeholder(tf.float32, shape = [None, num_features])

    #Build the Random Forest
    forest_graph = tensor_forest.RandomForestGraphs(hparams)

    #Compute inference result
    infer_op = forest_graph.inference_graph(X)
    infer_result = tf.argmax(infer_op, 1)
    #max confidence
    infer_op = tf.reduce_max(infer_op, axis= 1)
    feature_importance= forest_graph.feature_importances()

    # Initialize the variables (i.e. assign their default value)
    init_vars = tf.global_variables_initializer()

    # Start TensorFlow session
    sess = tf.Session()

    # Run the initializer
    sess.run(init_vars)

    model_path ="checkpoint_merge/variable"
    saver = tf.train.Saver()
    saver.restore(sess, tf.train.latest_checkpoint("checkpoint_merge"))

    result, confidence ,importances= sess.run([infer_result,infer_op,feature_importance], feed_dict= {X: infer_x})
    '''
    if result==0 : return "这个点不是病灶~！"
    else: return "这个点是病灶~！"
    '''
    
    '''
    #check graph weight
    variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
    doc = open("variables.txt","w")
    variables_name= [v.name for v in variables]
    doc.write("".join(variables_name))
    doc.close()
    '''
    #print(variables)
    return result, confidence, importances

if __name__ == '__main__':
     
    #input_pos=[1.42, 1.18, 1.08,0.1,1.06,-112,0.89,0.85] #1  
    #input_pos= [1.42,1.06,0.99,0.18,1.06,-93,0.84,0.85]
    #input_pos=[1.17,0.91,0.84,0.2,0.95,388.09,0.8,0.55] #fp 149 54 67
    #input_pos=[4.28, 2.57, 3.14, 0.45, 3.19, 62.5, 0.66, 0.5]
 #   input_pos=[1.087721824645996,0.9064348936080933,0.9935045705901252, \
 #   0.04513943444818001,0.9663764834403992,29.32172393798828,0.9490662166020377,0.9085662612326831]
 #   input_neg=[0,0,0,0,0,0.0014,1,1] #0 

 #   input_list=[input_pos, input_neg]
    # dev:
    # FP 边缘外部，0.45 ：0.55， 方法，卡阈值或增加负例
    # input_list = [[1.03,0.51,0.75,0.14,0.82,-106.44,0.86,0.83,7.54,-137.24]]
    # FP 标记时缺少 0.008: 0.992
    #input_list = [[5.16, 0.59,2.05,1.30,1.92,-85.91,0.42,0.35,51.83,562.07,10,20,30]]
    # input_list = [[0.71,0.34,0.47,0.09,0.34,291.59,0.85,0.74,17.37,335.75]]
    # FN 边缘内部 0.54:0.46
    #input_list = [[0.92,0.62,0.78,0.09,0.78,-137.602,0.92,0.89,25.95,23.82]]
    #input_list = [[0.94,0.42,0.68,0.12,0.78,-4.18,0.9,0.81,-69.26,219.88,20,30,50]]

    print("input info:", input_list)
    results, confidences, importance = infer_sts(input_list)
    print("result=",results)
    print(confidences.shape)
    print("confidence=",confidences)
    print("importance=",importance)

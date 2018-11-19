#reference: https://github.com/tensorflow/tensorflow/tree/r1.4/tensorflow/contrib/tensor_forest
import tensorflow as tf
from tensorflow.contrib.tensor_forest.python import tensor_forest

def random_forest_sts(batch_x, batch_y):

    # standard method for import MNIST data 
    #from tensorflow.examples.tutorials.mnist import input_data
    #mnist = input_data.read_data_sets("./data", one_hot=False)

    # Parameters
    num_steps = 200 # Total steps to train
    num_classes = 2   # non 0  labeled 1
    num_features = 13  #max min mean std self  
    num_trees = 20
    max_nodes = 200

    # Random Forest Parameters
    # fill():intelligently sets any non-specific parameters
    hparams = tensor_forest.ForestHParams(num_classes=num_classes,
                                        num_features=num_features,
                                        regression=False,
                                        num_trees=num_trees,
                                        max_nodes=max_nodes).fill()


    # Input and Target data
    X = tf.placeholder(tf.float32, shape=[None, num_features])
    # For random forest, labels must be integers (the class id)
    #shape(Y)=[None] because not it's one_hot label
    Y = tf.placeholder(tf.int32, shape=[None])   


    # Build the Random Forest
    forest_graph = tensor_forest.RandomForestGraphs(hparams)
    # Get training graph and loss
    train_op = forest_graph.training_graph(X, Y)
    loss_op = forest_graph.training_loss(X, Y)

    # Measure the accuracy
    infer_op = forest_graph.inference_graph(X)
    correct_prediction = tf.equal(tf.argmax(infer_op, 1), tf.cast(Y, tf.int64))
    #because X is totol instances ,accuracy is average of all instance
    accuracy_op = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    #use tf.metrics
    acc, acc_op = tf.metrics.accuracy(labels=Y, predictions=tf.argmax(infer_op, 1))
    pre, pre_op = tf.metrics.precision(labels=Y, predictions=tf.argmax(infer_op, 1))
    rec, rec_op = tf.metrics.recall(labels=Y, predictions=tf.argmax(infer_op, 1))

    # Initialize the variables (i.e. assign their default value)
    init_vars = [tf.global_variables_initializer(),tf.local_variables_initializer()]
    # Start TensorFlow session
    sess = tf.Session()

    # Run the initializer
    sess.run(init_vars)

    #model_path = "checkpoint_/variable"
    #model_path = "checkpoint_merge/variable"
    saver = tf.train.Saver()
    #saver.restore(sess, tf.train.latest_checkpoint("checkpoint_"))#for homo3
    saver.restore(sess, tf.train.latest_checkpoint("checkpoint_merge")) 
    #saver.restore(sess, tf.train.latest_checkpoint("checkpoint")) #for no homo
    #Test Model
    print("Validation Accuracy:", sess.run(accuracy_op, feed_dict={X: batch_x, Y: batch_y}))
    print("Validation Accuracy:", sess.run([acc,acc_op], feed_dict={X: batch_x, Y: batch_y}))
    _, p = sess.run([pre,pre_op], feed_dict={X: batch_x, Y: batch_y})
    print("Validation Precision:",p)
    _, r = sess.run([rec,rec_op], feed_dict={X: batch_x, Y: batch_y})
    print("Validation Recall:", r)
    print("Validation F1 score:", 2*p*r/(p+r))
    #预测结果比较
    prediction_list = sess.run(correct_prediction, feed_dict={X: batch_x, Y: batch_y})
    #print("prediction: ", prediction_list)

    FN = [ i for i in range(0,len(prediction_list)) \
            if (batch_y[i]=='1' and  prediction_list[i]==False)]
    FP = [ i for i in range(0,len(prediction_list)) \
            if (batch_y[i]=='0' and  prediction_list[i]==False)]            
    print(batch_x[13])
    print(len(FP))
    print(len(FN))
    f = open("pre_list.txt","w")
    for id in FP:
        print(str(id),file = f)
    print("\n",file = f)
    for id in FN:
        print(str(id),file = f)
    f.close()



    '''
    prediction = sess.run(correct_prediction, feed_dict={X: batch_x, Y: batch_y})
    print(len(prediction))
    index = []
    for i in range(len(prediction)):
        if( prediction[i]==False):
            index.append(i)
    print(len(index))
    print(index[1])

    doc = open("index.txt","w")
    doc.write(str(index))
    doc.close()
    print("Prediction:", prediction)
    '''

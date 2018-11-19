#reference: https://github.com/tensorflow/tensorflow/tree/r1.4/tensorflow/contrib/tensor_forest
import tensorflow as tf
from tensorflow.contrib.tensor_forest.python import tensor_forest
from sklearn.model_selection import KFold, StratifiedKFold
import numpy as np

def random_forest_sts(batch_x, batch_y, test_x, test_y):
    with tf.device('/cpu:0'):
        # standard method for import MNIST data 
        #from tensorflow.examples.tutorials.mnist import input_data
        #mnist = input_data.read_data_sets("./data", one_hot=False)

        # Parameters
        num_epochs = 200 # Total epochs  to train
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
        
        # input weights
        #weights = [1]*18707+[0.7]*(103640-18707)
        weights = [1]*15683+[0.1]*709329
        print("weight len:", len(weights))
        # Get training graph and loss
        train_op = forest_graph.training_graph(X, Y, input_weights= tf.constant(weights))
        loss_op = forest_graph.training_loss(X, Y)

        # Measure the accuracy
        infer_op = forest_graph.inference_graph(X)
        correct_prediction = tf.equal(tf.argmax(infer_op, 1), tf.cast(Y, tf.int64))
        #because X is totol instances ,accuracy is average of all instance
        accuracy_op = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        tf.summary.scalar('Accuracy', accuracy_op)
        
        #feature importances
        feature_importances =forest_graph.feature_importances()
        
        # Initialize the variables (i.e. assign their default value)
        init_vars = tf.global_variables_initializer()
        merged = tf.summary.merge_all()

    # Start TensorFlow session
    sess = tf.Session()
    #sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))
    writer = tf.summary.FileWriter('./graph',sess.graph)

    # Run the initializer
    sess.run(init_vars)

    #model_path = "checkpoint/variable"
    model_path = "checkpoint_merge/variable"
    saver = tf.train.Saver()

    def cross_validate(session, split_size=5):
        results = []
        #kf = KFold(n_splits=split_size, shuffle=True)
        kf = StratifiedKFold(n_splits=split_size, shuffle=True)
        for train_idx, val_idx in kf.split(batch_x, batch_y):
            #print("type of train_idx~!!!!!!",(train_idx[0]))
            #print("type of batch_x~!!!!!!",type(batch_x))
            #return 0
            train_x = np.array(batch_x)[train_idx]
            train_y = np.array(batch_y)[train_idx]
            val_x = np.array(batch_x)[val_idx]
            val_y = np.array(batch_y)[val_idx]
            print("Strat a new fold training...")
            run_train(session, train_x, train_y)
            results.append(session.run(accuracy_op, feed_dict={X: val_x, Y: val_y}))
        return results

    def run_train(sess, train_x, train_y):
        # Training
        for i in range(1, num_epochs  + 1): #[1,201), no batch
        # Prepare Data
        # Get the next batch of MNIST data (only images are needed, not labels)
        #batch_x, batch_y = mnist.train.next_batch(batch_size)
            _, l = sess.run([train_op, loss_op], feed_dict={X: train_x, Y: train_y})
            if i % 50 == 0 or i == 1:
                summary, acc = sess.run([merged,accuracy_op], feed_dict={X: batch_x, Y: batch_y})
                print('Epoch %i, Loss: %f, Acc: %f' % (i, l, acc))
                writer.add_summary(summary,i)

        importances = sess.run(feature_importances, feed_dict={X: batch_x, Y: batch_y})
        print("impotances of feature= ",importances)
        save_path = saver.save(sess, model_path)

 #   result = cross_validate(sess)
 #   print("Cross-validation result: %s" % result)
 #   print("Mean of Cross-validation result: %s" % np.mean(result))
    run_train(sess, batch_x, batch_y)

    writer.close()
    sess.close()
    # Test Model
    #test_x, test_y = mnist.test.images, mnist.test.labels
    #print("Test Accuracy:", sess.run(accuracy_op, feed_dict={X: test_x, Y: test_y}))


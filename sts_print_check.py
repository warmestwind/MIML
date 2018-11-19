from tensorflow.python.tools import inspect_checkpoint as chkp
from tensorflow.python import pywrap_tensorflow

chkp.print_tensors_in_checkpoint_file('checkpoint_merge/variable',
                tensor_name='', all_tensors = True)


reader = pywrap_tensorflow.NewCheckpointReader('checkpoint_merge/variable')
var_to_shape_map = reader.get_variable_to_shape_map()
for key in sorted(var_to_shape_map):
    print("tensor_name: ",key)
    #print(reader.get_tensor(key))          
    print("f0=====: ", reader.get_tensor('candidate_split_features-0'))

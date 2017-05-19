import tensorflow as tf
import numpy as np

# mean is already subtracted from start 
VGG_MEAN = [103.939, 116.779, 123.68]

class vgg16(object):
    def __init__(self, x, keep_prob, num_classes, skip_layer, weights_path = 'DEFAULT'):
        self.rgb = x
        self.NUM_CLASSES = num_classes
        self.KEEP_PROB = keep_prob
        self.SKIP_LAYER = skip_layer
        
        if weights_path == 'DEFAULT':      
            self.WEIGHTS_PATH = 'vgg16.npy'
        else:
            self.WEIGHTS_PATH = weights_path

        
    def create(self):
        
        rgb_scaled = self.rgb

        # Convert RGB to BGR
        red, green, blue = tf.split(split_dim = 3, num_split = 3, value = rgb_scaled)
        assert red.get_shape().as_list()[1:] == [224, 224, 1]
        assert green.get_shape().as_list()[1:] == [224, 224, 1]
        assert blue.get_shape().as_list()[1:] == [224, 224, 1]
        bgr = tf.concat(3, [
            blue, green, red
        ])
        assert bgr.get_shape().as_list()[1:] == [224, 224, 3]



        self.relu1_1 = conv_layer(bgr, 3, 3, 64, "conv1_1")
        self.relu1_2 = conv_layer(self.relu1_1, 3, 3, 64, "conv1_2")
        self.pool1 = max_pool(self.relu1_2, 'pool1')

        self.relu2_1 = conv_layer(self.pool1, 3, 3, 128, "conv2_1")
        self.relu2_2 = conv_layer(self.relu2_1, 3, 3, 128, "conv2_2")
        self.pool2 = max_pool(self.relu2_2, 'pool2')

        self.relu3_1 = conv_layer(self.pool2, 3, 3, 256, "conv3_1")
        self.relu3_2 = conv_layer(self.relu3_1, 3, 3, 256, "conv3_2")
        self.relu3_3 = conv_layer(self.relu3_2, 3, 3, 256, "conv3_3")
        self.pool3 = max_pool(self.relu3_3, 'pool3')

        self.relu4_1 = conv_layer(self.pool3, 3, 3, 512, "conv4_1")
        self.relu4_2 = conv_layer(self.relu4_1, 3, 3, 512, "conv4_2")
        self.relu4_3 = conv_layer(self.relu4_2, 3, 3, 512, "conv4_3")
        self.pool4 = max_pool(self.relu4_3, 'pool4')

        self.relu5_1 = conv_layer(self.pool4, 3, 3, 512, "conv5_1")
        self.relu5_2 = conv_layer(self.relu5_1, 3, 3, 512, "conv5_2")
        self.relu5_3 = conv_layer(self.relu5_2, 3, 3, 512, "conv5_3")
        self.pool5 = max_pool(self.relu5_3, 'pool5')

        self.fc6 = fc_layer(self.pool5, 7*7*512, 4096, "fc6")
        assert self.fc6.get_shape().as_list()[1:] == [4096]

        self.relu6 = tf.nn.relu(self.fc6)
        #if train:
            #self.relu6 = tf.nn.dropout(self.relu6, 0.5)

        self.fc7 = fc_layer(self.relu6, 4096, 4096, "fc7")
        self.relu7 = tf.nn.relu(self.fc7)
        #if train:
            #self.relu7 = tf.nn.dropout(self.relu7, 0.5)

        self.fc8 = fc_layer(self.relu7, 4096, self.NUM_CLASSES, "fc8")
        self.prob = tf.nn.softmax(self.fc8, name="prob")

    def load_initial_weights(self, session):
        """
        As the weights from http://www.cs.toronto.edu/~guerzhoy/tf_alexnet/ come 
        as a dict of lists (e.g. weights['conv1'] is a list) and not as dict of 
        dicts (e.g. weights['conv1'] is a dict with keys 'weights' & 'biases') we
        need a special load function
        """
        
        # Load the weights into memory
        weights_dict = np.load(self.WEIGHTS_PATH, encoding = 'bytes').item()
        
        # Loop over all layer names stored in the weights dict
        for op_name in weights_dict:
            
            # Check if the layer is one of the layers that should be reinitialized
            if op_name not in self.SKIP_LAYER:
            
                with tf.variable_scope(op_name, reuse = True):
                
                    # Loop over list of weights/biases and assign them to their corresponding tf variable
                    for data in weights_dict[op_name]:
                    
                        # Biases
                        if len(data.shape) == 1:
                      
                            var = tf.get_variable('biases', trainable = False)
                            session.run(var.assign(data))
                      
                        # Weights
                        else:
                      
                            var = tf.get_variable('weights', trainable = False)
                            session.run(var.assign(data))
                    

"""
Predefine all necessary layer for the VGG
""" 
def conv_layer(x, filter_height, filter_width, num_filters, name,
         padding='SAME'):
    """
    Adapted from: https://github.com/ethereon/caffe-tensorflow
    """
    # Get number of input channels
    input_channels = int(x.get_shape()[-1])
  
    with tf.variable_scope(name) as scope:
        # Create tf variables for the weights and biases of the conv layer
        weights = tf.get_variable('weights', shape = [filter_height, filter_width, input_channels, num_filters])
        biases = tf.get_variable('biases', shape = [num_filters])  
    
        conv = tf.nn.conv2d(x, weights, strides = [1, 1, 1, 1], padding = padding)

        # Add biases 
        bias = tf.reshape(tf.nn.bias_add(conv, biases), conv.get_shape().as_list())
    
        # Apply relu function
        relu = tf.nn.relu(bias, name = scope.name)
            
        return relu
  
def fc_layer(x, num_in, num_out, name):
    with tf.variable_scope(name) as scope:

        shape = x.get_shape().as_list()
        dim = 1
        for d in shape[1:]:
            dim *= d
        x = tf.reshape(x, [-1, dim])
    
        # Create tf variables for the weights and biases
        weights = tf.get_variable('weights', shape=[num_in, num_out], trainable=True)
        biases = tf.get_variable('biases', [num_out], trainable=True)
        
        # Matrix multiply weights and inputs and add bias
        act = tf.nn.xw_plus_b(x, weights, biases, name=scope.name)
        
        return act
    

def max_pool(x, name):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                        strides = [1, 2, 2, 1],
                        padding = 'SAME', name = name)

def avg_pool(x, name):
    return tf.nn.avg_pool(x, ksize=[1, 2, 2, 1], strides = [1, 2, 2, 1], padding = 'SAME', name = name)
  
#def dropout(self, x, keep_prob):
    #return tf.nn.dropout(x, keep_prob)
  

# Input should be an rgb image [batch, height, width, 3]
# values scaled [0, 1]


import numpy as np
from keras import backend as K
from keras.regularizers import Regularizer
from keras.layers import Reshape
import tensorflow as tf

class SoftRankRegularizer(Regularizer):
    """This class implements the SoftRank regularizer.
    
    Uses TF compatible syntax from now-deprecated EigenValueDecay regularizer in keras: https://github.com/keras-team/keras/commit/1c630c3e3c8969b40a47d07b9f2edda50ec69720
    
    Args:
        The constant that controls the regularization on the current layer

    Returns:
        The regularized loss (for the training data) and
        the original loss (for the validation data).
    """
    def __init__(self, k):
        self.k = k

    def __call__(self, W):
        power = 9  # number of iterations of the power method

        # Reshape W to 2D, combining 3 smallest dims
        print(W.shape)
        W_shape_sort = sorted(W.shape)
        print(W_shape_sort)
        W_rshp = tf.reshape(W, (W_shape_sort[0]*W_shape_sort[1]*W_shape_sort[2],W_shape_sort[3]))
        print(W_rshp.shape)

        WW = K.dot(K.transpose(W_rshp), W_rshp)
        print(WW.shape)
        dim1, dim2 = K.eval(K.shape(WW))
        k = self.k
        o = K.ones([dim1, 1])
        print(o.shape)

        # Power method for approximating the dominant eigenvector:
        domin_eigenvect = K.dot(WW, o)
        for n in range(power - 1):
            domin_eigenvect = K.dot(WW, domin_eigenvect)    
        
        WWd = K.dot(WW, domin_eigenvect)
        domin_eigenval = K.dot(K.transpose(WWd), domin_eigenvect) / K.dot(K.transpose(domin_eigenvect), domin_eigenvect)  # the corresponding dominant eigenvalue
        
        # Compute variance
        variance = tf.reduce_sum(tf.square(WW), keepdims=True) / tf.size(WW, out_type=tf.float32)

        regularization = (variance/domin_eigenval) * self.k 
        return K.sum(regularization)
    
    def get_config(self):
        return {"name": self.__class__.__name__,
                "k": self.k}
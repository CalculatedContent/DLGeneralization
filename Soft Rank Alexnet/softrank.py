import numpy as np
from keras import backend as K
from keras.regularizers import Regularizer

class SoftRankRegularizer(Regularizer):
    """This class implements the SoftRank regularizer.
    
    Args:
        The constant that controls the regularization on the current layer
    Returns:
        The regularized loss (for the training data) and
        the original loss (for the validation data).
    """
    def __init__(self, k):
        self.k = k


    def set_param(self, p):
        self.p = p


    def __call__(self, x):
        power = 9  # number of iterations of the power method
        W = x

        # import pdb
        # pdb.set_trace()
        
        # Reshape W to 2D, combining 3 smallest dims
        W_shape_sort = sorted(W.shape)
        W_rshp = tf.reshape(W, (W_shape_sort[0]*W_shape_sort[1]*W_shape_sort[2],W.shape_sort[3]))

        WW = K.dot(K.transpose(W_rshp), W_rshp)
        dim1, dim2 = K.eval(K.shape(WW))
        k = self.k
        # o = np.ones(dim1)  # initial values for the dominant eigenvector
        o = K.ones([dim1, 1])

        # power method for approximating the dominant eigenvector:
        domin_eigenvect = K.dot(WW, o)
        for n in range(power - 1):
            domin_eigenvect = K.dot(WW, domin_eigenvect)    
        
        WWd = K.dot(WW, domin_eigenvect)
        domin_eigenval = K.dot(WWd, domin_eigenvect) / K.dot(domin_eigenvect, domin_eigenvect)  # the corresponding dominant eigenvalue
        
        # Variance
        variance, _ = np.linalg.eig(WW)

        # regularized_loss = loss + (variance/domin_eigenval) * self.k
        regularization = (variance/domin_eigenval) * self.k 
        return K.sum(regularization)
        # return K.in_train_phase(regularized_loss, loss)
    

    def get_config(self):
        return {"name": self.__class__.__name__,
                "k": self.k}
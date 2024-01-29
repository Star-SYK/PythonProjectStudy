import numpy as np
import tensorflow as tf
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
print(tf.config.list_physical_devices('GPU'))
tf.config.run_functions_eagerly(True)
def tensorflow_demo():
    a_t = tf.constant(1)
    b_t = tf.constant(2)
    c_t = a_t + b_t
    print(c_t.numpy())
    return None

def matrix_calculate():
    """
    矩阵的计算
    :return:
    """
    return None


def tensorflow_using_demo():
   numpy_to_tensor =  tf.convert_to_tensor(np.ones([2,3]))

   aa = tf.constant([[1,2],[3,0]])
   print(numpy_to_tensor)
   print(aa)

if __name__ == "__main__":
    # tensorflow_demo()
    tensorflow_using_demo()

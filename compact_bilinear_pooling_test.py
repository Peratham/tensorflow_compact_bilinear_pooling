from compact_bilinear_pooling import compact_bilinear_pooling_layer

import tensorflow as tf
import numpy as np

def _bilinear_pooling(bottom1, bottom2):
    bottom1 = np.reshape((-1, bottom1.shape[-1]))
    bottom2 = np.reshape((-1, bottom2.shape[-1]))

batch_size = 100
height = 1
width = 1
input_dim = 2048
output_dim = 16000

bottom1 = tf.convert_to_tensor(np.random.randn(batch_size, height, width,
    input_dim), dtype=tf.float32)
bottom2 = tf.convert_to_tensor(np.random.randn(batch_size, height, width,
    input_dim), dtype=tf.float32)
top = compact_bilinear_pooling_layer(bottom1, bottom2, output_dim)

sess = tf.InteractiveSession()
cbp = sess.run(top)
sess.close()

print(cbp.shape)

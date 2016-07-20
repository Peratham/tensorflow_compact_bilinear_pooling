from __future__ import absolute_import, division, print_function

import os.path as osp

import tensorflow as tf
from tensorflow.python.framework import ops

# load module
module = tf.load_op_library(osp.join(osp.dirname(__file__),
                                     'build/sequential_batch_fft.so'))

sequential_batch_fft = module.sequential_batch_fft
sequential_batch_ifft = module.sequential_batch_ifft

@tf.RegisterShape("SequentialBatchFFT")
def _SequentialBatchFFTShape(op):
    return [op.inputs[0].get_shape()]

@tf.RegisterShape("SequentialBatchIFFT")
def _SequentialBatchIFFTShape(op):
    return [op.inputs[0].get_shape()]

@ops.RegisterGradient("SequentialBatchFFT")
def _SequentialBatchFFTGrad(_, grad):
    size = tf.cast(grad.shape[1], dtypes.float32)
    return sequential_batch_ifft(grad) * math_ops.complex(size, 0.)

@ops.RegisterGradient("SequentialBatchIFFT")
def _SequentialBatchIFFTGrad(_, grad):
    rsize = 1. / tf.cast(grad.shape[1], dtypes.float32)
    return sequential_batch_fft(grad) * math_ops.complex(rsize, 0.)

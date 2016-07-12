from __future__ import absolute_import, division, print_function

import numpy as np
import tensorflow as tf

def _generate_sketch_matrix(rand_h, rand_s, output_dim):
    """
    Return a sparse matrix used for tensor sketch operation in compact bilinear
    pooling

    Args:
        rand_h: an 1D numpy array containing indices in interval `[0, output_dim)`.
        rand_s: an 1D numpy array of 1 and -1, having the same shape as `rand_h`.
        output_dim: the output dimensions of compact bilinear pooling.

    Returns:
        a sparse matrix of `[input_dim, output_dim]` for tensor sketch.
    """

    # Generate a sparse matrix for tensor count sketch
    rand_h = rand_h.astype(np.int64)
    rand_s = rand_s.astype(np.float32)
    assert(rand_h.ndim==1 and rand_s.ndim==1 and len(rand_h)==len(rand_s))
    assert(np.all(rand_h >= 0) and np.all(rand_h < output_dim))

    input_dim = len(rand_h)
    indices = np.concatenate((np.arange(input_dim)[..., np.newaxis],
                              rand_h[..., np.newaxis]), axis=1)
    sparse_sketch_matrix = tf.sparse_reorder(
        tf.SparseTensor(indices, rand_s, [input_dim, output_dim]))
    return sparse_sketch_matrix

def compact_bilinear_pooling_layer(bottom1, bottom2, output_dim,
    rand_h_1=None, rand_s_1=None, rand_h_2=None, rand_s_2=None,
    seed_h_1=1, seed_s_1=3, seed_h_2=5, seed_s_2=7):
    """
    Compute compact bilinear pooling over two bottom inputs. Reference:

    Yang Gao, et al. "Compact Bilinear Pooling." in Proceedings of IEEE
    Conference on Computer Vision and Pattern Recognition (2016).
    Akira Fukui, et al. "Multimodal Compact Bilinear Pooling for Visual Question
    Answering and Visual Grounding." arXiv preprint arXiv:1606.01847 (2016).

    Args:
        bottom1: 1st input, a 2D Tensor of shape [batch_size, input_dim1].
        bottom2: 2nd input, a 2D Tensor of shape [batch_size, input_dim2].

        output_dim: output dimension for compact bilinear pooling.

        rand_h_1: (Optional) an 1D numpy array containing indices in interval
                  `[0, output_dim)`. Automatically generated from `seed_h_1`
                  if is None.
        rand_s_1: (Optional) an 1D numpy array of 1 and -1, having the same shape
                  as `rand_h_1`. Automatically generated from `seed_s_1` if is
                  None.
        rand_h_2: (Optional) an 1D numpy array containing indices in interval
                  `[0, output_dim)`. Automatically generated from `seed_h_2`
                  if is None.
        rand_s_2: (Optional) an 1D numpy array of 1 and -1, having the same shape
                  as `rand_h_2`. Automatically generated from `seed_s_2` if is
                  None.

    Returns:
        Compact bilinear pooled results `[batch_size, output_dim]`.
    """

    # Static shapes are needed to construction count sketch matrix
    input_dim1 = bottom1.get_shape().as_list()[-1]
    input_dim2 = bottom2.get_shape().as_list()[-1]

    # Step 0: Generate vectors and sketch matrix for tensor count sketch
    # This is only done once during graph construction, and fixed during each
    # operation
    if rand_h_1 is None:
        np.random.seed(seed_h_1)
        rand_h_1 = np.random.randint(output_dim, size=input_dim)
    if rand_s_1 is None:
        np.random.seed(seed_s_1)
        rand_s_1 = 2*np.random.randint(2, size=input_dim) - 1
    sparse_sketch_matrix1 = _generate_sketch_matrix(rand_h_1, rand_s_1, output_dim)
    if rand_h_2 is None:
        np.random.seed(seed_h_2)
        rand_h_2 = np.random.randint(output_dim, size=input_dim)
    if rand_s_2 is None:
        np.random.seed(seed_s_2)
        rand_s_2 = 2*np.random.randint(2, size=input_dim) - 1
    sparse_sketch_matrix2 = _generate_sketch_matrix(rand_h_2, rand_s_2, output_dim)

    # Step 1: Flatten the input tensors and count sketch
    bottom1 = tf.reshape(bottom1, [-1, input_dim1])
    bottom2 = tf.reshape(bottom2, [-1, input_dim2])
    # Essentially:
    #   sketch1 = bottom1 * sparse_sketch_matrix
    #   sketch2 = bottom2 * sparse_sketch_matrix
    # But tensorflow only supports left multiplying a sparse matrix, so:
    #   sketch1 = (sparse_sketch_matrix.T * bottom1.T).T
    #   sketch2 = (sparse_sketch_matrix.T * bottom2.T).T
    sketch1 = tf.transpose(tf.sparse_tensor_dense_matmul(sparse_sketch_matrix1,
        bottom1, adjoint_a=True, adjoint_b=True))
    sketch2 = tf.transpose(tf.sparse_tensor_dense_matmul(sparse_sketch_matrix2,
        bottom2, adjoint_a=True, adjoint_b=True))

    # Step 2: FFT
    fft1 = tf.batch_fft(tf.complex(real=sketch1, imag=tf.zeros_like(sketch1)))
    fft2 = tf.batch_fft(tf.complex(real=sketch2, imag=tf.zeros_like(sketch2)))

    # Step 3: Elementwise product
    fft_product = tf.mul(fft1, fft2)

    # Step 4: Inverse FFT and reshape back
    cbp = tf.real(tf.batch_ifft(fft_product))

    return cbp

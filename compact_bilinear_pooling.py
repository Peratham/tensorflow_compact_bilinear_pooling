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

def compact_bilinear_pooling_layer(bottom1, bottom2, output_dim, rand_h=None,
    rand_s=None, rand_h_seed=3, rand_s_seed=3):
    """
    Compute compact bilinear pooling over two bottom inputs. Reference:

    Yang Gao, et al. "Compact Bilinear Pooling." in Proceedings of IEEE
    Conference on Computer Vision and Pattern Recognition (2016).
    Akira Fukui, et al. "Multimodal Compact Bilinear Pooling for Visual Question
    Answering and Visual Grounding." arXiv preprint arXiv:1606.01847 (2016).

    Args:
        bottom1: first input, 2D Tensor of shape [batch_size, input_dim].
        bottom2: second input, 2D Tensor of shape [batch_size, input_dim].
        output_dim: output dimension for compact bilinear pooling
        rand_h: (Optional) an 1D numpy array containing indices in interval
                `[0, output_dim)`. Automatically generated from `rand_h_seed` if
                is None.
        rand_s: (Optional) an 1D numpy array of 1 and -1, having the same shape
                as `rand_h`. Automatically generated from `rand_s_seed` if is
                None.

    Returns:
        Compact bilinear pooled results `[batch_size, output_dim]`.
    """

    bottom_shape1 = bottom1.get_shape().as_list()
    bottom_shape2 = bottom2.get_shape().as_list()
    assert(bottom_shape1[-1] == bottom_shape2[-1])
    input_dim = bottom_shape1[-1]

    # Step 0: Generate vectors and sketch matrix for tensor count sketch
    # This is only done once during graph construction, and fixed during each
    # operation
    if rand_h is None:
        np.random.seed(rand_h_seed)
        rand_h = np.random.randint(output_dim, size=input_dim)
    if rand_s is None:
        np.random.seed(rand_s_seed)
        rand_s = 2*np.random.randint(2, size=input_dim) - 1
    sparse_sketch_matrix = _generate_sketch_matrix(rand_h, rand_s, output_dim)

    # Step 1: Count sketch
    # Essentially:
    #   sketch1 = bottom1 * sparse_sketch_matrix
    #   sketch2 = bottom2 * sparse_sketch_matrix
    # But tensorflow only supports left multiplying a sparse matrix, so:
    #   sketch1 = (sparse_sketch_matrix.T * bottom1.T).T
    #   sketch2 = (sparse_sketch_matrix.T * bottom2.T).T
    sketch1 = tf.transpose(tf.sparse_tensor_dense_matmul(sparse_sketch_matrix,
        bottom1, adjoint_a=True, adjoint_b=True))
    sketch2 = tf.transpose(tf.sparse_tensor_dense_matmul(sparse_sketch_matrix,
        bottom2, adjoint_a=True, adjoint_b=True))

    # Step 2: FFT
    fft1 = tf.batch_fft(tf.complex(real=sketch1, imag=tf.zeros_like(sketch1)))
    fft2 = tf.batch_fft(tf.complex(real=sketch2, imag=tf.zeros_like(sketch2)))

    # Step 3: Elementwise product
    fft_product = tf.mul(fft1, fft2)

    # Step 4: Inverse FFT
    cbp = tf.real(tf.batch_ifft(fft_product))

    return cbp

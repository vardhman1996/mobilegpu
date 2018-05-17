from __future__ import absolute_import, print_function

import tvm
import numpy as np
import topi
from layers.utils import REMOTE as remote
from tvm.contrib import rpc, util, ndk

# Global declarations of environment.

# llvm
# tgt_host="llvm"
# llvm, cuda, opencl, metal
# Change it to respective GPU if gpu is enabled Ex: cuda, opencl
# tgt="llvm"

def _export_module(f, name, remote):
    temp = util.tempdir()
    path_dso = temp.relpath("{0}.so".format(name))
    f.export_library(path_dso, ndk.create_shared)
    remote.upload(path_dso)
    f_new = remote.load_module("{0}.so".format(name))
    return f_new


def make_elemwise_add(shape, tgt, tgt_host, func_name, dtype="float32"):
    A = tvm.placeholder(shape, dtype=dtype, name="A")
    B = tvm.placeholder(shape, dtype=dtype, name="B")
    C = tvm.compute(A.shape, lambda *i: A(*i) + B(*i))

    s = tvm.create_schedule(C.op)
    f = tvm.build(s, [A, B, C], tgt, target_host=tgt_host, name=func_name)
    return _export_module(f, func_name, remote)


def make_elemwise_mul(shape, tgt, tgt_host, func_name, dtype="float32"):
    """TODO: Your code here"""
    A = tvm.placeholder(shape, dtype=dtype, name="A")
    B = tvm.placeholder(shape, dtype=dtype, name="B")
    C = tvm.compute(A.shape, lambda *i: A(*i) * B(*i))

    s = tvm.create_schedule(C.op)
    f = tvm.build(s, [A, B, C], tgt, target_host=tgt_host, name=func_name)
    return _export_module(f, func_name, remote)


def make_elemwise_add_by_const(shape, const_k, tgt, tgt_host, func_name,
                               dtype="float32"):
    """TODO: Your code here"""
    A = tvm.placeholder(shape, dtype=dtype, name="A")
    B = tvm.const(const_k, dtype=dtype)
    C = tvm.compute(A.shape, lambda *i: A(*i) + B)

    s = tvm.create_schedule(C.op)
    f = tvm.build(s, [A, C], tgt, target_host=tgt_host, name=func_name)
    return _export_module(f, func_name, remote)


def make_elemwise_mul_by_const(shape, const_k, tgt, tgt_host, func_name,
                            dtype="float32"):
    """TODO: Your code here"""
    A = tvm.placeholder(shape, dtype=dtype, name="A")
    B = tvm.const(const_k, dtype=dtype)
    C = tvm.compute(A.shape, lambda *i: A(*i) * B)

    s = tvm.create_schedule(C.op)
    f = tvm.build(s, [A, C], tgt, target_host=tgt_host, name=func_name)
    return _export_module(f, func_name, remote)

def make_relu(shape, tgt, tgt_host, func_name, dtype="float32"):
    """TODO: Your code here"""
    """Hint: use tvm.max, tvm.const(0, A.dtype)"""
    A = tvm.placeholder(shape, dtype=dtype, name='A')
    B = tvm.const(0, dtype=dtype)
    C = tvm.compute(A.shape, lambda *i: tvm.max(A(*i),  B))

    s = tvm.create_schedule(C.op)
    f = tvm.build(s, [A, C], tgt, target_host=tgt_host, name=func_name)
    return _export_module(f, func_name, remote)


def make_relu_gradient(shape, tgt, tgt_host, func_name, dtype="float32"):
    """TODO: Your code here"""
    """Hint: use tvm.select"""
    A = tvm.placeholder(shape, dtype=dtype, name='A')
    grad = tvm.placeholder(shape, dtype=dtype, name='grad')
    relu_grad = tvm.compute(A.shape, lambda *i: tvm.select(A(*i) <= 0, 0.0, 1.0 * grad(*i)))


    s = tvm.create_schedule([relu_grad.op])
    f = tvm.build(s, [A, grad, relu_grad], tgt, target_host=tgt_host, name=func_name)
    return _export_module(f, func_name, remote)

def make_matrix_mul(shapeA, transposeA, shapeB, transposeB, tgt, tgt_host,
                    func_name, dtype="float32"):
    """TODO: Your code here"""
    """Hint: use tvm.reduce_axis, tvm.sum"""
    """Hint: treat 4 cases of transposeA, transposeB separately"""
    """Hint: for tvm schedule, use split, reorder, vectorize, parallel"""
    """Hint: debug tvm schedule using tvm.lower"""
    X = tvm.placeholder(shapeA, dtype=dtype, name='X')
    Y = tvm.placeholder(shapeB, dtype=dtype, name='Y')

    if not transposeA and not transposeB:
        k = tvm.reduce_axis((0, shapeA[1]), name='k')
        Z = tvm.compute((shapeA[0], shapeB[1]), lambda i, j: tvm.sum(X[i, k] * Y[k, j], axis=k), name='Z')
    elif not transposeA and transposeB:
        k = tvm.reduce_axis((0, shapeA[1]), name='k')
        Z = tvm.compute((shapeA[0], shapeB[0]), lambda i, j: tvm.sum(X[i, k] * Y[j, k], axis=k), name='Z')
    elif transposeA and not transposeB:
        k = tvm.reduce_axis((0, shapeA[0]), name='k')
        Z = tvm.compute((shapeA[1], shapeB[1]), lambda i, j: tvm.sum(X[k, i] * Y[k, j], axis=k), name='Z')
    elif transposeA and transposeB:
        k = tvm.reduce_axis((0, shapeA[0]), name='k')
        Z = tvm.compute((shapeA[1], shapeB[0]), lambda i, j: tvm.sum(X[k, i] * Y[j, k], axis=k), name='Z')

    s = tvm.create_schedule(Z.op)
    xo, yo, xi, yi = s[Z].tile(Z.op.axis[0], Z.op.axis[1], 8, 8)
    k, = s[Z].op.reduce_axis
    ko, ki = s[Z].split(k, factor=5)
    s[Z].reorder(xo, yo, ko, xi, ki, yi)
    s[Z].vectorize(yi)
    s[Z].parallel(xo)

    block_x = tvm.thread_axis("blockIdx.x")
    thread_x = tvm.thread_axis("threadIdx.x")

    # s[Z].bind(xi, block_x)
    # if len(Z.op.axis) > 1:
    #     s[Z].bind(yi, thread_x)


    f = tvm.build(s, [X, Y, Z], tgt, target_host=tgt_host, name=func_name)
    return _export_module(f, func_name, remote)


def make_conv2d(shapeX, shapeF, tgt, tgt_host, func_name, dtype="float32"):
    assert(shapeX[1] == shapeF[1])
    N, C, H, W = shapeX
    M, C, R, S = shapeF

    """TODO: Your code here"""
    """Hint: use tvm.reduce_axis, tvm.sum"""
    """Hint: go by conv2d definition. Treat stride=1, padding=0 case only."""
    """For a challenge, treat the general case for stride and padding."""
    X = tvm.placeholder(shapeX, dtype=dtype, name='X')
    F = tvm.placeholder(shapeF, dtype=dtype, name='F')

    kx = tvm.reduce_axis((0, R), name='kx')
    ky = tvm.reduce_axis((0, S), name='ky')
    kc = tvm.reduce_axis((0, C), name='kc')
    Y = tvm.compute((N, M, H - R + 1, W - S + 1),
                    lambda n,m,h,w: tvm.sum(X[n, kc, h + kx, w + ky] * F[m, kc, kx, ky], axis=[kx,ky,kc]))

    s = tvm.create_schedule(Y.op)
    f = tvm.build(s, [X, F, Y], tgt, target_host=tgt_host, name=func_name)
    return _export_module(f, func_name, remote)


def make_matrix_softmax(shape, tgt, tgt_host, func_name, dtype="float32"):

    """TODO: Your code here"""
    """Hint: use tvm.reduce_axis, tvm.sum, tvm.max, tvm.exp"""
    """Hint: do not reuse the same reduction axis j."""
    """Hint: implement the following version for better stability
        e_x = np.exp(x - np.max(x))
        softmax(x)= e_x / e_x.sum()
    """
    X = tvm.placeholder(shape, dtype=dtype, name='X')
    ky = tvm.reduce_axis((0, shape[1]), name='ky')
    MAX_X = tvm.compute((shape[0],), lambda i: tvm.max(X[i, ky], axis=[ky]))
    E_X = tvm.compute(shape, lambda i, j: tvm.exp(X[i, j] - MAX_X(i)))
    ky_n = tvm.reduce_axis((0, shape[1]), name='ky_n')
    E_X_SUM = tvm.compute((shape[0],), lambda i: tvm.sum(E_X[i, ky_n], axis=[ky_n]))
    Y = tvm.compute(shape, lambda i,j: E_X[i,j] / E_X_SUM(i))
    s = tvm.create_schedule(Y.op)
    f = tvm.build(s, [X, Y], tgt, target_host=tgt_host, name=func_name)
    return _export_module(f, func_name, remote)

def make_matrix_softmax_cross_entropy(shape, tgt, tgt_host, func_name,
                                      dtype="float32"):
    """TODO: Your code here"""
    """Hint: output shape should be (1,)"""
    X = tvm.placeholder(shape, dtype=dtype, name='X')
    a1 = tvm.reduce_axis((0, shape[1]), name='a1')
    MAX_X = tvm.compute((shape[0],), lambda i: tvm.max(X[i, a1], axis=[a1]))
    E_X = tvm.compute(shape, lambda i, j: tvm.exp(X[i, j] - MAX_X(i)))
    a2 = tvm.reduce_axis((0, shape[1]), name='a2')
    E_X_SUM = tvm.compute((shape[0],), lambda i: tvm.sum(E_X[i, a2], axis=[a2]))
    SOFTMAX_X = tvm.compute(shape, lambda i, j: E_X[i, j] / E_X_SUM(i))

    LOG_SOFTMAX_X = tvm.compute(shape, lambda i,j: tvm.log(SOFTMAX_X[i, j]))
    X_P = tvm.placeholder(shape, dtype=dtype, name='X_P')

    MUL = tvm.compute(shape, lambda i,j: X_P[i,j] * LOG_SOFTMAX_X[i,j])
    a3 = tvm.reduce_axis((0, shape[1]), name='a3')
    SUM = tvm.compute((shape[0],), lambda i: tvm.sum(-MUL[i, a3], axis=[a3]))

    a4 = tvm.reduce_axis((0, shape[0]), name='a4')
    MEAN = tvm.compute((1,), lambda i: tvm.sum(SUM[a4] / shape[0], axis=[a4]))

    s = tvm.create_schedule(MEAN.op)
    f = tvm.build(s, [X, X_P, MEAN], tgt, target_host=tgt_host, name=func_name)
    return _export_module(f, func_name, remote)


def make_reduce_sum_axis_zero(shape, tgt, tgt_host, func_name, dtype="float32"):
    A = tvm.placeholder(shape, dtype=dtype, name="A")
    C = topi.sum(A, axis=0, keepdims=False)

    s = tvm.create_schedule(C.op)
    f = tvm.build(s, [A, C], tgt, target_host=tgt_host, name=func_name)
    return _export_module(f, func_name, remote)


def make_broadcast_to(shape, to_shape, tgt, tgt_host, func_name,
                      dtype="float32"):
    A = tvm.placeholder(shape, dtype=dtype, name="A")
    C = topi.broadcast_to(A, to_shape)

    s = tvm.create_schedule(C.op)
    f = tvm.build(s, [A, C], tgt, target_host=tgt_host, name=func_name)
    return _export_module(f, func_name, remote)


def make_sgd_update(shape, learning_rate, tgt, tgt_host, func_name,
                    dtype="float32"):
    X = tvm.placeholder(shape, dtype=dtype, name="A")
    grad = tvm.placeholder(shape, dtype=dtype, name="grad")
    Y = tvm.compute(shape, lambda *i: X(*i) - learning_rate * grad(*i))
    s = tvm.create_schedule(Y.op)

    block_x = tvm.thread_axis("blockIdx.x")
    thread_x = tvm.thread_axis("threadIdx.x")

    s[Y].bind(Y.op.axis[0], block_x)
    if len(shape) > 1:
        s[Y].bind(Y.op.axis[1], thread_x)

    f = tvm.build(s, [X, grad, Y], tgt, target_host=tgt_host, name=func_name)
    return _export_module(f, func_name, remote)
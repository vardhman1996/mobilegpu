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

    block_x = tvm.thread_axis("blockIdx.x")
    thread_x = tvm.thread_axis("threadIdx.x")

    s[C].bind(C.op.axis[0], block_x)
    if len(shape) > 1:
        s[C].bind(C.op.axis[1], thread_x)

    f = tvm.build(s, [A, B, C], tgt, target_host=tgt_host, name=func_name)
    return _export_module(f, func_name, remote)


def make_elemwise_mul(shape, tgt, tgt_host, func_name, dtype="float32"):
    """TODO: Your code here"""
    A = tvm.placeholder(shape, dtype=dtype, name="A")
    B = tvm.placeholder(shape, dtype=dtype, name="B")
    C = tvm.compute(A.shape, lambda *i: A(*i) * B(*i))

    s = tvm.create_schedule(C.op)

    block_x = tvm.thread_axis("blockIdx.x")
    thread_x = tvm.thread_axis("threadIdx.x")

    s[C].bind(C.op.axis[0], block_x)
    if len(shape) > 1:
        s[C].bind(C.op.axis[1], thread_x)

    f = tvm.build(s, [A, B, C], tgt, target_host=tgt_host, name=func_name)
    return _export_module(f, func_name, remote)

def make_oneslike_op(shape, tgt, tgt_host, func_name, dtype="float32"):
    """TODO: Your code here"""
    A = tvm.placeholder(shape, dtype=dtype, name="A")
    C = tvm.compute(A.shape, lambda *i: np.float32(1.0))

    s = tvm.create_schedule(C.op)

    block_x = tvm.thread_axis("blockIdx.x")
    thread_x = tvm.thread_axis("threadIdx.x")

    s[C].bind(C.op.axis[0], block_x)
    if len(shape) > 1:
        s[C].bind(C.op.axis[1], thread_x)

    f = tvm.build(s, [A, C], tgt, target_host=tgt_host, name=func_name)
    return _export_module(f, func_name, remote)

def make_zeroslike_op(shape, tgt, tgt_host, func_name, dtype="float32"):
    """TODO: Your code here"""
    A = tvm.placeholder(shape, dtype=dtype, name="A")
    C = tvm.compute(A.shape, lambda *i: np.float32(0.0))

    s = tvm.create_schedule(C.op)

    block_x = tvm.thread_axis("blockIdx.x")
    thread_x = tvm.thread_axis("threadIdx.x")

    s[C].bind(C.op.axis[0], block_x)
    if len(shape) > 1:
        s[C].bind(C.op.axis[1], thread_x)

    f = tvm.build(s, [A, C], tgt, target_host=tgt_host, name=func_name)
    return _export_module(f, func_name, remote)

def make_elemwise_add_by_const(shape, const_k, tgt, tgt_host, func_name,
                               dtype="float32"):
    """TODO: Your code here"""
    A = tvm.placeholder(shape, dtype=dtype, name="A")
    B = tvm.const(const_k, dtype=dtype)
    C = tvm.compute(A.shape, lambda *i: A(*i) + B)

    s = tvm.create_schedule(C.op)

    block_x = tvm.thread_axis("blockIdx.x")
    thread_x = tvm.thread_axis("threadIdx.x")

    s[C].bind(C.op.axis[0], block_x)
    if len(shape) > 1:
        s[C].bind(C.op.axis[1], thread_x)

    f = tvm.build(s, [A, C], tgt, target_host=tgt_host, name=func_name)
    return _export_module(f, func_name, remote)


def make_elemwise_mul_by_const(shape, const_k, tgt, tgt_host, func_name,
                            dtype="float32"):
    """TODO: Your code here"""
    A = tvm.placeholder(shape, dtype=dtype, name="A")
    B = tvm.const(const_k, dtype=dtype)
    C = tvm.compute(A.shape, lambda *i: A(*i) * B)

    s = tvm.create_schedule(C.op)

    block_x = tvm.thread_axis("blockIdx.x")
    thread_x = tvm.thread_axis("threadIdx.x")

    s[C].bind(C.op.axis[0], block_x)
    if len(shape) > 1:
        s[C].bind(C.op.axis[1], thread_x)

    f = tvm.build(s, [A, C], tgt, target_host=tgt_host, name=func_name)
    return _export_module(f, func_name, remote)

def make_relu(shape, tgt, tgt_host, func_name, dtype="float32"):
    """TODO: Your code here"""
    """Hint: use tvm.max, tvm.const(0, A.dtype)"""
    A = tvm.placeholder(shape, dtype=dtype, name='A')
    B = tvm.const(0, dtype=dtype)
    C = tvm.compute(A.shape, lambda *i: tvm.max(A(*i),  B))

    s = tvm.create_schedule(C.op)

    block_x = tvm.thread_axis("blockIdx.x")
    thread_x = tvm.thread_axis("threadIdx.x")

    s[C].bind(C.op.axis[0], block_x)
    if len(shape) > 1:
        s[C].bind(C.op.axis[1], thread_x)

    f = tvm.build(s, [A, C], tgt, target_host=tgt_host, name=func_name)
    return _export_module(f, func_name, remote)


def make_relu_gradient(shape, tgt, tgt_host, func_name, dtype="float32"):
    """TODO: Your code here"""
    """Hint: use tvm.select"""
    A = tvm.placeholder(shape, dtype=dtype, name='A')
    grad = tvm.placeholder(shape, dtype=dtype, name='grad')
    relu_grad = tvm.compute(A.shape, lambda *i: tvm.select(A(*i) <= 0, 0.0, 1.0 * grad(*i)))


    s = tvm.create_schedule([relu_grad.op])

    block_x = tvm.thread_axis("blockIdx.x")
    thread_x = tvm.thread_axis("threadIdx.x")

    s[relu_grad].bind(relu_grad.op.axis[0], block_x)
    if len(shape) > 1:
        s[relu_grad].bind(relu_grad.op.axis[1], thread_x)

    f = tvm.build(s, [A, grad, relu_grad], tgt, target_host=tgt_host, name=func_name)
    return _export_module(f, func_name, remote)

def make_matrix_mul_2(shapeA, transposeA, shapeB, transposeB, tgt, tgt_host,
                    func_name, dtype="float32"):
    # assert shapeA[0] == shapeA[1]
    # assert shapeB[0] == shapeB[1]

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
    # X_shared = s.cache_read(X, 'shared', [Z])  # should store (step * block_factor) items
    # Y_shared = s.cache_read(Y, 'shared', [Z])
    # X_local = s.cache_read(X_shared, 'local', [Z])
    # Y_local = s.cache_read(Y_shared, 'local', [Z])
    # Z_local = s.cache_write(Z, 'local')

    # tile consts
    tile = 2
    num_thread = 8
    block_factor = tile * num_thread
    step = 8
    vthread = 2

    block_x = tvm.thread_axis("blockIdx.x", name='bx')
    block_y = tvm.thread_axis("blockIdx.y", name='by')
    thread_x = tvm.thread_axis((0, num_thread), "threadIdx.x", name='tx')
    thread_y = tvm.thread_axis((0, num_thread), "threadIdx.y", name='ty')
    vthread_x = tvm.thread_axis((0, vthread), "vthread", name="vtx")
    vthread_y = tvm.thread_axis((0, vthread), "vthread", name="vty")

    # Split the workloads into blocks of threads
    hi, wi = s[Z].op.axis
    bx, wi = s[Z].split(wi, factor=block_factor) # wi ranges up to block_factor
    by, hi = s[Z].split(hi, factor=block_factor) # hi ranges up to block_factor

    s[Z].bind(bx, block_x) # bx number of blocks.
    s[Z].bind(by, block_y)

    # Split into virtual threads (vthread x vthread) grid
    vtx, wi = s[Z].split(wi, nparts=vthread) # vtx ranges up to vthread. wi ranges up to (block_factor/vthread)
    vty, hi = s[Z].split(hi, nparts=vthread) # vty ranges up to vthread. hi ranges up to (block_factor/vthread)

    # Split each vthread block into threads (num_thread x num_thread) grid
    tx, wi = s[Z].split(wi, nparts=num_thread) # tx ranges up to vthread. wi ranges up to (block_factor/vthread/num_thread)
    ty, hi = s[Z].split(hi, nparts=num_thread)  # ty ranges up to vthread. hi ranges up to (block_factor/vthread/num_thread)

    # Reorder from block to vthread to thread. Decreasing order of size of submatrix to be controlled
    s[Z].reorder(by, bx, vty, vtx, ty, tx)

    s[Z].bind(vty, vthread_y)
    s[Z].bind(vtx, vthread_x)
    s[Z].bind(tx, thread_x)
    s[Z].bind(ty, thread_y)

    # # Schedule Z_local local write
    # s[Z_local].compute_at(s[Z], tx) # In the computation of Z, when looping over tx (inner most and smallest granule), compute Z_local, which is a write to local memory
    # hi, wi = s[Z_local].op.axis
    # k, = s[Z_local].op.reduce_axis
    # ko, ki = s[Z_local].split(k, factor=step)
    # s[Z_local].reorder(ko, ki, hi, wi) # May be unnecessary
    #
    # # Attach computation to iteration variables
    #
    # s[X_shared].compute_at(s[Z_local], wi)
    # # s[Y_shared].compute_at(s[Z_local], hi)
    #
    # s[X_local].compute_at(s[Z_local], ki)
    # s[Y_local].compute_at(s[Z_local], ki)
    #
    # # Schedule for X's shared memory load
    # hi, wi =  s[X_shared].op.axis
    # ty, hi = s[X_shared].split(hi, nparts=num_thread)
    # tx, wi = s[X_shared].split(wi, nparts=num_thread)
    # _, wi = s[X_shared].split(wi, factor=4) # Is this 4 because of vthread = 2, vthread*vthread?
    # s[X_shared].reorder(ty, tx, hi, wi)
    # # tvm.lower(s, [X, Y, Z], simple_mode=True)
    # s[X_shared].bind(tx, thread_x)
    # s[X_shared].bind(ty, thread_y)
    # s[X_shared].vectorize(wi)
    #
    # # Schedule for Y's shared memory load
    # hi, wi = s[Y_shared].op.axis
    # ty, hi = s[Y_shared].split(hi, nparts=num_thread)
    # tx, wi = s[Y_shared].split(wi, nparts=num_thread)
    # _, wi = s[Y_shared].split(wi, factor=4)  # Is this 4 because of vthread = 2, vthread*vthread?
    # s[Y_shared].reorder(ty, tx, hi, wi)
    # s[Y_shared].bind(tx, thread_x)
    # s[Y_shared].bind(ty, thread_y)
    # s[Y_shared].vectorize(wi)

    f = tvm.build(s, [X, Y, Z], target=tgt, target_host=tgt_host, name=func_name)
    return _export_module(f, func_name, remote)

def make_matrix_mul(shapeA, transposeA, shapeB, transposeB, tgt, tgt_host,
                    func_name, dtype="float32", args_opt={'x_f': 8, 'y_f':1, 'k_f': 8}):
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

    x_f = args_opt['x_f']
    y_f = args_opt['y_f']
    k_f = args_opt['k_f']

    s = tvm.create_schedule(Z.op)
    xo, yo, xi, yi = s[Z].tile(Z.op.axis[0], Z.op.axis[1], x_f, y_f)

    k, = s[Z].op.reduce_axis
    ko, ki = s[Z].split(k, factor=k_f)
    s[Z].reorder(xo, yo, ko, xi, ki, yi)
    s[Z].vectorize(yi)

    # zz = s[Z].fuse(ko, xi)

    tvm.lower(s, [X,Y,Z], simple_mode=True)
    s[Z].bind(xo, tvm.thread_axis("blockIdx.x"))
    s[Z].bind(yo, tvm.thread_axis("threadIdx.x"))

    # s[Z].bind(Z.op.axis[0], tvm.thread_axis("blockIdx.x"))
    # s[Z].bind(Z.op.axis[1], tvm.thread_axis("threadIdx.x"))


    f = tvm.build(s, [X, Y, Z], tgt, target_host=tgt_host, name=func_name)
    return _export_module(f, func_name, remote)


def make_conv2d_unoptimized(shapeX, shapeF, tgt, tgt_host, func_name, dtype="float32"):
    in_size, in_size, in_channel, batch = shapeX
    kernel, kernel, in_channel, out_channel = shapeF
    pad = 1
    stride = 1

    A = tvm.placeholder((in_size, in_size, in_channel, batch), name='A')
    W = tvm.placeholder((kernel, kernel, in_channel, out_channel), name='W')
    out_size = (in_size - kernel + 2 * pad) // stride + 1
    # Pad input
    Apad = tvm.compute(
        (in_size + 2 * pad, in_size + 2 * pad, in_channel, batch),
        lambda yy, xx, cc, nn: tvm.select(
            tvm.all(yy >= pad, yy - pad < in_size,
                    xx >= pad, xx - pad < in_size),
            A[yy - pad, xx - pad, cc, nn], tvm.const(0.)),
        name='Apad')


    # Create reduction variables
    rc = tvm.reduce_axis((0, in_channel), name='rc')
    ry = tvm.reduce_axis((0, kernel), name='ry')
    rx = tvm.reduce_axis((0, kernel), name='rx')
    # Compute the convolution
    B = tvm.compute(
        (out_size, out_size, out_channel, batch),
        lambda yy, xx, ff, nn: tvm.sum(
            Apad[yy * stride + ry, xx * stride + rx, rc, nn] * W[ry, rx, rc, ff],
            axis=[ry, rx, rc]),
        name='B')

    s = tvm.create_schedule(B.op)

    s[Apad].bind(Apad.op.axis[0], tvm.thread_axis("blockIdx.x"))
    s[Apad].bind(Apad.op.axis[1], tvm.thread_axis("threadIdx.x"))

    s[B].bind(B.op.axis[0], tvm.thread_axis("blockIdx.x"))
    s[B].bind(B.op.axis[1], tvm.thread_axis("threadIdx.x"))

    f = tvm.build(s, [A, W, B], tgt, target_host=tgt_host, name=func_name)

    return _export_module(f, func_name, remote)


def make_conv2d(shapeX, shapeF, tgt, tgt_host, func_name, dtype="float32"):
    in_size, in_size, in_channel, batch = shapeX
    kernel, kernel, in_channel, out_channel  = shapeF

    print(tgt, tgt_host)

    """TODO: Your code here"""
    """Hint: use tvm.reduce_axis, tvm.sum"""
    """Hint: go by conv2d definition. Treat stride=1, padding=0 case only."""
    """For a challenge, treat the general case for stride and padding."""
    # X = tvm.placeholder(shapeX, dtype=dtype, name='X')
    # F = tvm.placeholder(shapeF, dtype=dtype, name='F')
    #
    # kx = tvm.reduce_axis((0, R), name='kx')
    # ky = tvm.reduce_axis((0, S), name='ky')
    # kc = tvm.reduce_axis((0, C), name='kc')
    # Y = tvm.compute((N, M, H - R + 1, W - S + 1),
    #                 lambda n,m,h,w: tvm.sum(X[n, kc, h + kx, w + ky] * F[m, kc, kx, ky], axis=[kx,ky,kc]))
    #
    # s = tvm.create_schedule(Y.op)
    #
    # block_x = tvm.thread_axis("blockIdx.x")
    # thread_x = tvm.thread_axis("threadIdx.x")
    #
    # s[Y].bind(kx, block_x)
    # s[Y].bind(ky, thread_x)
    #
    # f = tvm.build(s, [X, F, Y], tgt, target_host=tgt_host, name=func_name)


    pad = 1
    stride = 1

    # Algorithm
    A = tvm.placeholder((in_size, in_size, in_channel, batch), name='A')
    W = tvm.placeholder((kernel, kernel, in_channel, out_channel), name='W')
    out_size = (in_size - kernel + 2 * pad) // stride + 1
    # Pad input
    Apad = tvm.compute(
        (in_size + 2 * pad, in_size + 2 * pad, in_channel, batch),
        lambda yy, xx, cc, nn: tvm.select(
            tvm.all(yy >= pad, yy - pad < in_size,
                    xx >= pad, xx - pad < in_size),
            A[yy - pad, xx - pad, cc, nn], tvm.const(0.)),
        name='Apad')
    # Create reduction variables
    rc = tvm.reduce_axis((0, in_channel), name='rc')
    ry = tvm.reduce_axis((0, kernel), name='ry')
    rx = tvm.reduce_axis((0, kernel), name='rx')
    # Compute the convolution
    B = tvm.compute(
        (out_size, out_size, out_channel, batch),
        lambda yy, xx, ff, nn: tvm.sum(
            Apad[yy * stride + ry, xx * stride + rx, rc, nn] * W[ry, rx, rc, ff],
            axis=[ry, rx, rc]),
        name='B')

    # Designate the memory hierarchy
    s = tvm.create_schedule(B.op)
    s[Apad].compute_inline()  # compute Apad inline
    AA = s.cache_read(Apad, 'shared', [B])
    WW = s.cache_read(W, "shared", [B])
    AL = s.cache_read(AA, "local", [B])
    WL = s.cache_read(WW, "local", [B])
    BL = s.cache_write(B, "local")

    # tile consts
    tile = 2
    num_thread = 16
    block_factor = tile * num_thread
    step = 4
    vthread = 1

    # Get the GPU thread indices
    block_x = tvm.thread_axis("blockIdx.x")
    block_y = tvm.thread_axis("blockIdx.y")
    block_z = tvm.thread_axis("blockIdx.z")
    thread_x = tvm.thread_axis((0, num_thread), "threadIdx.x")
    thread_y = tvm.thread_axis((0, num_thread), "threadIdx.y")
    thread_xz = tvm.thread_axis((0, vthread), "vthread", name="vx")
    thread_yz = tvm.thread_axis((0, vthread), "vthread", name="vy")

    # Split the workloads
    hi, wi, fi, ni = s[B].op.axis
    bz = s[B].fuse(hi, wi)
    by, fi = s[B].split(fi, factor=block_factor)
    bx, ni = s[B].split(ni, factor=block_factor)

    # Bind the iteration variables to GPU thread indices
    s[B].bind(bz, block_z)
    s[B].bind(by, block_y)
    s[B].bind(bx, block_x)

    tyz, fi = s[B].split(fi, nparts=vthread)  # virtual thread split
    txz, ni = s[B].split(ni, nparts=vthread)  # virtual thread split
    ty, fi = s[B].split(fi, nparts=num_thread)
    tx, ni = s[B].split(ni, nparts=num_thread)
    s[B].reorder(bz, by, bx, tyz, txz, ty, tx, fi, ni)

    s[B].bind(tyz, thread_yz)
    s[B].bind(txz, thread_xz)
    s[B].bind(ty, thread_y)
    s[B].bind(tx, thread_x)

    # Schedule BL local write
    s[BL].compute_at(s[B], tx)
    yi, xi, fi, ni = s[BL].op.axis
    ry, rx, rc = s[BL].op.reduce_axis
    rco, rci = s[BL].split(rc, factor=step)
    s[BL].reorder(rco, ry, rx, rci, fi, ni)

    # Attach computation to iteration variables
    s[AA].compute_at(s[BL], rx)
    s[WW].compute_at(s[BL], rx)
    s[AL].compute_at(s[BL], rci)
    s[WL].compute_at(s[BL], rci)

    # Schedule for A's shared memory load
    yi, xi, ci, ni = s[AA].op.axis
    ty, ci = s[AA].split(ci, nparts=num_thread)
    tx, ni = s[AA].split(ni, nparts=num_thread)
    _, ni = s[AA].split(ni, factor=4)
    s[AA].reorder(ty, tx, yi, xi, ci, ni)
    s[AA].bind(ty, thread_y)
    s[AA].bind(tx, thread_x)
    s[AA].vectorize(ni)  # vectorize memory load

    # Schedule for W's shared memory load
    yi, xi, ci, fi = s[WW].op.axis
    ty, ci = s[WW].split(ci, nparts=num_thread)
    tx, fi = s[WW].split(fi, nparts=num_thread)
    _, fi = s[WW].split(fi, factor=4)
    s[WW].reorder(ty, tx, yi, xi, ci, fi)
    s[WW].bind(ty, thread_y)
    s[WW].bind(tx, thread_x)
    s[WW].vectorize(fi)  # vectorize memory load

    f = tvm.build(s, [A, W, B], tgt, target_host=tgt_host, name=func_name)

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

    block_x = tvm.thread_axis("blockIdx.x")
    thread_x = tvm.thread_axis("threadIdx.x")

    # MAX_X
    s[MAX_X].bind(MAX_X.op.axis[0], tvm.thread_axis("blockIdx.x"))
    s[MAX_X].bind(ky, tvm.thread_axis("threadIdx.x"))

    # E_X
    s[E_X].bind(E_X.op.axis[0], tvm.thread_axis("blockIdx.x"))
    s[E_X].bind(E_X.op.axis[1], tvm.thread_axis("threadIdx.x"))

    # E_X_SUM
    s[E_X_SUM].bind(E_X_SUM.op.axis[0], tvm.thread_axis("blockIdx.x"))
    s[E_X_SUM].bind(ky_n, tvm.thread_axis("threadIdx.x"))

    # SOFTMAX_X
    s[Y].bind(Y.op.axis[0], tvm.thread_axis("blockIdx.x"))
    s[Y].bind(Y.op.axis[1], tvm.thread_axis("threadIdx.x"))

    # print(tvm.lower(s, [X, Y], simple_mode=True))

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

    # s = tvm.create_schedule([MAX_X.op, E_X.op, E_X_SUM.op, SOFTMAX_X.op, LOG_SOFTMAX_X.op, MUL.op, SUM.op, MEAN.op])
    s = tvm.create_schedule(MEAN.op)

    # print(tvm.lower(s, [X, X_P, MEAN], simple_mode=True))

    # MAX_X
    s[MAX_X].bind(MAX_X.op.axis[0], tvm.thread_axis("blockIdx.x"))
    s[MAX_X].bind(a1, tvm.thread_axis("threadIdx.x"))

    # E_X
    s[E_X].bind(E_X.op.axis[0], tvm.thread_axis("blockIdx.x"))
    s[E_X].bind(E_X.op.axis[1], tvm.thread_axis("threadIdx.x"))

    # E_X_SUM
    s[E_X_SUM].bind(E_X_SUM.op.axis[0], tvm.thread_axis("blockIdx.x"))
    s[E_X_SUM].bind(a2, tvm.thread_axis("threadIdx.x"))

    # SOFTMAX_X
    s[SOFTMAX_X].bind(SOFTMAX_X.op.axis[0], tvm.thread_axis("blockIdx.x"))
    s[SOFTMAX_X].bind(SOFTMAX_X.op.axis[1], tvm.thread_axis("threadIdx.x"))

    # LOG_SOFT_MAX
    s[LOG_SOFTMAX_X].bind(LOG_SOFTMAX_X.op.axis[0], tvm.thread_axis("blockIdx.x"))
    s[LOG_SOFTMAX_X].bind(LOG_SOFTMAX_X.op.axis[1], tvm.thread_axis("threadIdx.x"))

    # MUL
    s[MUL].bind(MUL.op.axis[0], tvm.thread_axis("blockIdx.x"))
    s[MUL].bind(MUL.op.axis[1], tvm.thread_axis("threadIdx.x"))

    # SUM
    s[SUM].bind(SUM.op.axis[0], tvm.thread_axis("blockIdx.x"))
    s[SUM].bind(a3, tvm.thread_axis("threadIdx.x"))

    # MEAN
    # s[MEAN].bind(a4, tvm.thread_axis("blockIdx.x"))
    s[MEAN].bind(a4, tvm.thread_axis("threadIdx.x"))


    # print(tvm.lower(s, [X, X_P, MEAN], simple_mode=True))

    # block_x = tvm.thread_axis("blockIdx.x")
    # thread_x = tvm.thread_axis("threadIdx.x")

    # zo, zi = s[SUM].split(SUM.op.axis[0], 3)
    # print(tvm.lower(s, [X, X_P, MEAN], simple_mode=True))
    # s[SUM].bind(zo, block_x)
    # s[SUM].bind(zi, thread_x)

    f = tvm.build(s, [X, X_P, MEAN], tgt, target_host=tgt_host, name=func_name)
    return _export_module(f, func_name, remote)


def make_reduce_sum_axis_zero(shape, tgt, tgt_host, func_name, dtype="float32"):
    A = tvm.placeholder(shape, dtype=dtype, name="A")
    C = topi.sum(A, axis=0, keepdims=False)

    s = tvm.create_schedule(C.op)

    block_x = tvm.thread_axis("blockIdx.x")
    thread_x = tvm.thread_axis("threadIdx.x")

    print(C.op.axis, C)
    # s[C].bind(C.op.axis[0], block_x)
    s[C].bind(C.op.axis[0], thread_x)

    f = tvm.build(s, [A, C], tgt, target_host=tgt_host, name=func_name)
    return _export_module(f, func_name, remote)


def make_broadcast_to(shape, to_shape, tgt, tgt_host, func_name,
                      dtype="float32"):
    A = tvm.placeholder(shape, dtype=dtype, name="A")
    C = topi.broadcast_to(A, to_shape)

    s = tvm.create_schedule(C.op)

    block_x = tvm.thread_axis("blockIdx.x")
    thread_x = tvm.thread_axis("threadIdx.x")

    s[C].bind(C.op.axis[0], block_x)
    if len(to_shape) > 1:
        s[C].bind(C.op.axis[1], thread_x)

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
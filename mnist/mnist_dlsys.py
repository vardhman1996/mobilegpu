import numpy as np
import argparse
import six.moves.cPickle as pickle
import gzip
import os
import time

import tvm
from tvm.contrib import rpc, util, ndk
from layers import autodiff as ad
from layers import tvm_op
import csv
from tqdm import tqdm
import time

print(tvm.__file__)

os.environ["TVM_NDK_CC"] = "/opt/android-toolchain-arm64/bin/aarch64-linux-android-gcc"

def load_mnist_data(dataset):
    """ Load the dataset
    Code adapted from http://deeplearning.net/tutorial/code/logistic_sgd.py

    :type dataset: string
    :param dataset: the path to the dataset (here MNIST)
    """
    # Download the MNIST dataset if it is not present
    data_dir, data_file = os.path.split(dataset)
    if data_dir == "" and not os.path.isfile(dataset):
        # Check if dataset is in the data directory.
        new_path = os.path.join(
            os.path.split(__file__)[0],
            dataset
        )
        if os.path.isfile(new_path) or data_file == 'mnist.pkl.gz':
            dataset = new_path

    if (not os.path.isfile(dataset)) and data_file == 'mnist.pkl.gz':
        from six.moves import urllib
        origin = (
            'http://www.iro.umontreal.ca/~lisa/deep/data/mnist/mnist.pkl.gz'
        )
        print('Downloading data from %s' % origin)
        urllib.request.urlretrieve(origin, dataset)

    print('Loading data...')

    # Load the dataset
    with gzip.open(dataset, 'rb') as f:
        try:
            train_set, valid_set, test_set = pickle.load(f, encoding='latin1')
        except:
            train_set, valid_set, test_set = pickle.load(f)
    # train_set, valid_set, test_set format: tuple(input, target)
    # input is a numpy.ndarray of 2 dimensions (a matrix), np.float32
    # where each row corresponds to an example. target is a
    # numpy.ndarray of 1 dimension (vector), np.int64 that has the same length
    # as the number of rows in the input. It should give the target
    # to the example with the same index in the input.
    return train_set, valid_set, test_set


def convert_to_one_hot(vals):
    """Helper method to convert label array to one-hot array."""
    # one_hot_vals = np.zeros((vals.size, vals.max()+1))
    one_hot_vals = np.zeros((vals.size, 10))
    one_hot_vals[np.arange(vals.size), vals] = 1
    return one_hot_vals

print('shapeA', 'shapeB', 'x_f', 'y_f', 'k_f', 'time_taken')

def exp_per_mat_mul(shapeA, shapeB, executor_ctx, writer):
    x_f_all = range(1, 17)
    y_f_all = range(1, 9)
    k_f_all = range(1, 17)

    tgt = 'opencl'
    tgt_host = 'llvm -target=aarch64-linux-android'

    for x_f in tqdm(x_f_all):
        # time.sleep(np.random.randint(10))
        for y_f in tqdm(y_f_all):
            # time.sleep(np.random.randint(5))
            for k_f in tqdm(k_f_all):
                if y_f < min(x_f, k_f):
                    args_opt = {'x_f': x_f, 'y_f': y_f, 'k_f': k_f}
                    A = tvm.nd.array(np.random.uniform(0, 10, size=shapeA).astype("float32"), ctx=executor_ctx)
                    B = tvm.nd.array(np.random.uniform(0, 10, size=shapeB).astype("float32"), ctx=executor_ctx)
                    shapeOut = (shapeA[0], shapeB[1])
                    Out = tvm.nd.array(np.zeros(shapeOut).astype("float32"), ctx=executor_ctx)
                    id = str(shapeA[0]) + '_' + str(shapeA[1]) + '_' + str(shapeB[0]) + '_' + str(shapeB[1]) + '_' + str(x_f) + '_' + str(y_f) + '_' + str(k_f)
                    matrix_mul = tvm_op.make_matrix_mul(shapeA, False, shapeB, False, tgt, tgt_host, "matrix_mul_" + id, args_opt=args_opt)
                    start_time = time.time()
                    matrix_mul(A, B, Out)
                    total_time = time.time() - start_time
                    writer.writerow([str(shapeA), str(shapeB), x_f, y_f, k_f, total_time])


def exp_matrix_multiply(ctx):
    shapeW1 = (784, 256)
    shapeW2 = (256, 100)
    shapeW3 = (100, 10)

    batches = [4000]

    X_shapes_W1 = [(ele, 784) for ele in batches]

    all_muls_W1 = [(ele_x, shapeW1) for ele_x in X_shapes_W1]

    with open('exp_mat_mul_{}_batch.csv'.format(batches[0]), 'w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile, delimiter='|')
        csv_writer.writerow(['shapeA', 'shapeB', 'x_f', 'y_f', 'k_f', 'time_taken'])
        for shapeA, shapeB in tqdm(all_muls_W1):
            exp_per_mat_mul(shapeA, shapeB, ctx, csv_writer)
            print('Done combnation', shapeA, shapeB)



def matmul_exps(executor_ctx):
    tgt = 'opencl'
    tgt_host = 'llvm -target=aarch64-linux-android'

    X = ad.Variable(name="X")
    W1 = ad.Variable(name="W1")
    y_ = ad.Variable(name="y_")

    Y = ad.matmul_op(X, W1)

    loss = ad.softmaxcrossentropy_op(Y, y_)
    grad_W1, = ad.gradients(loss, [W1])
    print("HELLLO", grad_W1)
    print()
    executor = ad.Executor([Y, loss, grad_W1], ctx=executor_ctx)

    X_val = tvm.nd.array(np.ones((1, 784)).astype("float32"), ctx=executor_ctx)
    W1_val = tvm.nd.array(np.zeros((784, 10)).astype("float32"), ctx=executor_ctx)
    y_val = tvm.nd.array(np.zeros((1,10)).astype("float32"), ctx=executor_ctx)
    # y_val_copy = tvm.nd.array(np.zeros((1,10)).astype("float32"), ctx=executor_ctx)

    y_pred, loss_val, _ = executor.run(feed_dict={X: X_val, W1: W1_val, y_: y_val})
    print(y_pred)
    print(loss_val)




def mnist_logreg(executor_ctx, num_epochs=10, print_loss_val_each_epoch=False):
    print("=== Build logistic regression model...")

    print("mnist_logreg", executor_ctx.device_name)
    tgt = 'opencl'
    tgt_host = 'llvm -target=aarch64-linux-android'

    W1 = ad.Variable(name="W1")
    b1 = ad.Variable(name="b1")
    X = ad.Variable(name="X")
    y_ = ad.Variable(name="y_")

    z1 = ad.matmul_op(X, W1)
    y = z1 + ad.broadcastto_op(b1, z1)

    loss = ad.softmaxcrossentropy_op(y, y_)

    grad_W1, grad_b1 = ad.gradients(loss, [W1, b1])
    print(grad_W1)
    executor = ad.Executor([loss, grad_W1, grad_b1, y], ctx=executor_ctx)

    # Read input data
    datasets = load_mnist_data("mnist.pkl.gz")
    train_set_x, train_set_y = datasets[0]
    valid_set_x, valid_set_y = datasets[1]
    test_set_x, test_set_y = datasets[2]

    # Set up minibatch
    batch_size = 1000
    n_train_batches = train_set_x.shape[0] // batch_size
    n_valid_batches = valid_set_x.shape[0] // batch_size

    print("Start training loop...")

    # Initialize parameters
    W1_val = np.zeros((784, 10), dtype=np.float32)
    b1_val = np.zeros((10), dtype=np.float32)
    X_val = np.empty(shape=(batch_size, 784), dtype=np.float32)
    y_val = np.empty(shape=(batch_size, 10), dtype=np.float32)
    valid_X_val = np.empty(shape=(batch_size, 784), dtype=np.float32)
    valid_y_val = np.empty(shape=(batch_size, 10), dtype=np.float32)

    # wrap them under tvm.nd.array
    W1_val = tvm.nd.array(W1_val, ctx=executor_ctx)
    b1_val = tvm.nd.array(b1_val, ctx=executor_ctx)
    X_val = tvm.nd.array(X_val, ctx=executor_ctx)
    y_val = tvm.nd.array(y_val, ctx=executor_ctx)
    valid_X_val = tvm.nd.array(valid_X_val, ctx=executor_ctx)
    valid_y_val = tvm.nd.array(valid_y_val, ctx=executor_ctx)

    # training loop
    lr = 1e-3
    # JIT compile sgd update ops
    W1_sgd_update_func = tvm_op.make_sgd_update(
        W1_val.shape, lr, tgt, tgt_host, "W1_sgd_update")
    b1_sgd_update_func = tvm_op.make_sgd_update(
        b1_val.shape, lr, tgt, tgt_host, "b1_sgd_update")
    time_measurements = []
    for i in range(num_epochs):
        print("epoch %d" % i)
        start_time = time.time()
        for minibatch_index in range(n_train_batches):
            minibatch_start = minibatch_index * batch_size
            minibatch_end = (minibatch_index + 1) * batch_size
            X_val =  tvm.nd.array(train_set_x[minibatch_start:minibatch_end].astype(np.float32), ctx=executor_ctx)
            y_val =  tvm.nd.array(convert_to_one_hot(train_set_y[minibatch_start:minibatch_end]).astype(np.float32), ctx=executor_ctx)
            loss_val, grad_W1_val, grad_b1_val, _ = executor.run(
                feed_dict = {X: X_val, y_: y_val, W1: W1_val, b1: b1_val})

            print("ep: {} minibatch_in {} loss {}".format(i, minibatch_index * batch_size, loss_val))

            # SGD update
            # W1_val = W1_val - lr * grad_W1_val
            # b1_val = b1_val - lr * grad_b1_val
            W1_sgd_update_func(W1_val, grad_W1_val, W1_val)
            b1_sgd_update_func(b1_val, grad_b1_val, b1_val)
        time_measurements.append(time.time() - start_time)
        if print_loss_val_each_epoch:
            print("loss = %f; Time taken this epoch = %f s" 
                % (np.asscalar(loss_val.asnumpy()), time_measurements[-1]))

    correct_predictions = []
    for minibatch_index in range(n_valid_batches):
        minibatch_start = minibatch_index * batch_size
        minibatch_end = (minibatch_index + 1) * batch_size
        valid_X_val.copyfrom(valid_set_x[minibatch_start:minibatch_end])
        valid_y_val.copyfrom(
            convert_to_one_hot(valid_set_y[minibatch_start:minibatch_end]))
        _, _, _, valid_y_predicted = executor.run(
            feed_dict={
                        X: valid_X_val,
                        y_: valid_y_val,
                        W1: W1_val,
                        b1: b1_val},
            convert_to_numpy_ret_vals=True)
        correct_prediction = np.equal(
            np.argmax(valid_y_val.asnumpy(), 1),
            np.argmax(valid_y_predicted, 1)).astype(np.float)
        correct_predictions.extend(correct_prediction)
    accuracy = np.mean(correct_predictions)
    # validation set accuracy=0.928200
    print("Validation set accuracy = %f" % accuracy)
    print("Average Time per Training Epoch = %f s" % np.mean(time_measurements))
    

def mnist_mlp(executor_ctx=None, num_epochs=10,
              print_loss_val_each_epoch=True):
    print("=== Build 3-layer MLP model...")

    tgt = 'opencl'
    tgt_host = 'llvm -target=aarch64-linux-android'

    W1 = ad.Variable(name="W1")
    W2 = ad.Variable(name="W2")
    W3 = ad.Variable(name="W3")
    b1 = ad.Variable(name="b1")
    b2 = ad.Variable(name="b2")
    b3 = ad.Variable(name="b3")
    X = ad.Variable(name="X")
    y_ = ad.Variable(name="y_")


    # relu(X W1+b1)
    z1 = ad.matmul_op(X, W1)
    z2 = z1 + ad.broadcastto_op(b1, z1)
    z3 = ad.relu_op(z2)

    # relu(z3 W2+b2)
    z4 = ad.matmul_op(z3, W2)
    z5 = z4 + ad.broadcastto_op(b2, z4)
    z6 = ad.relu_op(z5)

    # softmax(z5 W2+b2)
    z7 = ad.matmul_op(z6, W3)
    y = z7 + ad.broadcastto_op(b3, z7)

    loss = ad.softmaxcrossentropy_op(y, y_)

    grad_W1, grad_W2, grad_W3, grad_b1, grad_b2, grad_b3 = ad.gradients(
        loss, [W1, W2, W3, b1, b2, b3])
    executor = ad.Executor(
        [loss, grad_W1, grad_W2, grad_W3, grad_b1, grad_b2, grad_b3, y],
        ctx=executor_ctx)

    # Read input data
    datasets = load_mnist_data("mnist.pkl.gz")
    train_set_x, train_set_y = datasets[0]
    valid_set_x, valid_set_y = datasets[1]
    test_set_x, test_set_y = datasets[2]
    # Set up minibatch
    batch_size = 500
    n_train_batches = train_set_x.shape[0] // batch_size
    n_valid_batches = valid_set_x.shape[0] // batch_size

    print("Start training loop...")

    # Initialize parameters
    rand = np.random.RandomState(seed=123)
    W1_val = rand.normal(scale=0.1, size=(784, 256)).astype(np.float32)
    W2_val = rand.normal(scale=0.1, size=(256, 100)).astype(np.float32)
    W3_val = rand.normal(scale=0.1, size=(100, 10)).astype(np.float32)
    b1_val = rand.normal(scale=0.1, size=(256)).astype(np.float32)
    b2_val = rand.normal(scale=0.1, size=(100)).astype(np.float32)
    b3_val = rand.normal(scale=0.1, size=(10)).astype(np.float32)
    X_val = np.empty(shape=(batch_size, 784), dtype=np.float32)
    y_val = np.empty(shape=(batch_size, 10), dtype=np.float32)
    valid_X_val = np.empty(shape=(batch_size, 784), dtype=np.float32)
    valid_y_val = np.empty(shape=(batch_size, 10), dtype=np.float32)

    # wrap with tvm.nd.array
    W1_val = tvm.nd.array(W1_val, ctx=executor_ctx)
    W2_val = tvm.nd.array(W2_val, ctx=executor_ctx)
    W3_val = tvm.nd.array(W3_val, ctx=executor_ctx)
    b1_val = tvm.nd.array(b1_val, ctx=executor_ctx)
    b2_val = tvm.nd.array(b2_val, ctx=executor_ctx)
    b3_val = tvm.nd.array(b3_val, ctx=executor_ctx)
    X_val = tvm.nd.array(X_val, ctx=executor_ctx)
    y_val = tvm.nd.array(y_val, ctx=executor_ctx)
    valid_X_val = tvm.nd.array(valid_X_val, ctx=executor_ctx)
    valid_y_val = tvm.nd.array(valid_y_val, ctx=executor_ctx)

    # training loop
    lr = 1.0e-3
    # JIT compile sgd update ops
    W1_sgd_update_func = tvm_op.make_sgd_update(
        W1_val.shape, lr, tgt, tgt_host, "W1_sgd_update")
    W2_sgd_update_func = tvm_op.make_sgd_update(
        W2_val.shape, lr, tgt, tgt_host, "W2_sgd_update")
    W3_sgd_update_func = tvm_op.make_sgd_update(
        W3_val.shape, lr, tgt, tgt_host, "W3_sgd_update")
    b1_sgd_update_func = tvm_op.make_sgd_update(
        b1_val.shape, lr, tgt, tgt_host, "b1_sgd_update")
    b2_sgd_update_func = tvm_op.make_sgd_update(
        b2_val.shape, lr, tgt, tgt_host, "b2_sgd_update")
    b3_sgd_update_func = tvm_op.make_sgd_update(
        b3_val.shape, lr, tgt, tgt_host, "b3_sgd_update")
    time_measurements = []
    for i in range(num_epochs):
        print("epoch %d" % i)
        start_time = time.time()
        for minibatch_index in range(n_train_batches):
            start_time_mb = time.time()
            minibatch_start = minibatch_index * batch_size
            minibatch_end = (minibatch_index + 1) * batch_size
            X_val.copyfrom(train_set_x[minibatch_start:minibatch_end])
            y_val.copyfrom(
                convert_to_one_hot(train_set_y[minibatch_start:minibatch_end]))
            loss_val, grad_W1_val, grad_W2_val, grad_W3_val, \
                grad_b1_val, grad_b2_val, grad_b3_val, _ = executor.run(
                    feed_dict={
                        X: X_val,
                        y_: y_val,
                        W1: W1_val,
                        W2: W2_val,
                        W3: W3_val,
                        b1: b1_val,
                        b2: b2_val,
                        b3: b3_val})
            # SGD update
            # W1_val = W1_val - lr * grad_W1_val
            # W2_val = W2_val - lr * grad_W2_val
            # W3_val = W3_val - lr * grad_W3_val
            # b1_val = b1_val - lr * grad_b1_val
            # b2_val = b2_val - lr * grad_b2_val
            # b3_val = b3_val - lr * grad_b3_val
            W1_sgd_update_func(W1_val, grad_W1_val, W1_val)
            W2_sgd_update_func(W2_val, grad_W2_val, W2_val)
            W3_sgd_update_func(W3_val, grad_W3_val, W3_val)
            b1_sgd_update_func(b1_val, grad_b1_val, b1_val)
            b2_sgd_update_func(b2_val, grad_b2_val, b2_val)
            b3_sgd_update_func(b3_val, grad_b3_val, b3_val)

            print("ep: {} minibatch_in {} loss {}".format(i, minibatch_index * batch_size, loss_val), 'time taken', time.time() - start_time_mb)

        time_measurements.append(time.time() - start_time)
        if print_loss_val_each_epoch:
            print("loss = %f; Time taken this epoch = %f s" 
                % (np.asscalar(loss_val.asnumpy()), time_measurements[-1]))


    correct_predictions = []
    for minibatch_index in range(n_valid_batches):
        minibatch_start = minibatch_index * batch_size
        minibatch_end = (minibatch_index + 1) * batch_size
        valid_X_val.copyfrom(valid_set_x[minibatch_start:minibatch_end])
        valid_y_val.copyfrom(
            convert_to_one_hot(valid_set_y[minibatch_start:minibatch_end]))
        _, _, _, _, _, _, _, valid_y_predicted = executor.run(
            feed_dict={
                X: valid_X_val,
                y_: valid_y_val,
                W1: W1_val,
                W2: W2_val,
                W3: W3_val,
                b1: b1_val,
                b2: b2_val,
                b3: b3_val},
            convert_to_numpy_ret_vals=True)
        correct_prediction = np.equal(
            np.argmax(valid_y_val.asnumpy(), 1),
            np.argmax(valid_y_predicted, 1)).astype(np.float)
        correct_predictions.extend(correct_prediction)
    accuracy = np.mean(correct_predictions)
    # validation set accuracy=0.970800
    print("Validation set accuracy = %f" % accuracy)
    print("Average Time per Training Epoch = %f s" % np.mean(time_measurements))
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-m", "--model",
        help="Choose model: all, logreg, mlp", default="all")
    parser.add_argument(
        "-c", "--executor_context",
        help="Choose executor context: cpu, gpu", default="cpu")
    parser.add_argument(
        "-e", "--num_epoch",
        help="Provide number of epochs to train.", type=int, default=10)
    parser.add_argument(
        "-l", "--print_loss_val_each_epoch",
        help="Print loss value at the end of each epoch", action="store_true")
    args = parser.parse_args()

    models = []
    executor_ctx = None
    print_loss_val_each_epoch = False
    if args.model == "logreg":
        models = [mnist_logreg]
    elif args.model == "mlp":
        models = [mnist_mlp]
    elif args.model == "all":
        models = [mnist_logreg, mnist_mlp]

    if args.executor_context == "cpu":
        tgt = "llvm"
        tgt_host = "llvm"
    elif args.executor_context == "gpu":
        tgt = "cuda"
        tgt_host = "llvm"
        assert False, "cuda codegen not ready"
    elif args.executor_context == "adreno":
        tgt = "opencl"
        tgt_host = "llvm -target=aarch64-linux-android"

    # create context object
    # executor_ctx = tvm.context(tgt, 0)

    print_loss_val_each_epoch = True if args.print_loss_val_each_epoch \
                                     else False


    # TODO: change this
    executor_ctx = tvm_op.remote.cl(0)
    print(executor_ctx.device_name)

    num_epochs = args.num_epoch

    # matmul(executor_ctx)
    mnist_mlp(executor_ctx, num_epochs=5)
    # for m in models:
    #     m(executor_ctx, 1, print_loss_val_each_epoch=True)

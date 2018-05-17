from tvm.contrib import rpc

REMOTE = rpc.connect("0.0.0.0", 9090, key="android")

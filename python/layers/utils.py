from tvm.contrib import rpc

REMOTE = rpc.connect("0.0.0.0", 9091, key="android")

# reproduce the crossed-line bug
import argparse
import logging
import subprocess
import time
import os
import random

import mxnet as mx

import byteps.mxnet as bps
from byteps.mxnet.ops import size, local_size, rank, local_rank
from byteps.mxnet.ops import byteps_push_pull, byteps_declare_tensor

def main():
    bps.init()

    ctx_idx = int(os.environ.get('NVIDIA_VISIBLE_DEVICES', '0'))
    context = mx.gpu(ctx_idx)
    # context = mx.cpu()

    tensor_1 = mx.nd.array([(-1) ** rank()], ctx=context)

    byteps_declare_tensor("tensor_1")

    for i in range(1, 100):

        tensor_1[:] = i * ((-1) ** rank())

        mx.nd.waitall()
        print("worker %d after pushpull tensor_1=" % (rank()), tensor_1.asnumpy())

        byteps_push_pull(tensor_1, name="tensor_1", priority=-1)

        mx.nd.waitall()
        # sometimes the sum is not zero
        # when the bug happens, the server side receive the same value from both workers
        print("worker %d after pushpull tensor_1=" % (rank()), tensor_1.asnumpy())
        assert tensor_1.asnumpy() == 0, "bug happened!"
        

if __name__ == '__main__':
    main()

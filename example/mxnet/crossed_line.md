# Zeno-PS Instructions

## MXNet

## Setup

```
pip uninstall mxnet-cu100 -y
pip install --pre --upgrade 'mxnet-cu100==1.7.0b20200728' -f https://dist.mxnet.io/python --user
pip install gluoncv
pip uninstall byteps -y
mkdir -p ~/src/bps
cd ~/src/bps
rm -rf byteps
git clone https://github.com/xcgoner/byteps.git -b crossed_line_bug --recursive
cd byteps/
python3 setup.py install --user
```

## Distributed Training (TCP)

Let's say you have 2 workers (1 GPU per worker), and one validator (1 GPU per worker). For simplicity we use one server. 
Note that we put all the processes (server, scheduler, worker) in the same machine, and set BYTEPS_FORCE_DISTRIBUTED=1.

The bug randomly happens.
Sometimes in a single run of crossed_line.py, the bug does not happen.
Note that when there is sufficient delay between server and worker (e.g., add some random sleep in crossed_line.py), the bug will be extremely rare. 
Maybe that's why people do not observer it in multi-node settings.

Also, to confirm that the bug is actually on the server side, turn on "PS_VERBOSE=1" for the server, and observe the actual value received in server.cc

------------

### Train with 2 workers:

For the scheduler:
```
export DMLC_NUM_WORKER=2
export DMLC_ROLE=scheduler
export DMLC_NUM_SERVER=1
export DMLC_PS_ROOT_URI=127.0.0.1
export DMLC_PS_ROOT_PORT=1234
export PS_VERBOSE=0
export BYTEPS_FORCE_DISTRIBUTED=1

python3 /home/ubuntu/src/bps/byteps/launcher/launch.py
```

For the server:
```
export DMLC_NUM_WORKER=2
export DMLC_ROLE=server
export DMLC_NUM_SERVER=1
export DMLC_PS_ROOT_URI=127.0.0.1
export DMLC_PS_ROOT_PORT=1234
export PS_VERBOSE=0
export BYTEPS_FORCE_DISTRIBUTED=1

export vCPUs=`cat /proc/cpuinfo | grep processor | wc -l`
export BYTEPS_SERVER_ENGINE_THREAD=$((vCPUs / 2))

python3 /home/ubuntu/src/bps/byteps/launcher/launch.py
```


For worker-0:
```
export NVIDIA_VISIBLE_DEVICES=0
export DMLC_WORKER_ID=0
export DMLC_NUM_WORKER=2
export DMLC_ROLE=worker
export DMLC_NUM_SERVER=1
export DMLC_PS_ROOT_URI=127.0.0.1
export DMLC_PS_ROOT_PORT=1234
export PS_VERBOSE=0
export BYTEPS_FORCE_DISTRIBUTED=1

python3 /home/ubuntu/src/bps/byteps/launcher/launch.py python3 /home/ubuntu/src/bps/byteps/example/mxnet/crossed_line.py

```

For worker-1:

```
export NVIDIA_VISIBLE_DEVICES=1
export DMLC_WORKER_ID=1
export DMLC_NUM_WORKER=2
export DMLC_ROLE=worker
export DMLC_NUM_SERVER=1
export DMLC_PS_ROOT_URI=127.0.0.1
export DMLC_PS_ROOT_PORT=1234
export PS_VERBOSE=0
export BYTEPS_FORCE_DISTRIBUTED=1

python3 /home/ubuntu/src/bps/byteps/launcher/launch.py python3 /home/ubuntu/src/bps/byteps/example/mxnet/crossed_line.py

```




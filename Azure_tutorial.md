# Tutorial: running jobs on GPUs with Tensorflow, Docker, and Azure

This tutorial will guide you through running Tensorflow on GPUs with Docker and Azure. The author took 6h30 to work through all the issues and kinks, so we hope that you will be able to do it in much less time (about 30 minutes).

## Setup

Set up an Azure Virtual Machine with `NVIDIA GPU Cloud Image` on hardware
`Standard NV6`. SSH into it. 

## Check NVIDIA installation

These directions follow [TensorFlow Docker](https://www.tensorflow.org/install/docker#gpu_support).

To check the NVIDIA installation on the virtual machine, run these commands:

```
$ lspci | grep -i nvidia
db4d:00:00.0 VGA compatible controller: NVIDIA Corporation GM204GL [Tesla M60] (rev a1)
```

And run also this command, which tells Docker to start a container from the image `nvidia/cuda` and run the `nvidia-smi` command inside it. SMI stands for System Management Interface.

```
$ docker run --runtime=nvidia --rm nvidia/cuda nvidia-smi
Tue Jan 22 16:50:49 2019       
+-----------------------------------------------------------------------------+
| NVIDIA-SMI 410.48                 Driver Version: 410.48                    |
|-------------------------------+----------------------+----------------------+
| GPU  Name        Persistence-M| Bus-Id        Disp.A | Volatile Uncorr. ECC |
| Fan  Temp  Perf  Pwr:Usage/Cap|         Memory-Usage | GPU-Util  Compute M. |
|===============================+======================+======================|
|   0  Tesla M60           On   | 0000D6FC:00:00.0 Off |                   0* |
| N/A   36C    P8    13W / 150W |      0MiB /  7618MiB |      1%      Default |
+-------------------------------+----------------------+----------------------+
                                                                               
+-----------------------------------------------------------------------------+
| Processes:                                                       GPU Memory |
|  GPU       PID   Type   Process name                             Usage      |
|=============================================================================|
|  No running processes found                                                 |
+-----------------------------------------------------------------------------+
```

Notice that the top box lists only one GPU: the "six" in `NV6` stands for 6
virtual CPUs, not for 6 GPUs. Later we will set up a machine with multiple GPUs
and send different jobs to them.

Notice also how the second box lists no running processes.

If these two commands fail, you may be unable to proceed with this tutorial. We list two pitfalls in the rest of the tutorial.

## Pitfall #1: `driver>=384,driver<385`

Go back to the beginning of the tutorial and set up the machines with the `Data Science
Virtual Machine for Linux (Ubuntu)` image instead of the `NVIDIA GPU Cloud
Image`.

As of January 2019, you will need to run `docker` and `nvidia-docker` commands
as root. You will also get a cryptic error at the second check for NVIDIA:

```
$ sudo docker run --runtime=nvidia --rm nvidia/cuda nvidia-smi
docker: Error response from daemon: OCI runtime create failed: container_linux.go:348: starting container process caused "process_linux.go:402: container init caused \"process_linux.go:385: running prestart hook 1 caused \\\"error running hook: exit status 1, stdout: , stderr: exec command: [/usr/bin/nvidia-container-cli --load-kmods configure --ldconfig=@/sbin/ldconfig.real --device=all --compute --utility --require=cuda>=10.0 brand=tesla,driver>=384,driver<385 --pid=17595 /data/docker/overlay2/abe7bc97a745273f37a19f74b11ec5f42e9239e6a59f7a9e48543127d04ecc5d/merged]\\\\nnvidia-container-cli: requirement error: unsatisfied condition: driver < 385\\\\n\\\"\"": unknown.
```

I don't know why this image fails. I raised the issue on [SuperUser](https://superuser.com/questions/1396689/unable-to-run-nvidia-docker-image-on-azure/1396690#1396690), which may have updates.


## Test the GPU with a TensorFlow job

Save the following code as `DOCKERFILE` and on the machine (e.g. save locally
and copy with `scp DOCKERFILE username@0.0.0.0:`, where `0.0.0.0` is the IP of
the virtual machine):

```docker
FROM tensorflow/tensorflow:latest-gpu-py3

CMD python -c "import tensorflow as tf; tf.enable_eager_execution(); print(tf.reduce_sum(tf.random_normal([20000, 60000])))"
```

The first line sets the starting image to the latest Tensorflow GPU-enabled image for Python 3. The second line indicates the command to run (which is really just a check that TensorFlow is installed and that takes a long time to do so).

Before building the image and running the container, confirm that the GPU is idle with `nvidia-smi`:

```
$ nvidia-smi
Tue Jan 22 17:02:22 2019       
+-----------------------------------------------------------------------------+
| NVIDIA-SMI 410.48                 Driver Version: 410.48                    |
|-------------------------------+----------------------+----------------------+
| GPU  Name        Persistence-M| Bus-Id        Disp.A | Volatile Uncorr. ECC |
| Fan  Temp  Perf  Pwr:Usage/Cap|         Memory-Usage | GPU-Util  Compute M. |
|===============================+======================+======================|
|   0  Tesla M60           On   | 0000D6FC:00:00.0 Off |                   0* |
| N/A   34C    P8    13W / 150W |      0MiB /  7618MiB |      0%      Default |
+-------------------------------+----------------------+----------------------+
                                                                               
+-----------------------------------------------------------------------------+
| Processes:                                                       GPU Memory |
|  GPU       PID   Type   Process name                             Usage      |
|=============================================================================|
|  No running processes found                                                 |
+-----------------------------------------------------------------------------+
```

In the virtual machine, build the docker image, giving it a name such as `tmp`, and start it in a container:

```bash
docker build -f DOCKERFILE -t tmp .
docker run --runtime=nvidia -d tmp
```

Quickly check the output of `nvidia-smi`, as this job takes only a few seconds
before throwing a `ResourceExhaustedError`:

```
$ nvidia-smi
Tue Jan 22 16:57:50 2019       
+-----------------------------------------------------------------------------+
| NVIDIA-SMI 410.48                 Driver Version: 410.48                    |
|-------------------------------+----------------------+----------------------+
| GPU  Name        Persistence-M| Bus-Id        Disp.A | Volatile Uncorr. ECC |
| Fan  Temp  Perf  Pwr:Usage/Cap|         Memory-Usage | GPU-Util  Compute M. |
|===============================+======================+======================|
|   0  Tesla M60           On   | 0000D6FC:00:00.0 Off |                   0* |
| N/A   34C    P0    36W / 150W |   7245MiB /  7618MiB |     62%      Default |
+-------------------------------+----------------------+----------------------+
                                                                               
+-----------------------------------------------------------------------------+
| Processes:                                                       GPU Memory |
|  GPU       PID   Type   Process name                             Usage      |
|=============================================================================|
|    0     11995      C   python                                      7234MiB |
+-----------------------------------------------------------------------------+
```

GPU 0 is running the Python command with a certain process ID and a certain GPU
memory usage.


If you look at the logs, you will see that the python command can see GPU 0:
```
$ docker logs <containerID>
...
2019-01-22 12:06:53.493777: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1511] Adding visible gpu devices: 0
2019-01-22 12:06:53.801888: I tensorflow/core/common_runtime/gpu/gpu_device.cc:982] Device interconnect StreamExecutor with strength 1 edge matrix:
2019-01-22 12:06:53.801958: I tensorflow/core/common_runtime/gpu/gpu_device.cc:988]      0 
2019-01-22 12:06:53.801968: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1001] 0:   N 
2019-01-22 12:06:53.802178: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1115] Created TensorFlow device (/job:localhost/replica:0/task:0/device:GPU:0 with 7098 MB memory) -> physical GPU (device: 0, name: Tesla M60, pci bus id: d6fc:00:00.0, compute capability: 5.2)
...
```

## Pitfall #2: `importerror: libcuda.so.1`

Now run the same image with a simpler command:

```
$ docker run tmp
Traceback (most recent call last):
...
ImportError: libcuda.so.1: cannot open shared object file: No such file or directory

...

Failed to load the native TensorFlow runtime.

See https://www.tensorflow.org/install/errors

for some common reasons and solutions.  Include the entire stack trace
above this error message when asking for help.
```

The problem is running this command with `docker` instead of either `nvidia-docker`
or `docker --runtime=nvidia`. When running jobs on GPUs, you have to use one of
the latter forms. Another difference is running it in attached mode so you could see the output (the flag `-d` in the previous commands ran the container in detached mode).

## Set up multiple GPUs

Now create a new virtual machine on Azure with multiple GPUs,
e.g. `Standard_NV12` or `Standard_NV24`, which have 12 or 24 virtual CPUs. The
baseline quota of virtual CPUs is 24, so you may need to delete some
resources. Then SSH into the machine.

## Running a TensorFlow job on multiple GPUs

Now we set up a real machine learning job: the official TensorFlow MNIST model. Save this code into `DOCKERFILE` on the virtual machine:

```
FROM tensorflow/tensorflow:latest-gpu-py3

### get git
RUN apt-get update; apt-get -y install git

# Install additional software from TensorFlow Probability
RUN git clone https://github.com/tensorflow/models.git
ENV PYTHONPATH="$PYTHONPATH:./models"

# Add this required library, which is not included in standard Python
RUN pip3 install requests

# Compute the MNIST official model
CMD python models/official/mnist/mnist.py
```

Then build the image, giving it a name like `mnist`, and run it in a container calling it
by that name:

```bash
docker build -f DOCKERFILE -t mnist .
nvidia-docker run -d mnist
```

Check the GPU:

```
$ nvidia-smi
Tue Jan 22 17:14:35 2019       
+-----------------------------------------------------------------------------+
| NVIDIA-SMI 410.48                 Driver Version: 410.48                    |
|-------------------------------+----------------------+----------------------+
| GPU  Name        Persistence-M| Bus-Id        Disp.A | Volatile Uncorr. ECC |
| Fan  Temp  Perf  Pwr:Usage/Cap|         Memory-Usage | GPU-Util  Compute M. |
|===============================+======================+======================|
|   0  Tesla M60           On   | 00004EE4:00:00.0 Off |                  Off |
| N/A   33C    P0    38W / 150W |   7730MiB /  8129MiB |      4%      Default |
+-------------------------------+----------------------+----------------------+
|   1  Tesla M60           On   | 00006DB1:00:00.0 Off |                  Off |
| N/A   30C    P0    35W / 150W |    147MiB /  8129MiB |      0%      Default |
+-------------------------------+----------------------+----------------------+
                                                                               
+-----------------------------------------------------------------------------+
| Processes:                                                       GPU Memory |
|  GPU       PID   Type   Process name                             Usage      |
|=============================================================================|
|    0     20876      C   python                                      7719MiB |
|    1     20876      C   python                                       136MiB |
+-----------------------------------------------------------------------------+
```

Notice that the same process ID is running on two GPUs, with an imbalance of
memory: one GPU is using a lot more memory than the other.

## Stop the container

Stop the container with `docker stop <containerID>`, or stop all containers with 

```bash
docker stop $(docker container ps -a -q)
```

You may want to stop all running containers before setting up new ones, so the GPU usage reflects your latest commands.

## Running on a single GPU

To run the same job on just one CPU, call `nvidia-docker run` passing
the `CUDA_VISIBLE_DEVICES` environment variable with the `-e` flag:

```
nvidia-docker run -d -e CUDA_VISIBLE_DEVICES='1' mnist
```

And now the container can only see GPU #1. Check that only GPU #1 has running processes:

```
$ nvidia-smi
Tue Jan 22 17:20:55 2019       
+-----------------------------------------------------------------------------+
| NVIDIA-SMI 410.48                 Driver Version: 410.48                    |
|-------------------------------+----------------------+----------------------+
| GPU  Name        Persistence-M| Bus-Id        Disp.A | Volatile Uncorr. ECC |
| Fan  Temp  Perf  Pwr:Usage/Cap|         Memory-Usage | GPU-Util  Compute M. |
|===============================+======================+======================|
|   0  Tesla M60           On   | 00004EE4:00:00.0 Off |                  Off |
| N/A   44C    P8    15W / 150W |      0MiB /  8129MiB |      0%      Default |
+-------------------------------+----------------------+----------------------+
|   1  Tesla M60           On   | 00006DB1:00:00.0 Off |                  Off |
| N/A   40C    P0    35W / 150W |   7761MiB /  8129MiB |      0%      Default |
+-------------------------------+----------------------+----------------------+
                                                                               
+-----------------------------------------------------------------------------+
| Processes:                                                       GPU Memory |
|  GPU       PID   Type   Process name                             Usage      |
|=============================================================================|
|    1     27063      C   python                                      7750MiB |
+-----------------------------------------------------------------------------+
```

## Running same job on different GPUs

You may want to run the same job on different GPUs, e.g. to have different runs
of a simulation. For example, the above job running on two GPUs was making poor
use of one of them, so it may be better to have two jobs running fully on each
GPU than one job running on two GPUs. You can do this with two different values
for the `CUDA_VISIBLE_DEVICES` environment variable:

```
nvidia-docker run -d -e CUDA_VISIBLE_DEVICES='0' mnist
nvidia-docker run -d -e CUDA_VISIBLE_DEVICES='1' mnist
```

After a few seconds, you can see two different processes running on each GPU
with similar memory usage:

```
$ nvidia-smi
Tue Jan 22 17:30:25 2019       
+-----------------------------------------------------------------------------+
| NVIDIA-SMI 410.48                 Driver Version: 410.48                    |
|-------------------------------+----------------------+----------------------+
| GPU  Name        Persistence-M| Bus-Id        Disp.A | Volatile Uncorr. ECC |
| Fan  Temp  Perf  Pwr:Usage/Cap|         Memory-Usage | GPU-Util  Compute M. |
|===============================+======================+======================|
|   0  Tesla M60           On   | 00004EE4:00:00.0 Off |                  Off |
| N/A   48C    P0   106W / 150W |   7854MiB /  8129MiB |     51%      Default |
+-------------------------------+----------------------+----------------------+
|   1  Tesla M60           On   | 00006DB1:00:00.0 Off |                  Off |
| N/A   31C    P0    36W / 150W |   7762MiB /  8129MiB |      0%      Default |
+-------------------------------+----------------------+----------------------+
                                                                               
+-----------------------------------------------------------------------------+
| Processes:                                                       GPU Memory |
|  GPU       PID   Type   Process name                             Usage      |
|=============================================================================|
|    0      3302      C   python                                      7843MiB |
|    1      3555      C   python                                      7751MiB |
+-----------------------------------------------------------------------------+
```

## Alternative ways of running on different GPUs

You can also set an environment variable in your Docker file:

```
# Run only on one GPU
ENV CUDA_VISIBLE_DEVICES='1'
```

The disadvantage is that you will have one image for each GPU you want to run
on.

See more options in [this StackOverflow thread](https://stackoverflow.com/questions/30494050/how-do-i-pass-environment-variables-to-docker-containers).

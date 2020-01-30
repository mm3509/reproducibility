# Tutorial: running TensorFlow on macOS laptop

## Python 3.6.7

Currently, TensorFlow supports only Python 3.6, not 3.7 (see [GitHub issue](https://github.com/tensorflow/tensorflow/issues/17022)). Download [Python 3.6.7](https://www.python.org/downloads/release/python-367/) and install (requires admin rights).

## No GPU on macOS

[This page](https://www.tensorflow.org/install/source#common_installation_problems) on common installation problems mentions

> There is _no_ GPU support for macOS.

## Install TensorFlow

[These instructions](https://www.tensorflow.org/install/pip) guide the installation of TensorFlow. 

Verify your Python installation with

```bash
python3 --version
pip3 --version
```

Install `virtualenv` for current user

```bash
export PYTHONUSERBASE=$HOME
pip3 install virtualenv
```

Create a new virtual environment

```bash
virtualenv --system-site-packages -p python3 ./venv
```

### Pitfall #1

If this last command fails, follow [this suggestion](https://stackoverflow.com/questions/39964635/error-virtualenv-command-not-found-but-install-location-is-in-pythonpath#):

```bash
python3 -m virtualenv ./venv
```

### Pitfall #2

If you already have Python 3.7 installed, and get this error:
```
$ virtualenv --system-site-packages -p python3 ./venv
-bash: /usr/local/bin/virtualenv: /usr/local/opt/python/bin/python3.7: bad interpreter: No such file or directory
```

install `virtualenv` again with `pip`, uninstall it, and install it again with `pip3` (as in [this SuperUser thread](https://superuser.com/questions/1380418/python3-7-bad-interpreter-no-such-file-or-directory/1380419)).

### After virtual environment installation

Activate the environment with

```bash
source ./venv/bin/activate
```

Upgrade pip in this environment

```bash
pip3 install --upgrade pip
```

Install tensorflow in this environment:

```bash
pip3 install --upgrade tensorflow
```

Verify that TensorFlow is working by running:

```
python -c "import tensorflow as tf; tf.enable_eager_execution(); print(tf.reduce_sum(tf.random_normal([1000, 1000])))"
```

# How To Install TensorFlow on M1 Mac (The Easy Way)

https://developer.apple.com/metal/tensorflow-plugin/

**Please follow upper link to install tensorflow-macos for M1 OS X arm64 (Apple Silicon)**

## Requirements
Mac computers with Apple silicon or AMD GPUs
macOS 12.0 or later (Get the latest beta)
Python 3.8 or later
Xcode command-line tools: xcode-select --install

## Get started

1. Set up

arm64 : Apple silicon
[Download Conda environment](https://repo.anaconda.com/miniconda/Miniconda3-latest-MacOSX-arm64.sh)
```
bash ~/miniconda.sh -b -p $HOME/miniconda
source ~/miniconda/bin/activate
conda install -c apple tensorflow-deps
```

x86 : AMD
Virtual environment
```
python3 -m venv ~/venv-metal
source ~/venv-metal/bin/activate
python -m pip install -U pip
```

2. Install base TensorFlow

`python -m pip install tensorflow-macos`

3. Install tensorflow-metal plug-in

`python -m pip install tensorflow-metal`

4. Verify

You can verify using a simple script:

```
import tensorflow as tf

cifar = tf.keras.datasets.cifar100
(x_train, y_train), (x_test, y_test) = cifar.load_data()
model = tf.keras.applications.ResNet50(
    include_top=True,
    weights=None,
    input_shape=(32, 32, 3),
    classes=100,)

loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
model.compile(optimizer="adam", loss=loss_fn, metrics=["accuracy"])
model.fit(x_train, y_train, epochs=5, batch_size=64)
```

https://caffeinedev.medium.com/how-to-install-tensorflow-on-m1-mac-8e9b91d93706

https://naolin.medium.com/conda-on-m1-mac-with-miniforge-bbc4e3924f2b

https://github.com/tensorflow/tensorflow/issues/60081

•    tensorflow: For x86_64 processors on any operating system
•    tensorflow-macos: For macOS systems, using either x86_64 or arm64 processors  https://pypi.org/project/tensorflow-macos/
•    tensorflow-cpu-aws: For Linux aarch64 systems

https://zhuanlan.zhihu.com/p/349295868

https://github.com/dmlc/dgl/issues/3748

```
$ arch
arm64

$ python --version
Python 3.8.16

$ system_profiler SPHardwareDataType
Hardware:

    Hardware Overview:

      Model Name: MacBook Pro
      Model Identifier: MacBookPro18,1
      Model Number: MK183CH/A
      Chip: Apple M1 Pro
      Total Number of Cores: 10 (8 performance and 2 efficiency)
      Memory: 16 GB
      System Firmware Version: 8419.80.7
      OS Loader Version: 8419.80.7
      Serial Number (system): WY04R3FHVJ
      Hardware UUID: 6AFE7A62-128E-518F-8BC8-1C9399324AC3
      Provisioning UDID: 00006000-000C098C3EB8401E
      Activation Lock Status: Disabled
```

If you get a mistake: ERROR: grpcio-1.33.2-cp38-cp38-macosx_10_16_arm64.whl is not a supported wheel on this platform.
you need to uninstall conda and install: https://github.com/conda-forge/miniforge#miniforge3

OS X x86_64
https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-MacOSX-x86_64.sh

OS X arm64 (Apple Silicon) 
https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-MacOSX-arm64.sh

```
Chmod +x Miniforge3-MacOSX-arm64.sh
./Miniforge3-MacOSX-arm64.sh

==> For changes to take effect, close and re-open your current shell. <==

If you'd prefer that conda's base environment not be activated on startup, 
   set the auto_activate_base parameter to false: 

conda config --set auto_activate_base false

Thank you for installing Miniforge3!
```

```
➜  ~ source .zshrc
(base) ➜  ~ conda --version
conda 23.1.0
```

reinstall python
```
% conda uninstall python 

The following packages will be REMOVED:
  tensorboard-2.11.2-pyhd8ed1ab_0
  tensorboard-data-server-0.6.1-py38h23f6d3d_4
  tensorboard-plugin-wit-1.8.1-pyhd8ed1ab_0
  tensorflow-estimator-2.6.0-py38hddd8853_0
  
% conda install python=3.8.16
```

https://github.com/apple/tensorflow_macos/issues/153

```
conda install python=3.8.10
```

```
conda config --set auto_activate_base false
conda env create --file=environment.yml --name=test
conda activate test
```

For Apple Silicon (arm64):
```
pip install --upgrade --force --no-dependencies https://github.com/apple/tensorflow_macos/releases/download/v0.1alpha3/tensorflow_macos-0.1a3-cp38-cp38-macosx_11_0_arm64.whl https://github.com/apple/tensorflow_macos/releases/download/v0.1alpha3/tensorflow_addons_macos-0.1a3-cp38-cp38-macosx_11_0_arm64.whl
```
For X86:
```
pip install --upgrade --force --no-dependencies https://github.com/apple/tensorflow_macos/releases/download/v0.1alpha3/tensorflow_macos-0.1a3-cp38-cp38-macosx_11_0_x86_64.whl https://github.com/apple/tensorflow_macos/releases/download/v0.1alpha3/tensorflow_addons_macos-0.1a3-cp38-cp38-macosx_11_0_x86_64.whl 
```

```
(test) ➜  tensorflow-macos-setup pip install --upgrade --force --no-dependencies https://github.com/apple/tensorflow_macos/releases/download/v0.1alpha3/tensorflow_macos-0.1a3-cp38-cp38-macosx_11_0_arm64.whl https://github.com/apple/tensorflow_macos/releases/download/v0.1alpha3/tensorflow_addons_macos-0.1a3-cp38-cp38-macosx_11_0_arm64.whl
Collecting tensorflow-macos==0.1a3
  Downloading https://github.com/apple/tensorflow_macos/releases/download/v0.1alpha3/tensorflow_macos-0.1a3-cp38-cp38-macosx_11_0_arm64.whl (124.2 MB)
     ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 124.2/124.2 MB 10.8 MB/s eta 0:00:00
Collecting tensorflow-addons-macos==0.1a3
  Downloading https://github.com/apple/tensorflow_macos/releases/download/v0.1alpha3/tensorflow_addons_macos-0.1a3-cp38-cp38-macosx_11_0_arm64.whl (598 kB)
     ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 598.4/598.4 kB 8.2 MB/s eta 0:00:00
Installing collected packages: tensorflow-macos, tensorflow-addons-macos
Successfully installed tensorflow-addons-macos-0.1a3 tensorflow-macos-0.1a3
```

```
(test) ➜  tensorflow-macos-setup pip3 show tensorflow-macos
Name: tensorflow
Version: 0.1a3
Summary: TensorFlow is an open source machine learning framework for everyone.
Home-page: https://www.tensorflow.org/
Author: Google Inc.
Author-email: packages@tensorflow.org
License: Apache 2.0
Location: /Users/gavinxiang/miniforge3/envs/test/lib/python3.8/site-packages
Requires: absl-py, astunparse, flatbuffers, gast, google-pasta, grpcio, h5py, keras-preprocessing, numpy, opt-einsum, protobuf, six, tensorboard, tensorflow-estimator, termcolor, typing-extensions, wheel, wrapt
Required-by: 
```

## How to exit Python in the Terminal

You can type in `quit()` or `exit()` to exit out of Python while using the terminal on a Linux or macOS computer.

On Linux and macOS, you should be able to use the `CTRL + D` shortcut to exit the Python prompt.

```
python
import tensorflow
```

```
(test) ➜  tensorflow-macos-setup python
Python 3.8.16 | packaged by conda-forge | (default, Feb  1 2023, 16:01:13) 
[Clang 14.0.6 ] on darwin
Type "help", "copyright", "credits" or "license" for more information.
>>> import tensorflow
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
  File "/Users/gavinxiang/miniforge3/envs/test/lib/python3.8/site-packages/tensorflow/__init__.py", line 38, in <module>
    import six as _six
ModuleNotFoundError: No module named 'six'
>>> import tensorflow
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
  File "/Users/gavinxiang/miniforge3/envs/test/lib/python3.8/site-packages/tensorflow/__init__.py", line 38, in <module>
    import six as _six
ModuleNotFoundError: No module named 'six'

>>> 
(test) ➜  tensorflow-macos-setup pip install six
Collecting six
  Using cached six-1.16.0-py2.py3-none-any.whl (11 kB)
Installing collected packages: six
ERROR: pip's dependency resolver does not currently take into account all the packages that are installed. This behaviour is the source of the following dependency conflicts.
tensorflow-macos 0.1a3 requires absl-py~=0.10, which is not installed.
tensorflow-macos 0.1a3 requires astunparse~=1.6.3, which is not installed.
tensorflow-macos 0.1a3 requires flatbuffers~=1.12.0, which is not installed.
tensorflow-macos 0.1a3 requires gast==0.3.3, which is not installed.
tensorflow-macos 0.1a3 requires google-pasta~=0.2, which is not installed.
tensorflow-macos 0.1a3 requires grpcio~=1.32.0, which is not installed.
tensorflow-macos 0.1a3 requires h5py~=2.10.0, which is not installed.
tensorflow-macos 0.1a3 requires keras-preprocessing~=1.1.2, which is not installed.
tensorflow-macos 0.1a3 requires numpy~=1.19.2, which is not installed.
tensorflow-macos 0.1a3 requires opt-einsum~=3.3.0, which is not installed.
tensorflow-macos 0.1a3 requires protobuf~=3.13.0, which is not installed.
tensorflow-macos 0.1a3 requires tensorboard~=2.3, which is not installed.
tensorflow-macos 0.1a3 requires tensorflow-estimator~=2.3.0, which is not installed.
tensorflow-macos 0.1a3 requires termcolor~=1.1.0, which is not installed.
tensorflow-macos 0.1a3 requires typing-extensions~=3.7.4, which is not installed.
tensorflow-macos 0.1a3 requires wrapt~=1.12.1, which is not installed.
tensorflow-macos 0.1a3 requires six~=1.15.0, but you have six 1.16.0 which is incompatible.
Successfully installed six-1.16.0
```

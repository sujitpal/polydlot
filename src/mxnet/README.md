# MXNet

## Installation
   
Installation involves building from source. Has problems working with Anaconda Python and OpenCV. Anaconda Python installs OpenCV 3.1 and MXNet works with either OpenCV 3.2 or 2.4, otherwise will core dump. Followed instructions on the [MXNet Installation Guide](http://newdocs.readthedocs.io/en/latest/build.html) to build shared library and install the MXNet Python interface system-wide.

Followed it up with removing current OpenCV and installing the older 2.4 version.

    conda uninstall opencv
    conda install -c menpo opencv=2.4.11

## Documentation

* [API Documentation](http://mxnet.io/api/python/)

## Tutorials

* [MNIST Notebook on dmlc/mxnet-notebooks](https://github.com/dmlc/mxnet-notebooks/blob/master/python/tutorials/mnist.ipynb)


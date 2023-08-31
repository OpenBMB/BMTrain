.. bmtrain-doc documentation master file, created by
   sphinx-quickstart on Sat Mar  5 17:05:02 2022.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

BMTrain's Documentation!
=============================

**BMTrain** is an efficient large model training toolkit that can be used to train large models with tens of billions of parameters. It can train models in a distributed manner while keeping the code as simple as stand-alone training.

=======================================

.. toctree::
   :maxdepth: 2
   :caption: Getting Started

   notes/installation.md 
   notes/quickstart.md
   notes/tech.md 

.. toctree::
   :maxdepth: 2
   :caption: Package Reference

   api/bmtrain.rst
   api/bmtrain.benchmark.rst
   api/bmtrain.distributed.rst
   api/bmtrain.inspect.rst
   api/bmtrain.lr_scheduler.rst
   api/bmtrain.nccl.rst
   api/bmtrain.optim.rst

API
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

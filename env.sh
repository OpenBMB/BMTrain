#!/bin/bash

export PATH=/public/software/apps/deeplearning-depend/opencv-2.4.13.6-build/bin:$PATH
export LD_LIBRARY_PATH=/public/software/apps/deeplearning-depend/opencv-2.4.13.6-build/lib:$LD_LIBRARY_PATH
#openblas-0.3.7-build
export PATH=/public/software/apps/deeplearning-depend/openblas-0.3.7-build/bin:$PATH
export LD_LIBRARY_PATH=/public/software/apps/deeplearning-depend/openblas-0.3.7-build/lib:$LD_LIBRARY_PATH
#gflags-2.1
export LD_LIBRARY_PATH=/public/software/apps/deeplearning-depend/gflags-2.1/:$LD_LIBRARY_PATH
export PATH=/public/software/apps/deeplearning-depend/gflags-2.1/bin:$PATH
export C_INCLUDE_PATH=/public/software/apps/deeplearning-depend/gflags-2.1.2-build/include/:$C_INCLUDE_PATH
export CPLUS_INCLUDE_PATH=/public/software/apps/deeplearning-depend/gflags-2.1.2-build/include/:$CPLUS_INCLUDE_PATH
#glog-build
export PATH=/public/software/apps/deeplearning-depend/glog-build/bin:$PATH
export LD_LIBRARY_PATH=/public/software/apps/deeplearning-depend/glog-build/lib:$LD_LIBRARY_PATH
export C_INCLUDE_PATH=/public/software/apps/deeplearning-depend/glog-build/include/:$C_INCLUDE_PATH
export CPLUS_INCLUDE_PATH=/public/software/apps/deeplearning-depend/glog-build/include/:$CPLUS_INCLUDE_PATH
#lmdb-0.9.24-build
export PATH=/public/software/apps/deeplearning-depend/lmdb-0.9.24-build/bin:$PATH
export LD_LIBRARY_PATH=/public/software/apps/deeplearning-depend/lmdb-0.9.24-build/lib:$LD_LIBRARY_PATH

module switch compiler/dtk/22.10.1 
export MIOPEN_SYSTEM_DB_PATH=/opt/dtk/miopen/share/miopen/db

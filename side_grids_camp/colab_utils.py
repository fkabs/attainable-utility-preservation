# Check if GPU is enabled. Note that this can cause Blas errors if uncommented(!)
"""
device_name = tf.test.gpu_device_name()

if device_name != '/device:GPU:0':
  raise SystemError('GPU device not found')

print('Found GPU at: {}'.format(device_name))
"""

# Get our repo and enable pycolab in colab
!git clone --recursive https://github.com/side-grids/ai-safety-gridworlds
os.chdir("ai-safety-gridworlds")
!cd ai_safety_gridworlds; git submodule init; git submodule update

# Check:
from ai_safety_gridworlds.environments.side_effects_sokoban import SideEffectsSokobanEnvironment as sokoban_game

%matplotlib inline


###################################################################################
# How much GPU is Google giving me?
!ln -sf /opt/bin/nvidia-smi /usr/bin/nvidia-smi
!pip install gputil
!pip install psutil
!pip install humanize

import psutil
import humanize
import os
import GPUtil as GPU

GPUs = GPU.getGPUs()
gpu = GPUs[0]

def printm():
  process = psutil.Process(os.getpid())
  print("Gen RAM Free: " + humanize.naturalsize( psutil.virtual_memory().available ), \
        " I Proc size: " + humanize.naturalsize( process.memory_info().rss))
  print("GPU RAM Free: {0:.0f}MB | Used: {1:.0f}MB | Util {2:3.0f}% | Total {3:.0f}MB".format(gpu.memoryFree, \
                                                          gpu.memoryUsed, gpu.memoryUtil*100, gpu.memoryTotal))

printm()

##################################################################################
# You might also get Blas errors after a while; means multiple of your processes are locking the GPU

!ps aux | grep "python"
!kill -9 PID

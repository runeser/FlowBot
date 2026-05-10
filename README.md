may10 2026
# Optical Flow Benchmark on Raspberry Pi 4
A lightweight optical flow processing with motion deriation, and benchmarking system designed 
for Raspberry Pi 4, using Middlebury dataset for validation.

The program expects frames of 640 by 480, and a framerate of 30 fps (or lower).



## Prerequisites
- Raspberry Pi 4 (2GB/4GB/8GB RAM recommended), or newer
- Raspberry Pi OS (64-bit recommended)
- Python 3.7+
- ~2GB free storage for datasets

## Setup

```bash
sudo apt update
sudo apt upgrade -y
```

#Create virtual environmant
```bash
sudo apt-get install python3-venv
python -m venv venv
```

or 


#Activate virtual environmen
```bash
.venv\Scripts\activate
```

#Install dependencies
```bash
pip install opencv-python numpy matplotlib
```



#Execute demo
#without resource tracking
```bash
python demo.py
```
or
#with resource tracking
```bash
sudo apt install time
/usr/bin/time -f "CPU: %P\nMax Memory: %M KB\nElapsed: %e sec\nExit: %x" timeout 15s python demo.py
```

## Benchmark
Go to https://vision.middlebury.edu/flow/data/
Download and unzip  
other-gray-allframes.zap
other-gt-flow.zip
datasets
Rename zip files as MB-frames and MB-flow respectively


## Create the data folder
Create a folder named 'data/' 
Place the MB-frames and MB-flow in the data folder.


## Execute dataset_test.py
```bash
python dataset_test.py
```

## Perform other test to your liking


## Future work
Rewrite code to work in c or c++ instead of python.

Extend the codebase with command to a micro-controller, located on the robot
Interpret motion, based on the magnitudes, and transform this to command 
    such that the robot-arm moves when really close to an objecg








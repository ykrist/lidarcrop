# About
`lidarcrop.py` is small Python script for trimming and cleaning airborne LIDAR point-cloud data in the form of PLY/LAS files.  It was written to work with output from HoverMap and consequently, it requires both a PLY point-cloud file and a trajectory ASCII file as input.  The output is a LAS file (optional) and a PLY file both containing the output point-cloud, the latter of which will contain additional information such as angle of incidence for each point, relative to the camera position.  The ASCII file should have the header format:
```
%time x y z
```
Extra columns will be ignored.  

This code in this repository is used in *Modelling the effects of fundamental UAV flight parameters on LiDAR point clouds to facilitate objectives-based planning* (2019) (https://doi.org/10.1016/j.isprsjprs.2019.01.020).

# Requirements
- Python 2.7/Anaconda with relevant packages (see `conda_req.txt` and `pip_req.txt`).

# Usage
Usage details can printed using
```bash
python lidarcrop.py -h
```

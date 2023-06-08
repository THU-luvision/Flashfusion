# FlashFusion: An efficient dense 3D reconstruction that only relies on CPU computing.

## Introduction
This is the official code repository for FlashFusion, an efficient dense 3D reconstruction system that only relies on CPU computing.

This is a project from LuVision SIGMA, Tsinghua University. Visit our website for more interesting works: http://www.luvision.net/

## License
This project is released under the [GPLv3 license](LICENSE). We only allow free use for academic use. For commercial use, please contact us to negotiate a different license by: `fanglu at tsinghua.edu.cn`

## Citing

If you find our code useful, please kindly cite the following papers:

```bibtex
@inproceedings{han2018flashfusion,
  title={FlashFusion: Real-time Globally Consistent Dense 3D Reconstruction using CPU Computing.},
  author={Han, Lei and Fang, Lu},
  booktitle={Robotics: Science and Systems},
  volume={1},
  number={6},
  pages={7},
  year={2018}
}
```
```bibtex
@ARTICLE{8606275,
  author={Han, Lei and Xu, Lan and Bobkov, Dmytro and Steinbach, Eckehard and Fang, Lu},
  journal={IEEE Transactions on Robotics}, 
  title={Real-Time Global Registration for Globally Consistent RGB-D SLAM}, 
  year={2019},
  volume={35},
  number={2},
  pages={498-508},
  doi={10.1109/TRO.2018.2882730}}
```


## Environment setup

### Preliminary Requirements:
* Ubuntu 16.04/18.04
* Intel cpu 

### Compile
```bash
source prepare.sh
mkdir build
cd build
cmake ..
make -j
```

### Data preparation
The format of offline dataset follows TUM RGBD Dataset. 
Example sequences can be downloaded at http://221.228.239.253:81/opensource_data/FlashFusion/dataset.zip 
Put the dataset at `dataset/`, organized as 
```
dataset
|---- iclnuim
|---- tum
|---- xtion
```

### Usage 
```
sudo ./FlashFusion sensor_type calib_file voxel_size dataset_path

sensor_type: 0:dataset 1:realtime openni_camera(such as xtion) 2: realtime realsense_camera

Example:
  sudo ./FlashFusion 0 ../param/calib_tum.txt 0.005 ../dataset/tum/fr1_desk/
  sudo ./FlashFusion 0 ../param/calib_icl.txt 0.005 ../dataset/iclnuim/icl0n/
  sudo ./FlashFusion 0 ../param/calib_xtion.txt 0.005 ../dataset/xtion/

  sudo ./FlashFusion 1 ../param/calib_xtion.txt 0.005

```

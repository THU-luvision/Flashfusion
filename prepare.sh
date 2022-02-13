# install basic dependency
sudo apt-get install build-essential pkg-config cmake  libwxgtk3.0-dev libftdi-dev freeglut3-dev  zlib1g-dev libusb-1.0-0-dev libudev-dev libfreenect-dev  libdc1394-22-dev libavformat-dev libswscale-dev
sudo apt-get install libassimp-dev libjpeg-dev libgtest-dev libeigen3-dev libsuitesparse-dev libpcap-dev libsuitesparse-dev  build-essential cmake git ffmpeg libopencv-dev libgtk-3-dev  python-numpy python3-numpy libdc1394-22 libdc1394-22-dev 
sudo apt-get install libjpeg-dev libpng-dev libtiff5-dev libjasper-dev libavcodec-dev  libavformat-dev libswscale-dev libxine2-dev libgstreamer1.0-dev  libgstreamer-plugins-base1.0-dev libv4l-dev libtbb-dev qtbase5-dev  libfaac-dev libmp3lame-dev libopencore-amrnb-dev
sudo apt-get install libopencore-amrwb-dev libtheora-dev libvorbis-dev libxvidcore-dev  x264 v4l-utils unzip libglew-dev libpython2.7-dev ffmpeg libavcodec-dev libavutil-dev libavformat-dev libswscale-dev libglfw3-dev 
sudo apt-get install libxmu-dev libxi-dev libboost-all-dev libxmu-dev libxi-dev qtbase5-dev -y g++ python  doxygen graphviz openjdk-8-jdk 

# openni support
sudo apt-get install libopenni2-dev
# test install result 
# pkg-config --modversion libopenni2
sudo ln -s /lib/x86_64-linux-gnu/libudev.so.1.6.4 /lib/x86_64-linux-gnu/libudev.so.0


cd third_party_library
unzip opencv-3.3.0.zip
tar -xzf Sophus.tar.gz
unzip librealsense-master.zip
unzip Pangolin-master.zip

# opencv
cd opencv-3.3.0
mkdir build
cd build
cmake -D CMAKE_BUILD_TYPE=RELEASE -DBUILD_opencv_stitching=OFF -D CMAKE_INSTALL_PREFIX=/usr/local -D WITH_TBB=ON -D WITH_V4L=ON -D WITH_QT=ON -D WITH_OPENGL=ON -D WITH_CUDA=OFF -DENABLE_PRECOMPILED_HEADERS=OFF -D CMAKE_BUILD_TYPE=RELEASE -D BUILD_DOCS=OFF -D BUILD_PERF_TESTS=OFF -D BUILD_TESTS=OFF -D BUILD_TIFF=ON  ..
make -j

# pangolin
cd ../../Pangolin-master
mkdir build
cd build
cmake .. 
make -j

# sophus 
cd ../../Sophus
mkdir build
cd build 
cmake .. 
make -j

# librealsense
cd ../../librealsense-master
mkdir build
cd build 
cmake .. 
make -j

# chisel
cd ../../../CHISEL
mkdir build
cd build 
cmake ../src
make -j
cd ../..




#ifndef FRAME_H
#define FRAME_H

#include <string>
#include <opencv/cv.h>
#include <opencv2/opencv.hpp>
#include <Eigen/Eigen>
#include <Eigen/StdVector>
#include <memory>

#include <sophus/se3.hpp>


typedef Eigen::Vector3i ChunkID;
typedef std::vector<ChunkID, Eigen::aligned_allocator<ChunkID> > ChunkIDList;
typedef std::vector<Eigen::Vector3f , Eigen::aligned_allocator<Eigen::Vector3d> > Point3fList;
typedef std::vector<Eigen::Vector3d , Eigen::aligned_allocator<Eigen::Vector3d> > Point3dList;
typedef std::vector<Sophus::SE3d , Eigen::aligned_allocator<Eigen::Vector3d> > PoseSE3dList;
typedef Sophus::SE3d PoseSE3d;



inline Eigen::Vector3d applyPose( const Sophus::SE3d &pose, const Eigen::Vector3d &point )
{
    return pose.so3() * point + pose.translation();
}


class Frame
{
public:
EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  // dense scene info
  int frame_index;
  cv::Mat rgb;
  cv::Mat depth;
  cv::Mat refined_depth;
  cv::Mat normal_map;
  cv::Mat weight;
  cv::Mat colorValidFlag;
  // sparse feature info
  std::vector<cv::KeyPoint > keypoints;
  cv::Mat descriptor;
  Point3dList local_points;

  float bluriness;

  PoseSE3dList pose_sophus; // pose_sophus[0] for current pose, pose_sophus[1] for next pose
  // time stamp
  double time_stamp;

  int tracking_success;
  int blur_flag;
  int is_keyframe;
  int is_fixed_frame;
  int origin_index;
  float depth_scale;


  //for tsdf fusion
  ChunkIDList validChunks;
  std::vector<void *> validChunksPtr;

  int GetOccupiedMemorySize()
  {
//      printf("memory occupied: %d %d %d %d      %d %d %d %d     %d %d %d %d\r\n",
//             (rgb.datalimit - rgb.data),
//             (depth.datalimit - depth.data) ,
//             (refined_depth.datalimit - refined_depth.data) ,
//             (normal_map.datalimit - normal_map.data) ,

//             (weight.datalimit - weight.data) ,
//             (descriptor.datalimit - descriptor.data) ,
//             keypoints.size() * sizeof(cv::KeyPoint) ,
//             feature_tracked_flag.size() * sizeof(unsigned char) ,

//             local_points.size() * sizeof(Eigen::Vector3d) ,
//             validChunks.size() * sizeof(ChunkID) ,
//             pose_sophus.size() * sizeof(Sophus::SE3d),
//             validChunksPtr.size() * sizeof(void *));
      return ( (rgb.datalimit - rgb.data) +
               (depth.datalimit - depth.data) +
               (refined_depth.datalimit - refined_depth.data) +
               (normal_map.datalimit - normal_map.data) +
               (weight.datalimit - weight.data) +
               (descriptor.datalimit - descriptor.data) +
               keypoints.size() * sizeof(cv::KeyPoint) +
               local_points.size() * sizeof(Eigen::Vector3d) +
               validChunks.size() * sizeof(ChunkID) +
               pose_sophus.size() * sizeof(Sophus::SE3d)+
               validChunksPtr.size() * sizeof(void *) +
               (colorValidFlag.datalimit - colorValidFlag.data)
                              );
  }

  // preserve feature/rgb/depth
  void clear_keyframe_memory()
  {
      depth.release();
      weight.release();
      normal_map.release();
  }

  // preserve local depth
  void clear_redudent_memoery()
  {
      rgb.release();
      colorValidFlag.release();
      depth.release();
      weight.release();
      normal_map.release();
      keypoints.clear();
      descriptor.release();
      local_points.clear();
  }

  // remove frames totally
  void clear_memory()
  {
    rgb.release();
    colorValidFlag.release();
    depth.release();
    refined_depth.release();
    weight.release();
    normal_map.release();
    keypoints.clear();
    descriptor.release();
    local_points.clear();
  }

  Eigen::Vector3d localToGlobal(const Eigen::Vector3d &point)
  {
      return pose_sophus[0].so3() * point + pose_sophus[0].translation();
  }

	Frame()
	{
		frame_index = 0;
		is_fixed_frame = 0;
		origin_index = 1e8;	// no origin by default
		keypoints.clear(); 
        descriptor.release();
		rgb.release();
		depth.release();
        refined_depth.release();
        local_points.clear();
		tracking_success = 0;
		blur_flag = 0;
        is_keyframe = 0;

        pose_sophus.push_back(Sophus::SE3d());
        pose_sophus.push_back(Sophus::SE3d());
	}
};



#endif

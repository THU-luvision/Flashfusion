#ifndef MULTI_VIEW_GEOMETRY_H
#define MULTI_VIEW_GEOMETRY_H

#include "frame.h"
#include "MILD/loop_closure_detector.hpp"
#include "MILD/BayesianFilter.hpp"
#include "MILD/sparse_match.hpp"
#include <string>
#include <xmmintrin.h>
#include <math.h>
#include <iostream>

#define HISTO_LENGTH 30
#define Truncation_ICP_POINTS 1000
#define minimum_3d_correspondence 30

#define USE_NORMAL_WEIGHT 1







namespace MultiViewGeometry
{

  class CameraPara {
  public:
    CameraPara(){}

    float GetFx()    { return c_fx; }
    float GetFy()    { return c_fy; }
    float GetCx()    { return c_cx; }
    float GetCy()    { return c_cy; }
    int GetWidth()    { return width; }
    int GetHeight()    { return height; }

    int width;
    int height;
    float c_fx;
    float c_fy;
    float c_cx;
    float c_cy;
    float d[5];		// distortionparameters
    float depth_scale;
    float maximum_depth;

  };

  class GlobalParameters
  {
  public:
    float salient_score_threshold;
    float minimum_disparity;
    float ransac_maximum_iterations;
    float reprojection_error_3d_threshold;
    float reprojection_error_2d_threshold;
    int use_fine_search;
    int maximum_local_frame_match_num;
    int remove_outlier_keyframes;
    int keyframe_minimum_distance;
    int maximum_keyframe_match_num;
    int save_ply_files;
    int debug_mode;
    int max_feature_num;
    int use_icp_registration;
    float icp_weight;
    int hamming_distance_threshold;

    float far_plane_distance;

    // test settings
    int runTestFlag;
    int runRefFrame;
    int runNewFrame;
    int runFrameNum;
  };


    class FrameCorrespondence
    {
    public:
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    FrameCorrespondence(Frame &f_r, Frame &f_n) :frame_ref(f_r), frame_new(f_n)
    {
            sum_weight = 0;
            rtst_error = 0;
            matches.clear();
            data_pairs_3d.clear();



        }
        void reset()
        {
      dense_feature_cnt = 0;
      sparse_feature_cnt = 0;
      sum_weight = 0;
      rtst_error = 0;
            matches.clear();
      matches.clear();
      data_pairs_3d.clear();
            weight_per_feature.clear();
      weight_per_pair.clear();
            sum_p_ref.setZero();
            sum_p_new.setZero();
            sum_p_ref_new.setZero();
            sum_p_ref_ref.setZero();
            sum_p_new_new.setZero();
            sum_p_new_ref.setZero();
        }

        ~FrameCorrespondence()
        {
      reset();
        }


    float calculate_average_disparity(CameraPara camera_para)
    {
      float average_disparity = 0;
      if(data_pairs_3d.size() > 0)
      {
        for(int k = 0; k < data_pairs_3d.size(); k++)
        {
          Eigen::Vector3d p0 = data_pairs_3d[k].first.cast<double>();
          Eigen::Vector3d p1 = data_pairs_3d[k].second.cast<double>();

          float ref_x = p0(0)/p0(2) * camera_para.c_fx + camera_para.c_cx;
          float ref_y = p0(1)/p0(2) * camera_para.c_fy + camera_para.c_cy;

          float new_x = p1(0)/p1(2) * camera_para.c_fx + camera_para.c_cx;
          float new_y = p1(1)/p1(2) * camera_para.c_fy + camera_para.c_cy;
          average_disparity += sqrt((ref_x - new_x) * (ref_x - new_x) + (ref_y - new_y)*(ref_y - new_y));
        }
        average_disparity /= (data_pairs_3d.size() + 1);
        average_disparity /= fmax(camera_para.c_fx,camera_para.c_fy);
      }
      else
      {

        for (int i = 0; i < matches.size(); i++)
        {
          Eigen::Vector3d p0 = frame_ref.local_points[matches[i].queryIdx];
          Eigen::Vector3d p1 = frame_new.local_points[matches[i].trainIdx];

          float ref_x = p0(0)/p0(2) * camera_para.c_fx + camera_para.c_cx;
          float ref_y = p0(1)/p0(2) * camera_para.c_fy + camera_para.c_cy;

          float new_x = p1(0)/p1(2) * camera_para.c_fx + camera_para.c_cx;
          float new_y = p1(1)/p1(2) * camera_para.c_fy + camera_para.c_cy;
          average_disparity += sqrt((ref_x - new_x) * (ref_x - new_x) + (ref_y - new_y)*(ref_y - new_y));
        }
        average_disparity /= (matches.size() + 1);
        average_disparity /= fmax(camera_para.c_fx,camera_para.c_fy);
      }

      return average_disparity;
    }

    void preIntegrateICP(float icp_weight)
        {

            int pairMatchNum = data_pairs_3d.size();
      dense_feature_cnt = data_pairs_3d.size();

      assert(data_pairs_3d.size() == NormalICP.size());


      XJ_NNT_XI = Eigen::MatrixXd(9,9);
      XI_NNT_XJ = Eigen::MatrixXd(9,9);
      XI_NNT = Eigen::MatrixXd(9,3);
      XJ_NNT = Eigen::MatrixXd(9,3);
      NNT = Eigen::MatrixXd(3,3);

      XJ_NNT_XI.setZero();
      XI_NNT_XJ.setZero();
      XI_NNT.setZero();
      XJ_NNT.setZero();
      NNT.setZero();


      Eigen::MatrixXd XI_ARRAY = Eigen::MatrixXd(9,3);
      Eigen::MatrixXd XJ_ARRAY = Eigen::MatrixXd(9,3);
            for (int i = 0; i < pairMatchNum; i++)
            {

                Eigen::Vector3d p0 = data_pairs_3d[i].first.cast<double>();
                Eigen::Vector3d p1 = data_pairs_3d[i].second.cast<double>();

        float depth = p0[2];
        double weight = 1 / (depth ) * icp_weight / pairMatchNum * Truncation_ICP_POINTS ;// * weight_per_pair[i];
#if USE_NORMAL_WEIGHT
        Eigen::Vector3d n = NormalICP[i].cast<double>();

        XI_ARRAY.setZero();
        XJ_ARRAY.setZero();
        XI_ARRAY.block<3,1>(0,0) = p0;
        XI_ARRAY.block<3,1>(3,1) = p0;
        XI_ARRAY.block<3,1>(6,2) = p0;
        XJ_ARRAY.block<3,1>(0,0) = p1;
        XJ_ARRAY.block<3,1>(3,1) = p1;
        XJ_ARRAY.block<3,1>(6,2) = p1;


        NNT += n * n.transpose();
        XJ_NNT_XI = XJ_ARRAY * n * n.transpose() * XI_ARRAY.transpose();
        XI_NNT_XJ = XI_ARRAY * n * n.transpose() * XJ_ARRAY.transpose();
        XI_NNT = XI_ARRAY * n * n.transpose();
        XJ_NNT = XJ_ARRAY * n * n.transpose();


#else
                sum_weight += weight;
                sum_p_ref += p0 * weight;
                sum_p_new += p1 * weight;
                sum_p_ref_new += p0 * p1.transpose() * weight;
                sum_p_ref_ref += p0 * p0.transpose() * weight;
                sum_p_new_new += p1 * p1.transpose() * weight;
                sum_p_new_ref += p1 * p0.transpose() * weight;

#endif

                sum_icp_p_ref_T_p_ref = p0.transpose() * p0;
                sum_icp_p_new_T_p_new = p1.transpose() * p1;
      }
            sum_p_ref_icp = sum_p_ref - sum_p_ref_feature;
            sum_p_new_icp = sum_p_new - sum_p_new_feature;

        }


    // only release memory
    void clear_memory()
    {
      sparse_feature_cnt = matches.size();
      dense_feature_cnt = data_pairs_3d.size();
      matches.clear();
      weight_per_feature.clear();
      data_pairs_3d.clear();

    }

    void preIntegrateWithHuberNorm()
    {
        sum_p_ref.setZero();
        sum_p_new.setZero();
        sum_p_ref_new.setZero();
        sum_p_ref_ref.setZero();
        sum_p_new_new.setZero();
        sum_p_new_ref.setZero();
        sum_weight = 0;
        double location_weight = 1;
        int use_weight = 0;


        sum_feature_p_ref_T_p_ref = 0;
        sum_feature_p_new_T_p_new = 0;
        sum_icp_p_ref_T_p_ref = 0;
        sum_icp_p_new_T_p_new = 0;
        if (weight_per_feature.size() == matches.size())
        {
            use_weight = 1;
        }
        sparse_feature_cnt = matches.size();


        Eigen::Matrix3d rotationRef = frame_ref.pose_sophus[0].rotationMatrix();
        Eigen::Vector3d translationRef = frame_ref.pose_sophus[0].translation();
        Eigen::Matrix3d rotationNew = frame_new.pose_sophus[0].rotationMatrix();
        Eigen::Vector3d translationNew = frame_new.pose_sophus[0].translation();

        for (int i = 0; i < matches.size(); i++)
        {
            float depth = frame_ref.local_points[matches[i].queryIdx][2];

            double weight = 1 / (depth * depth);
            if (use_weight)
            {
                weight = weight * weight_per_feature[i];
            }

            Eigen::Vector3d diff = rotationRef * frame_ref.local_points[matches[i].queryIdx] + translationRef -
                    rotationNew * frame_new.local_points[matches[i].trainIdx] - translationNew;
            float error = diff.norm();
            float huber_weight = 1;
            float huber_threshold = 0.001 * depth;
#if 1
            if(error > huber_threshold)
            {
                huber_weight = (huber_threshold / error);
            }
#endif
            weight *= huber_weight;
            weight *= fabs((frame_ref.frame_index - frame_new.frame_index));
          //  printf("error : %f %f\r\n",error,huber_weight);
            sum_weight += weight;
            sum_p_ref += frame_ref.local_points[matches[i].queryIdx] * weight;
            sum_p_new += frame_new.local_points[matches[i].trainIdx] * weight;
            sum_p_ref_new += frame_ref.local_points[matches[i].queryIdx] * frame_new.local_points[matches[i].trainIdx].transpose() * weight;
            sum_p_ref_ref += frame_ref.local_points[matches[i].queryIdx] * frame_ref.local_points[matches[i].queryIdx].transpose() * weight;
            sum_p_new_new += frame_new.local_points[matches[i].trainIdx] * frame_new.local_points[matches[i].trainIdx].transpose() * weight;
            sum_p_new_ref += frame_new.local_points[matches[i].trainIdx] * frame_ref.local_points[matches[i].queryIdx].transpose() * weight;

            sum_feature_p_ref_T_p_ref = frame_ref.local_points[matches[i].queryIdx].transpose() * frame_ref.local_points[matches[i].queryIdx];
            sum_feature_p_new_T_p_new = frame_new.local_points[matches[i].trainIdx].transpose() * frame_new.local_points[matches[i].trainIdx];
        }
        sum_p_ref_feature = sum_p_ref;
        sum_p_new_feature = sum_p_new;
    }


        void preIntegrate()
        {
            sum_p_ref.setZero();
            sum_p_new.setZero();
            sum_p_ref_new.setZero();
            sum_p_ref_ref.setZero();
            sum_p_new_new.setZero();
            sum_p_new_ref.setZero();
            sum_weight = 0;
            double location_weight = 1;
            int use_weight = 0;


            sum_feature_p_ref_T_p_ref = 0;
            sum_feature_p_new_T_p_new = 0;
            sum_icp_p_ref_T_p_ref = 0;
            sum_icp_p_new_T_p_new = 0;
            if (weight_per_feature.size() == matches.size())
            {
                use_weight = 1;
            }
            sparse_feature_cnt = matches.size();
            for (int i = 0; i < matches.size(); i++)
            {
                float depth = frame_ref.local_points[matches[i].queryIdx][2];

                double weight = 1 / (depth * depth);
                if (use_weight)
                {
                    weight = weight * weight_per_feature[i];
                }
                sum_weight += weight;
                sum_p_ref += frame_ref.local_points[matches[i].queryIdx] * weight;
                sum_p_new += frame_new.local_points[matches[i].trainIdx] * weight;
                sum_p_ref_new += frame_ref.local_points[matches[i].queryIdx] * frame_new.local_points[matches[i].trainIdx].transpose() * weight;
                sum_p_ref_ref += frame_ref.local_points[matches[i].queryIdx] * frame_ref.local_points[matches[i].queryIdx].transpose() * weight;
                sum_p_new_new += frame_new.local_points[matches[i].trainIdx] * frame_new.local_points[matches[i].trainIdx].transpose() * weight;
                sum_p_new_ref += frame_new.local_points[matches[i].trainIdx] * frame_ref.local_points[matches[i].queryIdx].transpose() * weight;

                sum_feature_p_ref_T_p_ref = frame_ref.local_points[matches[i].queryIdx].transpose() * frame_ref.local_points[matches[i].queryIdx];
                sum_feature_p_new_T_p_new = frame_new.local_points[matches[i].trainIdx].transpose() * frame_new.local_points[matches[i].trainIdx];
            }

#if 0
            if (rtst_error > 0)
            {
                sum_weight /= rtst_error;
                sum_p_ref /= rtst_error;
                sum_p_new /= rtst_error;
                sum_p_ref_new /= rtst_error;
                sum_p_ref_ref /= rtst_error;
                sum_p_new_new /= rtst_error;
                sum_p_new_ref /= rtst_error;

            }
#endif
            sum_p_ref_feature = sum_p_ref;
            sum_p_new_feature = sum_p_new;

        }


    Frame &frame_ref;
    Frame &frame_new;
        Eigen::Vector3d sum_p_ref_feature;		// sum(p_ref)
        Eigen::Vector3d sum_p_new_feature;		// sum(p_new)
        Eigen::Vector3d sum_p_ref_icp;		// sum(p_ref)
        Eigen::Vector3d sum_p_new_icp;		// sum(p_new)

        Eigen::Vector3d sum_p_ref;		// sum(p_ref)
        Eigen::Vector3d sum_p_new;		// sum(p_new)
        Eigen::Matrix3d sum_p_ref_new;  // sum(p_ref * p_new^T)
        Eigen::Matrix3d sum_p_ref_ref;  // sum(p_ref * p_ref^T)
        Eigen::Matrix3d sum_p_new_new;  // sum(p_new * p_new^T)
        Eigen::Matrix3d sum_p_new_ref;  // sum(p_new * p_ref^T)
        // to calculate reprojection error
        float sum_feature_p_ref_T_p_ref, sum_feature_p_new_T_p_new;
        float sum_icp_p_ref_T_p_ref, sum_icp_p_new_T_p_new;

        double sum_weight;
    std::vector< cv::DMatch > matches;	// query for ref, train for new
        std::vector<float> weight_per_feature; //weight per feature
    std::vector<float> weight_per_pair;
    int sparse_feature_cnt;
    int dense_feature_cnt;

    std::vector<std::pair<Eigen::Vector3f,Eigen::Vector3f> > data_pairs_3d;
    std::vector<Eigen::Vector3f> NormalICP;
        double rtst_error;

 // for normal weighted case
    Eigen::MatrixXd XJ_NNT_XI;
    Eigen::MatrixXd XI_NNT_XJ;
    Eigen::MatrixXd XI_NNT;
    Eigen::MatrixXd XJ_NNT;
    Eigen::MatrixXd NNT;


    };
    class KeyFrameDatabase
    {
    public:
EIGEN_MAKE_ALIGNED_OPERATOR_NEW
        KeyFrameDatabase(int currentIndex)
        {
            keyFrameIndex = currentIndex;
            corresponding_frames.clear();
            relative_pose_from_key_to_current.clear();
            localFrameCorrList.clear();
        }
        KeyFrameDatabase()
        {
            keyFrameIndex = -1;
            corresponding_frames.clear();
            relative_pose_from_key_to_current.clear();
            localFrameCorrList.clear();
        }
        int keyFrameIndex;                          // the index of keyframe in framelist
        std::vector<int> corresponding_frames;
        std::vector<FrameCorrespondence> localFrameCorrList;
        std::vector<int> corresponding_keyframes;   // the matched keyframes
        PoseSE3dList relative_pose_from_key_to_current;
    };


    class FastBAContext{
    public:

        EIGEN_MAKE_ALIGNED_OPERATOR_NEW
        std::vector<FrameCorrespondence> keyFrameCorrList;
        std::vector<KeyFrameDatabase>  KeyframeDataList;
        MILD::LoopClosureDetector mild;
        CameraPara camera_para;
        FastBAContext()
        {
        }

        void init(CameraPara camera_para_t)
        {
            int maximum_frame_num = 30000;
            mild.init(FEATURE_TYPE_ORB, 16, 0);
            keyFrameCorrList.reserve(maximum_frame_num * 5);
            KeyframeDataList.reserve(maximum_frame_num);
            camera_para = camera_para_t;
        }
    };



    extern GlobalParameters g_para;




  bool FrameMatchingTwoViewRGB(FrameCorrespondence &fCorr,
                               CameraPara para,
                               MILD::SparseMatcher frame_new_matcher,
                               PoseSE3d &relative_pose_from_ref_to_new, float &average_disparity, float &scale_change_ratio, bool &update_keyframe_from_dense_matching, bool use_initial_guess = 0);

  float optimizeKeyFrameMap(std::vector<FrameCorrespondence> &fCList, std::vector<Frame> &F, std::vector<KeyFrameDatabase> &kflist, int origin);







    // calculate jacobian matrix of 3d point based on pose
    inline Eigen::MatrixXd computeJacobian3DPoint(PoseSE3d p_cur_to_ref, Eigen::Vector3d pt_cur)
    {
        Eigen::MatrixXd J(3, 6);
        J.setZero();
        J(0, 0) = 1;
        J(1, 1) = 1;
        J(2, 2) = 1;
        Eigen::Vector3d pt_ref;
        pt_ref = applyPose(p_cur_to_ref,pt_cur);
        J(0, 4) = pt_ref(2);
        J(0, 5) = -pt_ref(1);
        J(1, 3) = -pt_ref(2);
        J(1, 5) = pt_ref(0);
        J(2, 3) = pt_ref(1);
        J(2, 4) = -pt_ref(0);
        return J;
    }



  inline __m128 CrossProduct(__m128 a, __m128 b)
  {
    return _mm_sub_ps(
      _mm_mul_ps(_mm_shuffle_ps(a, a, _MM_SHUFFLE(3, 0, 2, 1)), _mm_shuffle_ps(b, b, _MM_SHUFFLE(3, 1, 0, 2))),
      _mm_mul_ps(_mm_shuffle_ps(a, a, _MM_SHUFFLE(3, 1, 0, 2)), _mm_shuffle_ps(b, b, _MM_SHUFFLE(3, 0, 2, 1)))
    );
  }



  inline void ComputeThreeMaxima(std::vector<std::vector<int>> &histo, const int L, int &ind1, int &ind2, int &ind3)
    {
        int max1 = 0;
        int max2 = 0;
        int max3 = 0;

        for (int i = 0; i<L; i++)
        {
            const int s = histo[i].size();
            if (s>max1)
            {
                max3 = max2;
                max2 = max1;
                max1 = s;
                ind3 = ind2;
                ind2 = ind1;
                ind1 = i;
            }
            else if (s>max2)
            {
                max3 = max2;
                max2 = s;
                ind3 = ind2;
                ind2 = i;
            }
            else if (s>max3)
            {
                max3 = s;
                ind3 = i;
            }
        }

        if (max2<0.1f*(float)max1)
        {
            ind2 = -1;
            ind3 = -1;
        }
        else if (max3<0.1f*(float)max1)
        {
            ind3 = -1;
        }
    }
  inline void RefineByRotation(Frame & frame_ref, Frame & frame_new, std::vector< cv::DMatch > &init_matches)
    {
        // refine results by orientation

        int angle_interval = 20;
        int histo_length = round(360.0 / angle_interval);
    std::vector<std::vector<int>> rotHist(histo_length);
        const float factor = 1.0f / angle_interval;
        for (int i = 0; i<histo_length; i++)
            rotHist[i].reserve(500);

        for (int i = 0; i < init_matches.size(); i++)
        {
            float rot = frame_ref.keypoints[init_matches[i].queryIdx].angle - frame_new.keypoints[init_matches[i].trainIdx].angle;
            if (rot<0.0)
                rot += 360.0f;
            int bin = round(rot*factor);
            if (bin == histo_length)
                bin = 0;
            assert(bin >= 0 && bin<histo_length);
            rotHist[bin].push_back(i);
        }
        int ind1 = -1;
        int ind2 = -1;
        int ind3 = -1;

        ComputeThreeMaxima(rotHist, histo_length, ind1, ind2, ind3);
    std::vector< cv::DMatch > ransac_inlier_matches;

        for (int i = 0; i<histo_length; i++)
        {
      if (fmin(abs(i - ind1), abs(histo_length - (i - ind1))) < 2 || i == ind2 || i == ind3)
            {
                for (size_t j = 0, jend = rotHist[i].size(); j<jend; j++)
                {
                    ransac_inlier_matches.push_back(init_matches[rotHist[i][j]]);
                }
            }
        }
        init_matches = ransac_inlier_matches;
    }

  // Generate vertex map in camera coordinates
  inline cv::Mat gen_vertex_map(cv::Mat &_depth_image, const CameraPara &_camera)
  {
      int width = _depth_image.cols;
      int height = _depth_image.rows;
      float fx = _camera.c_fx;
      float fy = _camera.c_fy;
      float cx = _camera.c_cx;
      float cy = _camera.c_cy;

      cv::Mat vertex_image(height, width, CV_32FC3);
      for (int r_idx = 0; r_idx < height; ++r_idx) {
          for (int c_idx = 0; c_idx < width; ++c_idx) {
              float depth = _depth_image.at<unsigned short>(r_idx, c_idx)/ _camera.depth_scale;
              vertex_image.at<cv::Vec3f>(r_idx, c_idx)[0] = (c_idx-cx)*depth/fx;
              vertex_image.at<cv::Vec3f>(r_idx, c_idx)[1] = (r_idx-cy)*depth/fy;
              vertex_image.at<cv::Vec3f>(r_idx, c_idx)[2] = depth;
          }
      }

      return vertex_image;
  }

  inline cv::Mat gen_normal_map(cv::Mat _vertex_image)
  {
      const int ex = 5;
      const int ey = 5;
      int width = _vertex_image.cols;
      int height = _vertex_image.rows;

      cv::Mat normal_image(height, width, CV_32FC3);
      for (int r_idx = 0; r_idx < height; ++r_idx) {
          for (int c_idx = 0; c_idx < width; ++c_idx) {
              cv::Vec3f point_cv = _vertex_image.at<cv::Vec3f>(r_idx, c_idx);
              Eigen::Vector3f point(point_cv[0], point_cv[1], point_cv[2]);

              normal_image.at<cv::Vec3f>(r_idx, c_idx)[0] = 0;
              normal_image.at<cv::Vec3f>(r_idx, c_idx)[1] = 0;
              normal_image.at<cv::Vec3f>(r_idx, c_idx)[2] = 0;
              if (point.norm() > 0  && r_idx > 0 && r_idx + 1 < height && c_idx > 0 && c_idx + 1 <  width)
              {
                  int counter = 0;
                  Eigen::Vector3f center(0, 0, 0);
                  std::vector<Eigen::Vector3f> valid_p;
                  for (int yi = r_idx-ey/2; yi <= r_idx+ey/2; ++yi) {
                      for (int xi = c_idx-ex/2; xi <= c_idx+ex/2; ++xi) {
                          if (xi >= 0 && xi < width && yi >= 0 && yi < height) {
                              cv::Vec3f p_cv = _vertex_image.at<cv::Vec3f>(yi, xi);
                              if (cv::norm(p_cv) > 0) {
                                  Eigen::Vector3f p(p_cv[0], p_cv[1], p_cv[2]);
                                  center += p;
                                  valid_p.push_back(p);
                                  ++counter;
                              }
                          }
                      }
                  }
                  center /= (float)counter;

                  Eigen::MatrixXf A(3, counter);
                  for (int idx = 0; idx < counter; ++idx) {
                      A.col(idx) = valid_p[idx] - (Eigen::Vector3f)center;
                  }

                  Eigen::Matrix3f cov = A*A.transpose();

                  Eigen::EigenSolver<Eigen::Matrix3d> eigen_solver(cov.cast<double>(), true);

                  const Eigen::Vector3d &eigen_value = eigen_solver.eigenvalues().real();
                  int min_idx = 0;
                  eigen_value.minCoeff(&min_idx);
                  Eigen::Vector3d min_eigen_vector_d = eigen_solver.eigenvectors().real().col(min_idx);
                  Eigen::Vector3f min_eigen_vector = min_eigen_vector_d.cast<float>();

                  // Use eigen vector facing camera as normal
                  if (min_eigen_vector.dot(point) < 0) {
                      normal_image.at<cv::Vec3f>(r_idx, c_idx)[0] = min_eigen_vector(0);
                      normal_image.at<cv::Vec3f>(r_idx, c_idx)[1] = min_eigen_vector(1);
                      normal_image.at<cv::Vec3f>(r_idx, c_idx)[2] = min_eigen_vector(2);
                  }
                  else {
                      normal_image.at<cv::Vec3f>(r_idx, c_idx)[0] = -min_eigen_vector(0);
                      normal_image.at<cv::Vec3f>(r_idx, c_idx)[1] = -min_eigen_vector(1);
                      normal_image.at<cv::Vec3f>(r_idx, c_idx)[2] = -min_eigen_vector(2);
                  }
                  for (int cnt = 0; cnt < 3; cnt++)
                  {
                      if (fabs(normal_image.at<cv::Vec3f>(r_idx, c_idx)[cnt]) > 1e8)
                      {
                          normal_image.at<cv::Vec3f>(r_idx, c_idx)[0] = 0;
                          normal_image.at<cv::Vec3f>(r_idx, c_idx)[1] = 0;
                          normal_image.at<cv::Vec3f>(r_idx, c_idx)[2] = 0;
                      }
                  }
              }
          }
      }

      return normal_image;
  }


  float initGraphHuberNorm(std::vector<FrameCorrespondence> &fCList, std::vector<Frame> &F);



  float reprojection_error_3Dto3D(const FrameCorrespondence &fC, const Sophus::SE3d &relative_pose_from_ref_to_new);
  float reprojection_error_3Dto3D(std::vector<FrameCorrespondence> fCList);
  float reprojection_error_3Dto3D(const FrameCorrespondence &fC);
  float reprojection_error_3Dto3D(std::vector<FrameCorrespondence> fCList, std::vector<int>candidates);
  float reprojection_error_3Dto3D_perFrame(std::vector<FrameCorrespondence> fCList,
                                           Eigen::VectorXd &errorPerFrame,
                                           Eigen::VectorXd &pointsPerFrame,
                                           Eigen::VectorXd &connectionsPerFrame);
  float reprojection_error_3Dto3D_perFrame(std::vector<FrameCorrespondence> fCList,
                                           Eigen::VectorXd &errorPerFrame,
                                           Eigen::VectorXd &pointsPerFrame,
                                           Eigen::VectorXd &connectionsPerFrame,
                                           std::vector<std::vector<int> > &related_connections);
  float reprojection_error_3Dto3D(Point3dList pt_ref,
                                  Point3dList pt_new,
                                  Sophus::SE3d relative_pose_from_ref_to_new,
                                  int use_huber_norm = 0,
                                  float huber_norm_threshold = 0.008);
  float reprojection_error_3Dto3D(Frame frame_ref,
                                  Frame frame_new,
                                  std::vector< cv::DMatch > inlier_matches,
                                  Sophus::SE3d relative_pose_from_ref_to_new,
                                  int use_huber_norm = 0,
                                  float huber_norm_threshold = 0.008);


  Eigen::Matrix3d skewMatrixProduct(Eigen::Vector3d t1, Eigen::Vector3d t2);
  Eigen::Matrix3d getSkewSymmetricMatrix(Eigen::Vector3d t);

}

#endif

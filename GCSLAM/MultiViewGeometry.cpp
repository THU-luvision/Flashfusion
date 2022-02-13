
#include "MultiViewGeometry.h"
#include "ORBSLAM/ORBextractor.h"
#include <xmmintrin.h>


#include <smmintrin.h>
#include <time.h>
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include <opencv2/core/eigen.hpp>
#include <Eigen/Sparse>
//#include <Eigen/CholmodSupport>

using namespace std;
using namespace cv;

#define REPROJECTION_TH 0.01




class mystreambuf: public std::streambuf
{
};
mystreambuf nostreambuf;
std::ostream nocout(&nostreambuf);

#define COUT_THRESHOLD 3
#define LOG_INFO(x) ((x >= COUT_THRESHOLD)? std::cout : nocout)




namespace MultiViewGeometry
{

  GlobalParameters g_para;



   void optimize_3d_to_3d_huber_filter(Frame &frame_ref,
                                             Frame& frame_new,
                                             std::vector< cv::DMatch > &ransac_inlier_matches,
                                             PoseSE3d &relative_pose_from_ref_to_new,
                                             std::vector<float> &weight_per_point,
                                             float outlier_threshold = 0.015,
                                             int max_iter_num = 4,
                                             float huber_threshold = 0.008)
    {

    weight_per_point.clear();
    clock_t  start,end;
        int valid_3d_cnt = ransac_inlier_matches.size();

    Eigen::MatrixXf Je(6, 1), JTJ(6, 6);
    Eigen::Vector3f err;
    Eigen::VectorXf delta(6);
    Point3dList p_ref, p_new;
        p_ref.clear();
        p_new.clear();
        p_ref.reserve(valid_3d_cnt);
        p_new.reserve(valid_3d_cnt);



    std::vector<__m128> ref_points_vectorized(valid_3d_cnt);
    std::vector<__m128> new_points_vectorized(valid_3d_cnt);
    std::vector<float> depth(valid_3d_cnt);
        for (size_t i = 0; i < valid_3d_cnt; i++)
        {

            Eigen::Vector3d pt_ref(frame_ref.local_points[ransac_inlier_matches[i].queryIdx]);
            Eigen::Vector3d pt_new(frame_new.local_points[ransac_inlier_matches[i].trainIdx]);
            p_ref.push_back(pt_ref);
            p_new.push_back(pt_new);

          ref_points_vectorized[i] = _mm_setr_ps(pt_ref.x(),pt_ref.y(),pt_ref.z(),1);
          new_points_vectorized[i] = _mm_setr_ps(pt_new.x(),pt_new.y(),pt_new.z(),0);
        depth[i] = pt_ref.z();
        }
//		float init_error = reprojection_error_3Dto3D(p_ref, p_new, relative_pose_from_ref_to_new, 1);
        for (int iter_cnt = 0; iter_cnt < max_iter_num; iter_cnt++)
        {

            Je.setZero();
            JTJ.setZero();


            Eigen::MatrixXd R_ref = relative_pose_from_ref_to_new.rotationMatrix();
            Eigen::Vector3d t_ref = relative_pose_from_ref_to_new.translation();
            __m128 T_ref[3];

            T_ref[0] = _mm_setr_ps(R_ref(0,0),R_ref(0,1),R_ref(0,2),t_ref(0));
            T_ref[1] = _mm_setr_ps(R_ref(1,0),R_ref(1,1),R_ref(1,2),t_ref(1));
            T_ref[2] = _mm_setr_ps(R_ref(2,0),R_ref(2,1),R_ref(2,2),t_ref(2));

            Eigen::MatrixXf J_i_sse(3,6);

            J_i_sse.setZero();
            J_i_sse(0, 0) = 1;
            J_i_sse(1, 1) = 1;
            J_i_sse(2, 2) = 1;
            __m128 res, reprojection_error_vec;
            res[3] = 0;
            reprojection_error_vec[3] = 0;
            for (int i = 0; i < valid_3d_cnt; i++)
            {

                res = _mm_add_ps(_mm_dp_ps(T_ref[1], ref_points_vectorized[i], 0xf2),
                    _mm_dp_ps(T_ref[0], ref_points_vectorized[i], 0xf1));
                res = _mm_add_ps(res, _mm_dp_ps(T_ref[2], ref_points_vectorized[i], 0xf4));
                reprojection_error_vec = _mm_sub_ps(res, new_points_vectorized[i]);

                float error_sse = sqrt(_mm_cvtss_f32(_mm_dp_ps(res, res, 0x71))) / depth[i];

                float weight_huber = 1;
                if (error_sse > huber_threshold)
                {
                  weight_huber = huber_threshold / error_sse;
                }
                float weight = weight_huber / (depth[i]);


                const __m128 scalar = _mm_set1_ps(weight);
                reprojection_error_vec = _mm_mul_ps(reprojection_error_vec, scalar);
                __m128 cross_value = CrossProduct(res,reprojection_error_vec);
                J_i_sse(0, 4) = res[2];
                J_i_sse(0, 5) = -res[1];
                J_i_sse(1, 3) = -res[2];
                J_i_sse(1, 5) = res[0];
                J_i_sse(2, 3) = res[1];
                J_i_sse(2, 4) = -res[0];
                Je(0,0) += reprojection_error_vec[0];
                Je(1,0) += reprojection_error_vec[1];
                Je(2,0) += reprojection_error_vec[2];
                Je(3,0) += cross_value[0];
                Je(4,0) += cross_value[1];
                Je(5,0) += cross_value[2];

                JTJ += J_i_sse.transpose() * J_i_sse * weight;

            }
            delta = JTJ.inverse() * Je;
            Eigen::VectorXd delta_double = delta.cast<double>();


            relative_pose_from_ref_to_new = Sophus::SE3d::exp(delta_double).inverse() * relative_pose_from_ref_to_new;
        }
        std::vector< cv::DMatch > matches_refined;
        matches_refined.reserve(valid_3d_cnt);
        for (int i = 0; i < p_ref.size(); i++)
        {
            Eigen::Vector3d reprojection_error = applyPose(relative_pose_from_ref_to_new, p_ref[i]) - (p_new[i]);



            if (reprojection_error.norm() / p_ref[i].z() < outlier_threshold)
            {

                float weight_huber = 1;
                float error = reprojection_error.norm() / p_ref[i].z();
                if (error > huber_threshold)
                {
                  weight_huber = huber_threshold / error;
                }
                matches_refined.push_back(ransac_inlier_matches[i]);
                weight_per_point.push_back(weight_huber);
            }
        }
        ransac_inlier_matches = matches_refined;
    }


  void estimateRigid3DTransformation(Frame &frame_ref,
      Frame &frame_new,
      std::vector< DMatch > &init_matches,
      Eigen::Matrix3d &R, Eigen::Vector3d &t,
      float reprojection_error_threshold,
      int max_iter_num)
    {



      std::vector<DMatch> matches_before_filtering = init_matches;
      // random 100 times test
      int N_predict_inliers = init_matches.size();
      int N_total = matches_before_filtering.size();
      Eigen::Vector3d ref_points[4], mean_ref_points;
      Eigen::Vector3d new_points[4], mean_new_points;
      int candidate_seed;
      Eigen::Matrix3d H, UT, V;
      Eigen::Matrix3d temp_R;
      Eigen::Vector3d temp_t;
      int count;
      int best_results = 0;
      float reprojection_error_threshold_square = reprojection_error_threshold *reprojection_error_threshold;
          //	[R t] * [x,1]  3x4 * 4xN
      std::vector<__m128> ref_points_vectorized(N_total);
      std::vector<__m128> new_points_vectorized(N_total);
      std::vector<float> depth_square(N_total);
      for (size_t i = 0; i < N_total; i++)
      {
        Eigen::Vector3d ref_point = frame_ref.local_points[matches_before_filtering[i].queryIdx];
        Eigen::Vector3d new_point = frame_new.local_points[matches_before_filtering[i].trainIdx];
        ref_points_vectorized[i] = _mm_setr_ps((float)ref_point[0],(float)ref_point[1],(float)ref_point[2],1);
        new_points_vectorized[i] = _mm_setr_ps((float)new_point[0],(float)new_point[1],(float)new_point[2],1);
        depth_square[i] = ref_point[2] * ref_point[2];
      }
      Eigen::MatrixXf rigid_transform(4, 3);
      for (int cnt = 0; cnt < max_iter_num; cnt++)
      {
        H.setZero();
        mean_ref_points.setZero();
        mean_new_points.setZero();
        for (int i = 0; i < 4; i++)
        {
          candidate_seed = rand() % N_predict_inliers;
          ref_points[i] = frame_ref.local_points[init_matches[candidate_seed].queryIdx];
          new_points[i] = frame_new.local_points[init_matches[candidate_seed].trainIdx];
          mean_ref_points += ref_points[i];
          mean_new_points += new_points[i];
        }
        mean_ref_points /= 4;
        mean_new_points /= 4;
        for (int i = 0; i < 4; i++)
        {
          H += (ref_points[i] - mean_ref_points) * (new_points[i] - mean_new_points).transpose();
        }
        Eigen::JacobiSVD<Eigen::MatrixXd> svd(H, Eigen::ComputeThinU | Eigen::ComputeThinV);
        UT = svd.matrixU().transpose();
        V = svd.matrixV();
        temp_R = V * UT;
        if (temp_R.determinant() < 0)
        {
          V(0, 2) = -V(0, 2);
          V(1, 2) = -V(1, 2);
          V(2, 2) = -V(2, 2);
          temp_R = V * UT;
        }
        temp_t = mean_new_points - temp_R * mean_ref_points;
        Eigen::MatrixXd transformation(3,4);

        unsigned char mask0 = 0xf1;
        unsigned char mask1 = 0xf2;
        unsigned char mask2 = 0xf4;
        unsigned char mask3 = 0xf8;
        __m128 transform_vectorized[4];
        for (int i = 0; i < 3; i++)
        {
          transform_vectorized[i] = _mm_setr_ps( (float)temp_R(i, 0), (float)temp_R(i, 1), (float)temp_R(i, 2), (float)temp_t(i));
        }
        count = 0;
        for (size_t i = 0; i < N_total; i++)
        {
          __m128 res;
          res = _mm_add_ps(_mm_dp_ps(transform_vectorized[1], ref_points_vectorized[i], 0xf2),
              _mm_dp_ps(transform_vectorized[0], ref_points_vectorized[i], 0xf1));
          res = _mm_add_ps(res, _mm_dp_ps(transform_vectorized[2], ref_points_vectorized[i], 0xf4));
          res = _mm_sub_ps(res, new_points_vectorized[i]);
          count += (_mm_cvtss_f32(_mm_dp_ps(res, res, 0x71)) / depth_square[i] < reprojection_error_threshold_square);
        }
        if (count > best_results)
        {
          best_results = count;
          R = temp_R;
          t = temp_t;
        }

      }
    }



  float ransac3D3D(Frame &frame_ref, Frame &frame_new, std::vector< DMatch > &init_matches, std::vector< DMatch > &matches_before_filtering,
      float reprojectionThreshold, int max_iter_num, FrameCorrespondence &fCorr,
      PoseSE3d &relative_pose_from_ref_to_new, CameraPara para)
    {

      if (init_matches.size() < minimum_3d_correspondence)
      {
        return 1e7;
      }
      clock_t start, end;
      clock_t start_total, end_total;

      start_total = clock();
      start = clock();
      Eigen::Matrix3d direct_R;
      Eigen::Vector3d direct_t;
      estimateRigid3DTransformation(frame_ref, frame_new, init_matches, direct_R, direct_t, reprojectionThreshold * 2, max_iter_num);
      end = clock();
      float time_ransac = end - start;
      start = clock();
      std::vector< DMatch > ransac_inlier_matches;
      std::vector< DMatch > ransac_2d_inlier_matches;
      ransac_inlier_matches.clear();
      ransac_inlier_matches.reserve(matches_before_filtering.size());
      int previous_query_index = -1;
      for (size_t i = 0; i < matches_before_filtering.size(); i++)
      {
        Eigen::Vector3d ref_point = frame_ref.local_points[matches_before_filtering[i].queryIdx];
        Eigen::Vector3d new_point = frame_new.local_points[matches_before_filtering[i].trainIdx];

        Eigen::Vector3d estimate_new_local_points = direct_R * ref_point + direct_t;
        Eigen::Vector3d normalized_predict_points = estimate_new_local_points / estimate_new_local_points(2);
        cv::KeyPoint predict_new_2D = frame_ref.keypoints[matches_before_filtering[i].queryIdx];
        cv::KeyPoint new_2D = frame_new.keypoints[matches_before_filtering[i].trainIdx];
        predict_new_2D.pt.x = normalized_predict_points(0) * para.c_fx + para.c_cx;
        predict_new_2D.pt.y = normalized_predict_points(1) * para.c_fy + para.c_cy;
        float delta_x = predict_new_2D.pt.x - new_2D.pt.x;
        float delta_y = predict_new_2D.pt.y - new_2D.pt.y;
        float average_reprojection_error_2d = sqrt(delta_x * delta_x + delta_y * delta_y);
        float reprojection_error = (new_point - direct_R * ref_point - direct_t).norm() / ref_point(2);
        if (average_reprojection_error_2d < 2)
        {
          ransac_2d_inlier_matches.push_back(matches_before_filtering[i]);
        }
        if (reprojection_error < reprojectionThreshold * 2)
        {
          if (g_para.runTestFlag)
          {
            cout << i << " " << average_reprojection_error_2d << " " << reprojection_error << endl;
          }
          // only add the first match
          if (previous_query_index != matches_before_filtering[i].queryIdx)
          {
            previous_query_index = matches_before_filtering[i].queryIdx;
            ransac_inlier_matches.push_back(matches_before_filtering[i]);
          }
        }
      }

      if (ransac_inlier_matches.size() < minimum_3d_correspondence)
      {
        return 1e7;
      }
      end = clock();
      float time_select_init_inliers = end - start;
      if (g_para.runTestFlag)
      {
        Mat img_matches;
        std::vector< DMatch > no_matches;
        no_matches.clear();
#if 0
        cv::drawMatches(frame_ref.rgb, frame_ref.keypoints, frame_new.rgb, frame_new.keypoints,
          ransac_inlier_matches, img_matches);
        char frameMatchName[256];
        memset(frameMatchName, '\0', 256);
        sprintf(frameMatchName, "frame matching after ransac");
        cv::imshow(frameMatchName, img_matches);
        cvWaitKey();
#endif
      }

      relative_pose_from_ref_to_new = Sophus::SE3d(direct_R,direct_t);

      start = clock();
      //refine match
      std::vector<float> weight_per_feature;
      optimize_3d_to_3d_huber_filter(frame_ref, frame_new,
                                     ransac_inlier_matches,
                                     relative_pose_from_ref_to_new,
                                     weight_per_feature,
                                     reprojectionThreshold, 6,0.005);
      end = clock();
      float time_nonlinear_opt = end - start;


#if 1
      start = clock();
      init_matches.clear();
      weight_per_feature.clear();
      weight_per_feature.reserve(matches_before_filtering.size());
      init_matches.reserve(matches_before_filtering.size());
      Eigen::MatrixXd H = relative_pose_from_ref_to_new.matrix3x4();
      Eigen::MatrixXd invH = relative_pose_from_ref_to_new.matrix().inverse().block<3, 4>(0, 0);
      float average_reprojection_error_2D = 0;
      for (size_t i = 0; i < matches_before_filtering.size(); i++)
      {
        Eigen::Vector3d ref_point = frame_ref.local_points[matches_before_filtering[i].queryIdx];
        Eigen::Vector3d new_point = frame_new.local_points[matches_before_filtering[i].trainIdx];
        cv::KeyPoint predict_new_2D = frame_ref.keypoints[matches_before_filtering[i].queryIdx];
        cv::KeyPoint predict_ref_2D = frame_new.keypoints[matches_before_filtering[i].trainIdx];
        cv::KeyPoint ref_2D = frame_ref.keypoints[matches_before_filtering[i].queryIdx];
        cv::KeyPoint new_2D = frame_new.keypoints[matches_before_filtering[i].trainIdx];
        Eigen::Vector4d homo_ref_points,homo_new_points;

        homo_ref_points << ref_point(0), ref_point(1), ref_point(2), 1;
        homo_new_points << new_point(0), new_point(1), new_point(2), 1;
        Eigen::Vector3d estimate_new_local_points = H * homo_ref_points;
        Eigen::Vector3d estimate_ref_local_points = invH * homo_new_points;


        predict_new_2D.pt.x = estimate_new_local_points(0) / estimate_new_local_points(2) * para.c_fx + para.c_cx;
        predict_new_2D.pt.y = estimate_new_local_points(1) / estimate_new_local_points(2)* para.c_fy + para.c_cy;
        predict_ref_2D.pt.x = estimate_ref_local_points(0) / estimate_ref_local_points(2) * para.c_fx + para.c_cx;
        predict_ref_2D.pt.y = estimate_ref_local_points(1) / estimate_ref_local_points(2) * para.c_fy + para.c_cy;
        float delta_x = predict_new_2D.pt.x - new_2D.pt.x;
        float delta_y = predict_new_2D.pt.y - new_2D.pt.y;
        float average_reprojection_error_2d_ref = sqrt(delta_x * delta_x + delta_y * delta_y);
        delta_x = predict_ref_2D.pt.x - ref_2D.pt.x;
        delta_y = predict_ref_2D.pt.y - ref_2D.pt.y;
        float average_reprojection_error_2d_new = sqrt(delta_x * delta_x + delta_y * delta_y);
        float reprojection_error = (new_point - estimate_new_local_points).norm() / ref_point(2);
        double reprojection_error_2d = average_reprojection_error_2d_ref;// max(average_reprojection_error_2d_ref, average_reprojection_error_2d_new);

        if (reprojection_error < reprojectionThreshold  && reprojection_error_2d < g_para.reprojection_error_2d_threshold)
        {
          average_reprojection_error_2D += reprojection_error_2d;
          if (g_para.runTestFlag)
          {
            cout << i << " " << average_reprojection_error_2d_ref << " " << average_reprojection_error_2d_new << " " << reprojection_error << endl;
          }


          float huber_threshold = 0.008;
          float weight_huber = 1;
          if (reprojection_error > huber_threshold)
          {
            weight_huber = huber_threshold / reprojection_error;
          }

          init_matches.push_back(matches_before_filtering[i]);
          weight_per_feature.push_back(weight_huber);
        }
      }
      if (g_para.runTestFlag)
      {
        cout << "2D/3D inlier num: " << ransac_2d_inlier_matches.size() << " " << init_matches.size() << endl;
        if (ransac_2d_inlier_matches.size() > 0)
        {
#if 0
          Mat img_matches;
          cv::drawMatches(frame_ref.rgb, frame_ref.keypoints, frame_new.rgb, frame_new.keypoints,
            ransac_2d_inlier_matches, img_matches);
          cv::imshow("2D match inliers", img_matches);
#endif
        }
      }
      ransac_inlier_matches = init_matches;
#endif
      end = clock();
      float time_select_final_inlier = end - start;

#if 0
      start = clock();
      optimize_3d_to_3d_huber_filter(frame_ref, frame_new,
                                     ransac_inlier_matches,
                                     relative_pose_from_ref_to_new,
                                     weight_lp,
                                     reprojectionThreshold, 6,0.005);
      end = clock();
#endif

      float time_last_non_linear = end - start;



//      float reprojection_error = reprojection_error_3Dto3D(frame_ref, frame_new, ransac_inlier_matches, (relative_pose_from_ref_to_new), 0);
      init_matches = ransac_inlier_matches;
      // make sure there is no outliers
  //		RefineByRotation(frame_ref, frame_new, init_matches);
  //		outlierFiltering(frame_ref, frame_new, init_matches, 5,0.01);
  //		outlierFiltering(frame_ref, frame_new, init_matches, 5,0.01);
  //    cout << "reprojection error after lp optimzation: " << reprojection_error << endl;


      end_total = clock();
      float time_total = end_total - start_total ;
#if 0
      LOG_INFO(1) << "ransac 3D to 3D total time: " << time_total << "    " << "time ransac/init/nonlinear/final/nonlinear: "
               << time_ransac << "/"
               << time_select_init_inliers << "/"
               << time_nonlinear_opt << "/"
               << time_select_final_inlier << "/"
               << time_last_non_linear <<endl;
#endif
      if (init_matches.size() > minimum_3d_correspondence)
      {
        fCorr.matches = init_matches;
        fCorr.weight_per_feature = weight_per_feature;
        fCorr.preIntegrate();
        float reprojection_error = reprojection_error_3Dto3D(fCorr,(relative_pose_from_ref_to_new));
        return reprojection_error;

      }
      return 1e6;
    }



  void outlierFiltering(Frame &frame_ref, Frame &frame_new, std::vector< cv::DMatch > &init_matches)
    {
        int candidate_num = 8;
        float distance_threshold = 0.015;
        int N = init_matches.size();
        std::vector< cv::DMatch > filtered_matches;
        filtered_matches.reserve(N);
        for (size_t i = 0; i < N; i++)
        {
            Eigen::Vector3d ref_point = frame_ref.local_points[init_matches[i].queryIdx];
            Eigen::Vector3d new_point = frame_new.local_points[init_matches[i].trainIdx];

            int distance_preserve_flag = 0;
            for (size_t j = 0; j < candidate_num; j++)
            {
                int rand_choice = rand() % N;
                Eigen::Vector3d ref_point_p = frame_ref.local_points[init_matches[rand_choice].queryIdx];
                Eigen::Vector3d new_point_p = frame_new.local_points[init_matches[rand_choice].trainIdx];
                double d1 = (ref_point_p - ref_point).norm();
                double d2 = (new_point_p - new_point).norm();
                if (fabs(d1 - d2) / ref_point(2) < distance_threshold)
                {
                    distance_preserve_flag = 1;
                    break;
                }
            }
            if (distance_preserve_flag)
            {
                filtered_matches.push_back(init_matches[i]);
            }
        }
        init_matches = filtered_matches;
    }


  bool FrameMatchingTwoViewRGB(FrameCorrespondence &fCorr,
                               MultiViewGeometry::CameraPara camera_para,
                               MILD::SparseMatcher frame_new_matcher,
                               PoseSE3d &relative_pose_from_ref_to_new,
                               float &average_disparity,
                               float &scale_change_ratio,
                               bool &update_keyframe_from_dense_matching,
                               bool use_initial_guess)
  {

    float time_feature_matching,time_ransac,time_filter,time_refine,time_rotation_filter ;
    update_keyframe_from_dense_matching = 0;

    float reprojection_error_feature_based;
    float reprojection_error_dense_based;

    PoseSE3d init_guess_relative_pose_from_ref_to_new = relative_pose_from_ref_to_new;
    Frame &frame_ref = fCorr.frame_ref;
    Frame &frame_new = fCorr.frame_new;
    bool matching_success = 0;
    bool dense_success = 0, sparse_success = 0;

    LOG_INFO(1) << "************frame registration: " << frame_ref.frame_index << " vs "<< frame_new.frame_index << "************" << endl;


    vector<vector<DMatch>> matches;
    std::vector< DMatch > init_matches;
    init_matches.clear();
    matches.clear();
    clock_t start, end;
    clock_t start_total, end_total;
    double duration;
    start_total = clock();

    // feature matching based on hamming distance

    frame_new_matcher.search_8(frame_ref.descriptor, init_matches, g_para.hamming_distance_threshold);
    int matched_feature_pairs = init_matches.size();
    int rotation_inliers = 0;

    if (0)
    {
      Mat img_matches;
      std::vector< DMatch > no_matches;
      no_matches.clear();
      cv::drawMatches(frame_ref.rgb, frame_ref.keypoints, frame_new.rgb, frame_new.keypoints,
        init_matches, img_matches);
      char frameMatchName[256];
      memset(frameMatchName, '\0', 256);
      //	sprintf(frameMatchName, "match_%04d_VS_%04d.jpg", frame_ref.frame_index, frame_new.frame_index);
      sprintf(frameMatchName, "frame matching");
      cv::imwrite("frame_match.jpg", img_matches);
      cvWaitKey(1);
    }
#if 1


    RefineByRotation(frame_ref, frame_new, init_matches);
    rotation_inliers = init_matches.size();
#endif

    // use ransac to remove outliers
    int candidate_num = 8;
    float min_distance_threshold = 0.015;
    int inliers_num_first, inliers_num_second;			// make sure 90% are inliers
    inliers_num_first = matched_feature_pairs;
    std::vector< DMatch > matches_before_filtering = init_matches;
 //   RefineByRotation(frame_ref, frame_new, init_matches);
 //   RefineByRotation(frame_ref, frame_new, init_matches);
 //   RefineByRotation(frame_ref, frame_new, init_matches);
 //   RefineByRotation(frame_ref, frame_new, init_matches);
 //   RefineByRotation(frame_ref, frame_new, init_matches);
    outlierFiltering(frame_ref, frame_new, init_matches);
    outlierFiltering(frame_ref, frame_new, init_matches);
    outlierFiltering(frame_ref, frame_new, init_matches);
    outlierFiltering(frame_ref, frame_new, init_matches);
    outlierFiltering(frame_ref, frame_new, init_matches);



    int ransac_input_num = init_matches.size();

    double reprojection_error;


    reprojection_error = ransac3D3D(frame_ref,
                                    frame_new,
                                    init_matches,
                                    matches_before_filtering,
                                    g_para.reprojection_error_3d_threshold,
                                    g_para.ransac_maximum_iterations,
                                    fCorr,
                                    relative_pose_from_ref_to_new,
                                    camera_para);

    start = clock();

    /********************** fine search **********************/
    {
      // refine binary feature search results
      Eigen::MatrixXd H = relative_pose_from_ref_to_new.matrix3x4();
      std::vector< DMatch > predict_matches;
      std::vector<cv::KeyPoint> predict_ref_points(frame_ref.local_points.size());
      for (int i = 0; i < frame_ref.local_points.size(); i++)
      {
        Eigen::Vector4d homo_points;
        homo_points << frame_ref.local_points[i](0), frame_ref.local_points[i](1), frame_ref.local_points[i](2), 1;
        Eigen::Vector3d predict_points = H*homo_points;
        predict_points = predict_points / predict_points(2);
        cv::KeyPoint predict_ref = frame_ref.keypoints[i];
        predict_ref.pt.x = predict_points(0) * camera_para.c_fx + camera_para.c_cx;
        predict_ref.pt.y = predict_points(1) * camera_para.c_fy + camera_para.c_cy;
        predict_ref_points[i] = predict_ref;
        cv::DMatch m;
        m.queryIdx = i;
        m.trainIdx = i;
        if (i % 20 == 0)
        {
          predict_matches.push_back(m);

        }
      }

      if (g_para.runTestFlag)
      {
#if 0
        Mat img_matches;
        cv::drawMatches(frame_ref.rgb, frame_ref.keypoints, frame_new.rgb, predict_ref_points,
          predict_matches, img_matches);
        char frameMatchName[256];
        memset(frameMatchName, '\0', 256);
        //	sprintf(frameMatchName, "match_%04d_VS_%04d.jpg", frame_ref.frame_index, frame_new.frame_index);
        sprintf(frameMatchName, "frame matching predict");
        cv::imshow(frameMatchName, img_matches);
        cvWaitKey(1);
#endif

      }
      init_matches.clear();
      frame_new_matcher.search_8_with_range(frame_ref.descriptor, init_matches, frame_new.keypoints, predict_ref_points, 30,
                                              g_para.hamming_distance_threshold * 1.5);
      //RefineByRotation(frame_ref, frame_new, init_matches);
      std::vector< DMatch > complete_matches = init_matches;
      outlierFiltering(frame_ref, frame_new, init_matches);
      outlierFiltering(frame_ref, frame_new, init_matches);
      outlierFiltering(frame_ref, frame_new, init_matches);
      outlierFiltering(frame_ref, frame_new, init_matches);
      reprojection_error = ransac3D3D(frame_ref, frame_new, init_matches, complete_matches, g_para.reprojection_error_3d_threshold,
        g_para.ransac_maximum_iterations, fCorr, relative_pose_from_ref_to_new,camera_para);
    //  cout << "reprojection error: " << reprojection_error << endl;
    }

    end = clock();
    time_refine = end - start ;

    start = clock();


    std::vector<float> feature_weight_lp = fCorr.weight_per_feature;

    reprojection_error_feature_based = reprojection_error;
    float scale_increase = 0, scale_decrease = 0;
    for (int i = 0; i < init_matches.size(); i++)
    {
      cv::KeyPoint predict_new_2D = frame_ref.keypoints[init_matches[i].queryIdx];
      cv::KeyPoint predict_ref_2D = frame_new.keypoints[init_matches[i].trainIdx];
      if (predict_new_2D.octave >  predict_ref_2D.octave)
      {
        scale_increase++;
      }
      if (predict_new_2D.octave < predict_ref_2D.octave)
      {
        scale_decrease++;
      }
    }
    scale_change_ratio = fmax(scale_decrease, scale_increase) / (init_matches.size()+1);

    if(reprojection_error < REPROJECTION_TH)
    {
      sparse_success = 1;
    }

    if(reprojection_error < REPROJECTION_TH)
    {
      fCorr.preIntegrate();
      average_disparity = fCorr.calculate_average_disparity(camera_para);
//        fCorr.clear_memory();
      matching_success = 1;
    }



    end = clock();
    float time_finishing = end - start;

    if(0)
    {
        LOG_INFO(1) << "average reprojection error : " << reprojection_error_3Dto3D(fCorr,(relative_pose_from_ref_to_new)) << "    "
                 << "average disparity: "<< average_disparity << "    "
                 << "scale change ratio: " << scale_change_ratio << endl;
        LOG_INFO(1) << "sparse match: " << sparse_success << " " << init_matches.size() << " " << reprojection_error  << "    " << endl;
        LOG_INFO(1) << "run time: featureMatching/rotationFilter/filter/ransac/refine/finishing: "
                 << time_feature_matching << "/"
                 << time_rotation_filter << "/"
                 << time_filter << "/"
                 << time_ransac << "/"
                 << time_refine << "/"
                 << time_finishing << endl;
    }


    return matching_success;
  }

  void ComputeJacobianInfo(FrameCorrespondence &fC,
    Eigen::MatrixXd &Pre_JiTr,
    Eigen::MatrixXd &Pre_JjTr,
    Eigen::MatrixXd &Pre_JiTJi,
    Eigen::MatrixXd &Pre_JiTJj,
    Eigen::MatrixXd &Pre_JjTJi,
    Eigen::MatrixXd &Pre_JjTJj)
  {
    int valid_3d_cnt = fC.sparse_feature_cnt +  fC.dense_feature_cnt;
    // construct the four matrix based on pre-integrated points
    Pre_JiTr.setZero();
    Pre_JjTr.setZero();
    Pre_JiTJi.setZero();
    Pre_JiTJj.setZero();
    Pre_JjTJi.setZero();
    Pre_JjTJj.setZero();
    if (valid_3d_cnt < minimum_3d_correspondence)
    {
      return;
    }
    //prepare data
    Eigen::Matrix3d R_ref = fC.frame_ref.pose_sophus[0].rotationMatrix();
    Eigen::Vector3d t_ref = fC.frame_ref.pose_sophus[0].translation();
    Eigen::Matrix3d R_new = fC.frame_new.pose_sophus[0].rotationMatrix();
    Eigen::Vector3d t_new = fC.frame_new.pose_sophus[0].translation();
    Eigen::Matrix3d Eye3x3;

    Eye3x3.setIdentity();
    Eigen::Matrix3d riWrj, riWri, rjWrj;
    riWrj = R_ref * fC.sum_p_ref_new * R_new.transpose();
    riWri = R_ref * fC.sum_p_ref_ref * R_ref.transpose();
    rjWrj = R_new * fC.sum_p_new_new * R_new.transpose();

    Eigen::Vector3d R_ref_sum_p_ref = R_ref * fC.sum_p_ref;
    Eigen::Vector3d R_new_sum_p_new = R_new * fC.sum_p_new;
    Eigen::Vector3d residual = R_ref_sum_p_ref + fC.sum_weight * (t_ref - t_new) - R_new_sum_p_new;
    //calculating JTr, see ProblemFormulation.pdf
    Pre_JiTr.block<3, 1>(0, 0) = residual;
    Pre_JiTr.block<3, 1>(3, 0) = Eigen::Vector3d(riWrj(2, 1) - riWrj(1, 2), -riWrj(2, 0) + riWrj(0, 2), riWrj(1, 0) - riWrj(0, 1))
      + R_ref_sum_p_ref.cross(t_ref - t_new) + t_ref.cross(residual);

    Pre_JjTr.block<3, 1>(0, 0) = residual;
    Pre_JjTr.block<3, 1>(3, 0) = Eigen::Vector3d(riWrj(2, 1) - riWrj(1, 2), -riWrj(2, 0) + riWrj(0, 2), riWrj(1, 0) - riWrj(0, 1))
      + R_new_sum_p_new.cross(t_ref - t_new) + t_new.cross(residual);
    Pre_JjTr = -Pre_JjTr;

    //calculating JTJ
    Pre_JiTJi.block<3, 3>(0, 0) = Eye3x3 *fC.sum_weight;
    Pre_JiTJi.block<3, 3>(0, 3) = -getSkewSymmetricMatrix(R_ref_sum_p_ref + fC.sum_weight * t_ref);
    Pre_JiTJi.block<3, 3>(3, 0) = -Pre_JiTJi.block<3, 3>(0, 3);
    Pre_JiTJi(3, 3) = riWri(2, 2) + riWri(1, 1);	Pre_JiTJi(3, 4) = -riWri(1, 0);					Pre_JiTJi(3, 5) = -riWri(2, 0);
    Pre_JiTJi(4, 3) = -riWri(0, 1);					Pre_JiTJi(4, 4) = riWri(0, 0) + riWri(2, 2);	Pre_JiTJi(4, 5) = -riWri(2, 1);
    Pre_JiTJi(5, 3) = -riWri(0, 2);					Pre_JiTJi(5, 4) = -riWri(1, 2);					Pre_JiTJi(5, 5) = riWri(0, 0) + riWri(1, 1);
    Pre_JiTJi.block<3, 3>(3, 3) += -skewMatrixProduct(t_ref, R_ref_sum_p_ref) - skewMatrixProduct(R_ref_sum_p_ref, t_ref) - fC.sum_weight * 1 * skewMatrixProduct(t_ref, t_ref);

    Pre_JjTJj.block<3, 3>(0, 0) = Eye3x3 *fC.sum_weight;
    Pre_JjTJj.block<3, 3>(0, 3) = -getSkewSymmetricMatrix(R_new_sum_p_new + fC.sum_weight * t_new);
    Pre_JjTJj.block<3, 3>(3, 0) = -Pre_JjTJj.block<3, 3>(0, 3);
    Pre_JjTJj(3, 3) = rjWrj(2, 2) + rjWrj(1, 1);	Pre_JjTJj(3, 4) = -rjWrj(1, 0);					Pre_JjTJj(3, 5) = -rjWrj(2, 0);
    Pre_JjTJj(4, 3) = -rjWrj(0, 1);					Pre_JjTJj(4, 4) = rjWrj(0, 0) + rjWrj(2, 2);	Pre_JjTJj(4, 5) = -rjWrj(2, 1);
    Pre_JjTJj(5, 3) = -rjWrj(0, 2);					Pre_JjTJj(5, 4) = -rjWrj(1, 2);					Pre_JjTJj(5, 5) = rjWrj(0, 0) + rjWrj(1, 1);
    Pre_JjTJj.block<3, 3>(3, 3) += -skewMatrixProduct(t_new, R_new_sum_p_new) - skewMatrixProduct(R_new_sum_p_new, t_new) - fC.sum_weight * 1 * skewMatrixProduct(t_new, t_new);


    Pre_JiTJj.block<3, 3>(0, 0) = Eye3x3 *fC.sum_weight;
    Pre_JiTJj.block<3, 3>(0, 3) = -getSkewSymmetricMatrix(R_new_sum_p_new + fC.sum_weight * t_new);
    Pre_JiTJj.block<3, 3>(3, 0) = -getSkewSymmetricMatrix(R_ref_sum_p_ref + fC.sum_weight * t_ref).transpose();
    Pre_JiTJj(3, 3) = riWrj(2, 2) + riWrj(1, 1);	Pre_JiTJj(3, 4) = -riWrj(1, 0);	Pre_JiTJj(3, 5) = -riWrj(2, 0);
    Pre_JiTJj(4, 3) = -riWrj(0, 1);	Pre_JiTJj(4, 4) = riWrj(0, 0) + riWrj(2, 2);	Pre_JiTJj(4, 5) = -riWrj(2, 1);
    Pre_JiTJj(5, 3) = -riWrj(0, 2);	Pre_JiTJj(5, 4) = -riWrj(1, 2);		Pre_JiTJj(5, 5) = riWrj(0, 0) + riWrj(1, 1);
    Pre_JiTJj.block<3, 3>(3, 3) += -skewMatrixProduct(t_ref, R_new_sum_p_new) - skewMatrixProduct(R_ref_sum_p_ref, t_new) - fC.sum_weight * 1 * skewMatrixProduct(t_ref, t_new);
    Pre_JiTJj = -Pre_JiTJj;
    Pre_JjTJi = Pre_JiTJj.transpose();

  #if 0
    cout << "precomputing jacobian matrics: " << fC.frame_new.frame_index << " " << fC.frame_ref.frame_index << endl;
    cout << "JiTr:" << Pre_JiTr.transpose() << endl;
    cout << "JjTr:" << Pre_JjTr.transpose() << endl;
    cout << "JiTJi: " << endl << Pre_JiTJi << endl;
    cout << "JiTJj: " << endl << Pre_JiTJj << endl;
    cout << "JjTJi: " << endl << Pre_JjTJi << endl;
    cout << "JjTJj: " << endl << Pre_JjTJj << endl;
  #endif
  }

  template <class NumType>
  void addBlockToTriplets(std::vector<Eigen::Triplet<NumType>> &coeff, Eigen::MatrixXd b,
    int start_x,int start_y)
  {
    int rows = b.rows();
    int cols = b.cols();
    for (int i = 0; i < rows; i++)
    {
      for (int j = 0; j < cols; j++)
      {
        coeff.push_back(Eigen::Triplet<NumType>(start_x+i,start_y+j,b(i,j)));
      }
    }


  }

typedef double SPARSE_MATRIX_NUM_TYPE;
#define USE_ROBUST_COST

  float initGraphHuberNorm(std::vector<FrameCorrespondence> &fCList, std::vector<Frame> &F)
  {

      int origin = 0;
      vector<float> average_error_per_frame(F.size());
      vector<int> keyframe_candidate_fcorrs;

      std::vector<int> keyframes;
      for (int i = 0; i < F.size(); i++)
      {
        if (F[i].is_keyframe && F[i].origin_index == origin)
        {
          keyframes.push_back(i);
        }
      }
      for (int i = 0; i < keyframes.size(); i++)
      {
        LOG_INFO(1) << i << " " << keyframes[i] << endl;
      }
      if (keyframes.size() < 3)
      {
        LOG_INFO(1) << "no need to optimize!" << endl;
        return -1;
      }
      int N = F.size();
      std::vector<int> getKeyFramePos(N);
      for (int i = 0; i < N; i++)
      {
        getKeyFramePos[i] = -1;
      }
      for (int i = 0; i < keyframes.size(); i++)
      {
        getKeyFramePos[keyframes[i]] = i;
      }

      for (int i = 0; i < fCList.size(); i++)
      {
          Frame &frame_ref = fCList[i].frame_ref;
        Frame &frame_new = fCList[i].frame_new;
        int frame_ref_pos = getKeyFramePos[frame_ref.frame_index];
        int frame_new_pos = getKeyFramePos[frame_new.frame_index];

        if (frame_ref_pos < 0 || frame_new_pos < 0)
        {
          continue;
        }
        keyframe_candidate_fcorrs.push_back(i);
        float error = reprojection_error_3Dto3D(fCList[i]);
        if(g_para.debug_mode)
        {
  #if 0
          LOG_INFO(1) << frame_ref.frame_index << " " << frame_new.frame_index
                   << " " << error
                   << " " << fCList[i].sum_weight
                   << endl;
  #endif
        }

      }


      for (int i = 0; i < keyframe_candidate_fcorrs.size(); i++)
      {
        Frame &frame_ref = fCList[keyframe_candidate_fcorrs[i]].frame_ref;
        Frame &frame_new = fCList[keyframe_candidate_fcorrs[i]].frame_new;
        int frame_ref_pos = getKeyFramePos[frame_ref.frame_index];
        int frame_new_pos = getKeyFramePos[frame_new.frame_index];

        if (frame_ref_pos < 0 || frame_new_pos < 0)
        {
          continue;
        }

        fCList[keyframe_candidate_fcorrs[i]].preIntegrateWithHuberNorm();

      }

  }

  float optimizeKeyFrameMapRobust(vector<FrameCorrespondence> &fCList, vector<Frame> &F,
                            std::vector<MultiViewGeometry::KeyFrameDatabase> &kflist,int origin, float robust_u)
  {
    vector<float> average_error_per_frame(F.size());
    vector<int> keyframe_candidate_fcorrs;

    std::vector<int> keyframes;
    for (int i = 0; i < F.size(); i++)
    {
      if (F[i].is_keyframe && F[i].origin_index == origin)
      {
        keyframes.push_back(i);
      }
    }
    for (int i = 0; i < keyframes.size(); i++)
    {
      LOG_INFO(1) << i << " " << keyframes[i] << endl;
    }
    if (keyframes.size() < 3)
    {
      LOG_INFO(1) << "no need to optimize!" << endl;
      return -1;
    }
    int N = F.size();
    std::vector<int> getKeyFramePos(N);
    for (int i = 0; i < N; i++)
    {
      getKeyFramePos[i] = -1;
    }
    for (int i = 0; i < keyframes.size(); i++)
    {
      getKeyFramePos[keyframes[i]] = i;
    }

    int latest_keyframe_index = 0;
    for (int i = 0; i < fCList.size(); i++)
    {
        Frame &frame_ref = fCList[i].frame_ref;
      Frame &frame_new = fCList[i].frame_new;
      int frame_ref_pos = getKeyFramePos[frame_ref.frame_index];
      int frame_new_pos = getKeyFramePos[frame_new.frame_index];

      if (frame_ref_pos < 0 || frame_new_pos < 0)
      {
        continue;
      }
      latest_keyframe_index = frame_ref.frame_index > latest_keyframe_index ? frame_ref.frame_index : latest_keyframe_index;
      latest_keyframe_index = frame_new.frame_index > latest_keyframe_index ? frame_new.frame_index : latest_keyframe_index;

      keyframe_candidate_fcorrs.push_back(i);
      float error = reprojection_error_3Dto3D(fCList[i]);
      if(g_para.debug_mode)
      {
#if 0
        LOG_INFO(1) << frame_ref.frame_index << " " << frame_new.frame_index
                 << " " << error
                 << " " << fCList[i].sum_weight
                 << endl;
#endif
      }

    }



    std::vector<float> weight_per_pair(keyframe_candidate_fcorrs.size());

    if(g_para.debug_mode)
    {
      double init_error = reprojection_error_3Dto3D(fCList, keyframe_candidate_fcorrs);
      LOG_INFO(1) << "init error		: " << init_error << endl;
    }

    // will be replaced by conjugate gradient descent.
    int optNum = keyframes.size() - 1;
    Eigen::MatrixXd J, err;
    Eigen::MatrixXd delta(6 * optNum, 1), JTe(6 * optNum, 1);
    Eigen::SparseMatrix<SPARSE_MATRIX_NUM_TYPE> JTJ(6 * optNum, 6 * optNum);

    int valid_observation_cnt = 0;
    double prev_err = 10000;

    clock_t start, end;

    // the solver is only built at the first iteration
    Eigen::SimplicialLDLT	<Eigen::SparseMatrix<SPARSE_MATRIX_NUM_TYPE> > SimplicialLDLTSolver;
    std::vector<Eigen::Triplet<SPARSE_MATRIX_NUM_TYPE>> coeff;
    coeff.reserve(6 * 6 * 4 * fCList.size());
    Eigen::MatrixXd JiTJi_pre(6, 6), JiTJj_pre(6, 6), JjTJi_pre(6, 6), JjTJj_pre(6, 6), JiTe_pre(6, 1), JjTe_pre(6, 1);

    clock_t start_opt, end_opt;
    double time_opt;
    start_opt = clock();

    PoseSE3dList frame_poses;
    for(int i = 0; i < F.size(); i++)
    {
        frame_poses.push_back(F[i].pose_sophus[0]);
    }
    vector<FrameCorrespondence> optimized_fc;
    for (int i = 0; i < keyframe_candidate_fcorrs.size(); i++)
    {
        Frame &frame_ref = fCList[keyframe_candidate_fcorrs[i]].frame_ref;
        Frame &frame_new = fCList[keyframe_candidate_fcorrs[i]].frame_new;

        if(frame_ref.frame_index == latest_keyframe_index || frame_new.frame_index == latest_keyframe_index)
        {
            continue;
        }
        optimized_fc.push_back(fCList[keyframe_candidate_fcorrs[i]]);
    }

    float init_error = reprojection_error_3Dto3D(optimized_fc);
    float init_total_error = reprojection_error_3Dto3D(fCList, keyframe_candidate_fcorrs);


    for (int iter = 0; iter < 3; iter++)
    {
      JTe.setZero();
      JTJ.setZero();
      err.setZero();
      coeff.clear();
      valid_observation_cnt = 0;

      double time_framePair;
      double time_generatingJacobian;
      double time_buildSolver;
      start = clock();
      float robust_weight;


      for (int i = 0; i < keyframe_candidate_fcorrs.size(); i++)
      {
        Frame &frame_ref = fCList[keyframe_candidate_fcorrs[i]].frame_ref;
        Frame &frame_new = fCList[keyframe_candidate_fcorrs[i]].frame_new;
        int frame_ref_pos = getKeyFramePos[frame_ref.frame_index];
        int frame_new_pos = getKeyFramePos[frame_new.frame_index];

        if (frame_ref_pos < 0 || frame_new_pos < 0)
        {
          continue;
        }
        //fCList[keyframe_candidate_fcorrs[i]].preIntegrateWithHuberNorm();

#if 0
        float error = reprojection_error_3Dto3D(fCList[keyframe_candidate_fcorrs[i]]);
        robust_weight = robust_u / (robust_u + error);
#else
        robust_weight = 1.0f;
#endif

        clock_t start_jacobian, end_jacobian;
        start_jacobian = clock();
        ComputeJacobianInfo(fCList[keyframe_candidate_fcorrs[i]],
          JiTe_pre,
          JjTe_pre,
          JiTJi_pre,
          JiTJj_pre,
          JjTJi_pre,
          JjTJj_pre);

        JiTe_pre *= robust_weight;
        JjTe_pre *= robust_weight;
        JiTJi_pre *= robust_weight;
        JiTJj_pre *= robust_weight;
        JjTJj_pre *= robust_weight;
        JjTJi_pre *= robust_weight;

        if (frame_ref_pos == 0)
        {
          addBlockToTriplets(coeff, JjTJj_pre, (frame_new_pos - 1) * 6, (frame_new_pos - 1) * 6);
          JTe.block<6, 1>((frame_new_pos - 1) * 6, 0) += JjTe_pre;
        }
        else
        {
          addBlockToTriplets(coeff, JiTJi_pre, (frame_ref_pos - 1) * 6, (frame_ref_pos - 1) * 6);
          addBlockToTriplets(coeff, JiTJj_pre, (frame_ref_pos - 1) * 6, (frame_new_pos - 1) * 6);
          addBlockToTriplets(coeff, JjTJi_pre, (frame_new_pos - 1) * 6, (frame_ref_pos - 1) * 6);
          addBlockToTriplets(coeff, JjTJj_pre, (frame_new_pos - 1) * 6, (frame_new_pos - 1) * 6);
          JTe.block<6, 1>((frame_ref_pos - 1) * 6, 0) += JiTe_pre;
          JTe.block<6, 1>((frame_new_pos - 1) * 6, 0) += JjTe_pre;
        }
      }

      end = clock();
      time_framePair = (double)(end - start) / CLOCKS_PER_SEC * 1000;
      start = clock();
      JTJ.setFromTriplets(coeff.begin(), coeff.end());
      end = clock();
      time_generatingJacobian = (double)(end - start) / CLOCKS_PER_SEC * 1000;
      start = clock();
      SimplicialLDLTSolver.compute(JTJ);
      end = clock();
      time_buildSolver = (double)(end - start) / CLOCKS_PER_SEC * 1000;


      // update the pose of each frame
      start = clock();
      delta = SimplicialLDLTSolver.solve(JTe);
      end = clock();
      double time_svd = (double)(end - start) / CLOCKS_PER_SEC * 1000;
      for (int i = 1; i < keyframes.size(); i++)
      {
          Eigen::VectorXd delta_i = delta.block<6, 1>(6 * (i - 1), 0);
          if(isnan(delta_i(0)))
          {
            cout << "nan detected in pose update! " << endl;
            continue;
          }

          F[keyframes[i]].pose_sophus[0] = Sophus::SE3d::exp(delta_i).inverse() *
                  F[keyframes[i]].pose_sophus[0];
      }

      double time_calculate_reprojection_error;
      double refined_error;
      if(g_para.debug_mode)
      {
        start = clock();
        float average_reprojection_error = 0;
        float count = 0;
        for (int i = 0; i < keyframe_candidate_fcorrs.size(); i++)
        {
          average_reprojection_error += reprojection_error_3Dto3D(fCList[keyframe_candidate_fcorrs[i]]) * fCList[keyframe_candidate_fcorrs[i]].sum_weight;
          count += fCList[keyframe_candidate_fcorrs[i]].sum_weight;
        }
        refined_error =  average_reprojection_error / count;

        end = clock();
        time_calculate_reprojection_error = (double)(end - start) / CLOCKS_PER_SEC * 1000;
       if (fabs(refined_error - prev_err) < 1e-11)
        {
          break;
        }
        prev_err = refined_error;
      }
//      cout << "global refined error		: "
//               << refined_error << " "
//               << time_framePair << " "
//               << time_generatingJacobian << " "
//               << time_buildSolver << " "
//               << time_svd << " "
//               << time_calculate_reprojection_error << endl;
    }

    end_opt = clock();

    time_opt = (double )(end_opt - start_opt) / CLOCKS_PER_SEC * 1000;
    // update local frame
    for (int i = 0; i < kflist.size(); i++)
    {
      for (int j = 0; j < kflist[i].corresponding_frames.size(); j++)
      {

          // local_pose  =keypose * relative_from_key_to_current;
        F[kflist[i].corresponding_frames[j]].pose_sophus[0] = F[kflist[i].keyFrameIndex].pose_sophus[0] *
                kflist[i].relative_pose_from_key_to_current[j];

      }
    }

    float final_error = reprojection_error_3Dto3D(optimized_fc);
    float final_total_error = reprojection_error_3Dto3D(fCList,keyframe_candidate_fcorrs);
    cout << "init/final error " << init_error << "/" << final_error
         << "       " << init_total_error << "/" << final_total_error << endl;



    // remove outliers
    if((final_error - init_error) / init_error > 0.05)
    {
        // remove all outliers

        int min_dist = 1e4;
        for (int i = 0; i < keyframe_candidate_fcorrs.size(); i++)
        {
            Frame &frame_ref = fCList[keyframe_candidate_fcorrs[i]].frame_ref;
            Frame &frame_new = fCList[keyframe_candidate_fcorrs[i]].frame_new;

            int frame_ref_pos = getKeyFramePos[frame_ref.frame_index];
            int frame_new_pos = getKeyFramePos[frame_new.frame_index];
            if(frame_ref.frame_index == latest_keyframe_index || frame_new.frame_index == latest_keyframe_index)
            {
                if(fabs(frame_ref_pos - frame_new_pos) < min_dist && fCList[keyframe_candidate_fcorrs[i]].matches.size() > 0)
                {
                    cout << "try to remove: "<< fabs(frame_ref_pos - frame_new_pos) << " " << min_dist << " " << frame_ref.frame_index << " " << frame_new.frame_index << endl;

                    min_dist = fabs(frame_ref_pos - frame_new_pos);
                }
            }
        }
        for (int i = 0; i < keyframe_candidate_fcorrs.size(); i++)
        {
            Frame &frame_ref = fCList[keyframe_candidate_fcorrs[i]].frame_ref;
            Frame &frame_new = fCList[keyframe_candidate_fcorrs[i]].frame_new;

            if(fCList[keyframe_candidate_fcorrs[i]].matches.size() > 0)
            {

            }
            int frame_ref_pos = getKeyFramePos[frame_ref.frame_index];
            int frame_new_pos = getKeyFramePos[frame_new.frame_index];
            if(frame_ref.frame_index == latest_keyframe_index || frame_new.frame_index == latest_keyframe_index)
            {
                if(fabs(frame_ref_pos - frame_new_pos) > min_dist)
                {
                    fCList[keyframe_candidate_fcorrs[i]].reset();
                }
            }
        }
        for(int i = 0; i < F.size(); i++)
        {
            F[i].pose_sophus[0] = frame_poses[i];
        }

    }
    return prev_err;

  }


  float optimizeKeyFrameMap(vector<FrameCorrespondence> &fCList, vector<Frame> &F,
                            std::vector<MultiViewGeometry::KeyFrameDatabase> &kflist,int origin)
  {
    float robust_u = 1;
     for(int i = 0; i < 1; i++)
     {
       robust_u = 0.5 * robust_u;
       optimizeKeyFrameMapRobust(fCList,F,kflist,origin,robust_u);
     }
  }




  float reprojection_error_3Dto3D(const FrameCorrespondence &fC,  const Sophus::SE3d &relative_pose_from_ref_to_new)
  {
    Eigen::MatrixXd R_ref = relative_pose_from_ref_to_new.rotationMatrix();
    Eigen::Vector3d t_ref = relative_pose_from_ref_to_new.translation();
    // pre-integration method for norm-2 distance
    float total_error = 0;

    if (fC.sum_weight > 0)
    {
      total_error = fC.sum_p_ref_ref(0,0) + fC.sum_p_ref_ref(1,1) + fC.sum_p_ref_ref(2,2) +
          fC.sum_p_new_new(0,0) + fC.sum_p_new_new(1,1) + fC.sum_p_new_new(2,2) +
          fC.sum_weight * t_ref.transpose() * t_ref
        - 2 * (float)(t_ref.transpose() * fC.sum_p_new) + 2 * (float)(t_ref.transpose() * R_ref * fC.sum_p_ref)
        - 2 * R_ref.cwiseProduct(fC.sum_p_new_ref).sum();

      if(total_error < 0)
      {
        cout << "total error: " << total_error << endl;

      }
      else
      {
        total_error = sqrt(total_error)  / fC.sum_weight;
      }
    }
    return total_error;
  }
  float reprojection_error_3Dto3D(const FrameCorrespondence &fC)
  {
    return reprojection_error_3Dto3D(fC, fC.frame_new.pose_sophus[0].inverse() * fC.frame_ref.pose_sophus[0]);
  }

  float reprojection_error_3Dto3D(std::vector<FrameCorrespondence> fCList)
  {
    float average_reprojection_error = 0;
    float count = 0;
    for (int i = 0; i < fCList.size(); i++)
    {
      average_reprojection_error += reprojection_error_3Dto3D(fCList[i]) * fCList[i].sum_weight;
      count += fCList[i].sum_weight;
    }
    return average_reprojection_error / count;
  }


  float reprojection_error_3Dto3D(std::vector<FrameCorrespondence> fCList, std::vector<int>candidates)
  {
    float average_reprojection_error = 0;
    float count = 0;
    for (int i = 0; i < candidates.size(); i++)
    {
      average_reprojection_error += reprojection_error_3Dto3D(fCList[candidates[i]]) * fCList[candidates[i]].sum_weight;
      count += fCList[candidates[i]].sum_weight;
    }
    return average_reprojection_error / count;
  }
  float reprojection_error_3Dto3D_perFrame(std::vector<FrameCorrespondence> fCList,
        Eigen::VectorXd &errorPerFrame,
        Eigen::VectorXd &pointsPerFrame,
        Eigen::VectorXd &connectionsPerFrame)
    {
        int frameNum = errorPerFrame.size();
        errorPerFrame.setZero();
        pointsPerFrame.setZero();
        connectionsPerFrame.setZero();
        float average_reprojection_error = 0;
        float count = 0;
        for (int i = 0; i < fCList.size(); i++)
        {
            average_reprojection_error += reprojection_error_3Dto3D(fCList[i]) * fCList[i].sum_weight;
            count += fCList[i].sum_weight;

            int ref_frame_index = fCList[i].frame_ref.frame_index;
            int new_frame_index = fCList[i].frame_new.frame_index;
            assert(ref_frame_index < frameNum);
            assert(new_frame_index < frameNum);
            errorPerFrame[ref_frame_index] += reprojection_error_3Dto3D(fCList[i]) * fCList[i].sum_weight;
            errorPerFrame[new_frame_index] += reprojection_error_3Dto3D(fCList[i]) * fCList[i].sum_weight;
            pointsPerFrame[ref_frame_index] += fCList[i].sum_weight;
            pointsPerFrame[new_frame_index] += fCList[i].sum_weight;
            connectionsPerFrame[ref_frame_index] += 1;
            connectionsPerFrame[new_frame_index] += 1;
        }

        for (int i = 0; i < frameNum; i++)
        {
            if (pointsPerFrame[i] < 1)
                errorPerFrame[i] = 1e8;
            else
                errorPerFrame[i] /= pointsPerFrame[i];
        }
        return average_reprojection_error / count;
    }
  float reprojection_error_3Dto3D_perFrame(std::vector<FrameCorrespondence> fCList,
        Eigen::VectorXd &errorPerFrame,
        Eigen::VectorXd &pointsPerFrame,
        Eigen::VectorXd &connectionsPerFrame,
    std::vector<std::vector<int> > &related_connections)
    {
        int frameNum = errorPerFrame.size();
        errorPerFrame.setZero();
        pointsPerFrame.setZero();
        float average_reprojection_error = 0;
        int count = 0;
        for (int i = 0; i < fCList.size(); i++)
        {
            average_reprojection_error += reprojection_error_3Dto3D(fCList[i]) * fCList[i].sum_weight;
            count += fCList[i].sum_weight;

            int ref_frame_index = fCList[i].frame_ref.frame_index;
            int new_frame_index = fCList[i].frame_new.frame_index;
            assert(ref_frame_index < frameNum);
            assert(new_frame_index < frameNum);
            errorPerFrame[ref_frame_index] += reprojection_error_3Dto3D(fCList[i]) * fCList[i].sum_weight;
            errorPerFrame[new_frame_index] += reprojection_error_3Dto3D(fCList[i]) * fCList[i].sum_weight;
            pointsPerFrame[ref_frame_index] += fCList[i].sum_weight;
            pointsPerFrame[new_frame_index] += fCList[i].sum_weight;
            connectionsPerFrame[ref_frame_index] += 1;
            connectionsPerFrame[new_frame_index] += 1;
            related_connections[ref_frame_index].push_back(i);
            related_connections[new_frame_index].push_back(i);
        }

        for (int i = 0; i < frameNum; i++)
        {
            if (pointsPerFrame[i] < 1)
                errorPerFrame[i] = 1e8;
            else
                errorPerFrame[i] /= pointsPerFrame[i];
        }
        return average_reprojection_error / count;
    }

  float reprojection_error_3Dto3D(Point3dList pt_ref,
    Point3dList pt_new,
        Sophus::SE3d relative_pose_from_ref_to_new,
        int use_huber_norm,
        float huber_norm_threshold)
    {
        float reprojection_error_3d = 0;

        if (use_huber_norm)
        {
            for (int i = 0; i < pt_ref.size(); i++)
            {
                Eigen::Vector3d reprojection_error = applyPose(relative_pose_from_ref_to_new,pt_ref[i]) - pt_new[i];

                float error = reprojection_error.norm() / pt_ref[i].z();
                float weight_huber = 1;
                if (error > 0.008)
                {
                    weight_huber = 0.008 / error;
                }
                error = error * error * weight_huber;
                reprojection_error_3d += error;
            }
        }
        else
        {
            for (int i = 0; i < pt_ref.size(); i++)
            {
                Eigen::Vector3d reprojection_error = applyPose(relative_pose_from_ref_to_new,pt_ref[i]) - pt_new[i];
                float error = reprojection_error.norm() / pt_ref[i].z();
                error = error * error;
                reprojection_error_3d += error;
            }
        }
    reprojection_error_3d = sqrt(reprojection_error_3d);
        reprojection_error_3d /= pt_ref.size();
        return reprojection_error_3d;
    }
  float reprojection_error_3Dto3D(Frame frame_ref,
        Frame frame_new,
    std::vector< cv::DMatch > inlier_matches,
        Sophus::SE3d relative_pose_from_ref_to_new,
        int use_huber_norm,
        float huber_norm_threshold)
    {
        Point3dList p_ref, p_new;
        p_ref.clear(); p_new.clear();
        p_ref.reserve(inlier_matches.size());
        p_new.reserve(inlier_matches.size());
        for (size_t i = 0; i < inlier_matches.size(); i++)
        {
            Eigen::Vector3d pt_ref(frame_ref.local_points[inlier_matches[i].queryIdx]);
            Eigen::Vector3d pt_new(frame_new.local_points[inlier_matches[i].trainIdx]);
            p_ref.push_back(pt_ref);
            p_new.push_back(pt_new);
        }
        return reprojection_error_3Dto3D(p_ref, p_new, relative_pose_from_ref_to_new, use_huber_norm, huber_norm_threshold);
    }


  Eigen::Matrix3d skewMatrixProduct(Eigen::Vector3d t1, Eigen::Vector3d t2)
  {
      Eigen::Matrix3d M;
      M(0, 0) = -t1(1)*t2(1) - t1(2)*t2(2); M(0, 1) = t1(1)*t2(0); M(0, 2) = t1(2)*t2(0);
      M(1, 0) = t1(0)*t2(1);	 M(1, 1) = -t1(2)*t2(2) - t1(0)*t2(0); M(1, 2) = t1(2)*t2(1);
      M(2, 0) = t1(0)*t2(2);   M(2, 1) = t1(1)*t2(2); M(2, 2) = -t1(1)*t2(1) - t1(0)*t2(0);
      return M;
  }

  Eigen::Matrix3d getSkewSymmetricMatrix(Eigen::Vector3d t)
  {
      Eigen::Matrix3d t_hat;
      t_hat << 0, -t(2), t(1),
          t(2), 0, -t(0),
          -t(1), t(0), 0;
      return t_hat;
  }


}


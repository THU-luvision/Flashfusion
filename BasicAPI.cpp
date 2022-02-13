
#include <boost/filesystem.hpp>
#include <sys/stat.h>
#include <dirent.h>

#include "BasicAPI.h"
#include "GCSLAM/ORBSLAM/ORBextractor.h"
#include "Tools/RealSenseInterface.h"

using namespace std;
using namespace cv;
using namespace MultiViewGeometry;


namespace BasicAPI
{
void loadGlobalParameters(GlobalParameters &g_para, std::string para_file)
{
    cv::FileStorage fSettings;
    fSettings = cv::FileStorage(para_file.c_str(), cv::FileStorage::READ);
    g_para.runTestFlag = fSettings["runTestFlag"];
    g_para.runRefFrame = fSettings["runRefFrame"];
    g_para.runNewFrame = fSettings["runNewFrame"];
    g_para.runFrameNum = fSettings["runFrameNum"];
    g_para.debug_mode = fSettings["debug_mode"];


    g_para.salient_score_threshold = fSettings["salient_score_threshold"];
    g_para.minimum_disparity = fSettings["minimum_disparity"];
    g_para.ransac_maximum_iterations = fSettings["ransac_maximum_iterations"];
    g_para.reprojection_error_3d_threshold = fSettings["reprojection_error_3d_threshold"];
    g_para.reprojection_error_2d_threshold = fSettings["reprojection_error_2d_threshold"];
    g_para.use_fine_search = fSettings["use_fine_search"];
    g_para.maximum_local_frame_match_num = fSettings["maximum_local_frame_match_num"];
    g_para.remove_outlier_keyframes = fSettings["remove_outlier_keyframes"];
    g_para.keyframe_minimum_distance = fSettings["keyframe_minimum_distance"];
    g_para.maximum_keyframe_match_num = fSettings["maximum_keyframe_match_num"];
    g_para.save_ply_files = fSettings["save_ply_files"];
    g_para.max_feature_num = fSettings["max_feature_num"];
    g_para.use_icp_registration = fSettings["use_icp_registration"];
    g_para.icp_weight = fSettings["icp_weight"];
    g_para.hamming_distance_threshold = fSettings["hamming_distance_threshold"];
    g_para.far_plane_distance = fSettings["far_plane_distance"];
}


void  saveTrajectoryFrameList(std::vector<Frame> &F,std::string fileName)
{
  FILE * fp = fopen(fileName.c_str(), "w+");
  if (fp == NULL)
  {
    std::cerr << "file open error ! " << fileName.c_str() << std::endl;
  }
  for (int i = 0; i < F.size(); i++)
  {
    if (F[i].tracking_success  && F[i].origin_index == 0)
    {
      Sophus::SE3d p = F[i].pose_sophus[0];
      fprintf(fp, "%.8f %.8f %.8f %.8f %.8f %.8f %.8f %.8f\n",
              F[i].time_stamp,
              p.translation()(0),
              p.translation()(1),
              p.translation()(2),
              p.unit_quaternion().coeffs().data()[0],
              p.unit_quaternion().coeffs().data()[1],
              p.unit_quaternion().coeffs().data()[2],
              p.unit_quaternion().coeffs().data()[3]);
    }
  }
  fclose(fp);
}

void  saveTrajectoryKeyFrameList(std::vector<Frame> &F,std::string fileName)
{
  FILE * fp = fopen(fileName.c_str(), "w+");
  if (fp == NULL)
  {
    std::cerr << "file open error ! " << fileName.c_str() << std::endl;
  }
  for (int i = 0; i < F.size(); i++)
  {
    if (F[i].tracking_success  && F[i].origin_index == 0 && F[i].is_keyframe)
    {
      Sophus::SE3d p = F[i].pose_sophus[0];
      fprintf(fp, "%.8f %.8f %.8f %.8f %.8f %.8f %.8f %.8f\n",
              F[i].time_stamp,
              p.translation()(0),
              p.translation()(1),
              p.translation()(2),
              p.unit_quaternion().coeffs().data()[0],
              p.unit_quaternion().coeffs().data()[1],
              p.unit_quaternion().coeffs().data()[2],
              p.unit_quaternion().coeffs().data()[3]);
    }
  }
  fclose(fp);
}

void savePLYFiles(std::string fileName,
                  std::vector<Eigen::Vector3f, Eigen::aligned_allocator<Eigen::Vector3d> > p,
                  std::vector<Eigen::Vector3i, Eigen::aligned_allocator<Eigen::Vector3d> >color)
{

  std::ofstream output_file(fileName.c_str(), std::ios::out | std::ios::trunc);
  int pointNum = fmin(p.size(), color.size());
  output_file << "ply" << std::endl;
  output_file << "format ascii 1.0           { ascii/binary, format version number }" << std::endl;
  output_file << "comment made by Greg Turk  { comments keyword specified, like all lines }" << std::endl;
  output_file << "comment this file is a cube" << std::endl;
  output_file << "element vertex " << pointNum << "           { define \"vertex\" element, 8 of them in file }" << std::endl;
  output_file << "property float x" << std::endl;
  output_file << "property float y" << std::endl;
  output_file << "property float z" << std::endl;
  output_file << "property uchar red" << std::endl;
  output_file << "property uchar green" << std::endl;
  output_file << "property uchar blue" << std::endl;
  output_file << "end_header" << std::endl;
  for (int i = 0; i < pointNum; i++)
  {
    output_file << p[i](0) << " " << p[i](1) << " " << p[i](2) << " "
      << color[i](2) << " " << color[i](1) << " " << color[i](0) << " " << std::endl;
  }
  output_file.close();

}


void savePLYFrame(std::string fileName, const Frame &f, const CameraPara &para)
{

  std::vector<Eigen::Vector3f, Eigen::aligned_allocator<Eigen::Vector3d> > p;
  std::vector<Eigen::Vector3i, Eigen::aligned_allocator<Eigen::Vector3d> >color;
  int width = f.depth.cols;
  int height = f.depth.rows;
  for (int j = 0; j < height; j++)
  {

    for (int i = 0; i < width; i++)
    {
      if (f.depth.at<unsigned short>(j, i) > 0)
      {
        float x, y, z;
        x = (i - para.c_cx) / para.c_fx * (f.depth.at<unsigned short>(j, i)) / para.depth_scale;
        y = (j - para.c_cy) / para.c_fy * (f.depth.at<unsigned short>(j, i)) / para.depth_scale;
        z = f.depth.at<unsigned short>(j, i) / para.depth_scale;

        Eigen::Vector3d v = applyPose(f.pose_sophus[0],Eigen::Vector3d(x,y,z));
        p.push_back(Eigen::Vector3f(v(0),v(1),v(2)));

        color.push_back(Eigen::Vector3i(f.rgb.at<cv::Vec3b>(j, i)[0], f.rgb.at<cv::Vec3b>(j, i)[1], f.rgb.at<cv::Vec3b>(j, i)[2]));
      }
    }
  }
  savePLYFiles(fileName,p,color);
}

int detectAndExtractFeatures(Frame &t, int feature_num, CameraPara para)
{

  int frame_width = t.rgb.cols;
  int frame_height = t.rgb.rows;
  clock_t start, end;
  start = clock();

  cv::Mat feature_desc;
  vector<KeyPoint> feature_points;
#if 0
  Ptr<ORB> orb = ORB::create();
  orb->setMaxFeatures(feature_num);
  orb->detectAndCompute(t.rgb, noArray(), feature_points, feature_desc);
#else

  ORB_SLAM2::ORBextractor orb(feature_num, 1.2, 8, 20, 7);

  Mat grayImage;
  cv::cvtColor(t.rgb, grayImage, CV_RGB2GRAY);
  orb(grayImage, cv::noArray(), feature_points, feature_desc);

#endif
  vector<KeyPoint> undistort_feature_points(feature_points.size());

  end = clock();
  double time_featureExtraction = (double)(end - start) / CLOCKS_PER_SEC * 1000;
  start = clock();
#if 0
  // Fill matrix with points
  int N = feature_points.size();
  cv::Mat mat(N, 2, CV_32F);
  for (int i = 0; i<N; i++)
  {
    mat.at<float>(i, 0) = feature_points[i].pt.x;
    mat.at<float>(i, 1) = feature_points[i].pt.y;
  }
  //distortion for camera 1
  cv::Mat DistCoef(5, 1, CV_32F);

  DistCoef.at<float>(0) = g_para.d0;
  DistCoef.at<float>(1) = g_para.d1;
  DistCoef.at<float>(2) = g_para.d2;
  DistCoef.at<float>(3) = g_para.d3;
  DistCoef.at<float>(4) = g_para.d4;

  cv::Mat K = cv::Mat::eye(3, 3, CV_32F);
  K.at<float>(0, 0) = para.c_fx;
  K.at<float>(1, 1) = para.c_fy;
  K.at<float>(0, 2) = para.c_cx;
  K.at<float>(1, 2) = para.c_cy;
  K.at<float>(2, 2) = 1;
  // Undistort points
  mat = mat.reshape(2);
  cv::undistortPoints(mat, mat, K, DistCoef, cv::Mat(), K);
  mat = mat.reshape(1);

  // Fill undistorted keypoint vector
  undistort_feature_points.resize(N);
  for (int i = 0; i<N; i++)
  {
    cv::KeyPoint kp = feature_points[i];
    kp.pt.x = mat.at<float>(i, 0);
    kp.pt.y = mat.at<float>(i, 1);
    undistort_feature_points[i] = kp;
  }
#else
  for (int i = 0; i<feature_points.size(); i++)
  {
    cv::KeyPoint kp = feature_points[i];
    undistort_feature_points[i] = kp;
  }
#endif

  end = clock();
  double time_undistortion = (double)(end - start) / CLOCKS_PER_SEC * 1000;
  start = clock();
  t.keypoints.reserve(feature_points.size());
  t.local_points.reserve(feature_points.size());

#if 0
  cv::Mat_<float> depth_float_tmp;
  cv::Mat_<float> depth_float_tmp_out;
  t.depth.convertTo(depth_float_tmp, CV_32F);
  cv::bilateralFilter(depth_float_tmp, depth_float_tmp_out, 5, 5, 10); //----
  depth_float_tmp_out.convertTo(t.depth, CV_16U);
#endif

  for (int i = 0; i < feature_points.size(); i++)
  {
    feature_points[i].class_id = t.frame_index;
    cv::Point2f current_pt = feature_points[i].pt;
    float current_depth;
    if (current_pt.x > frame_width - 1 || current_pt.y > frame_height - 1 ||
      current_pt.x < 0 || current_pt.y < 0)
    {
      current_depth = -1;
    }
    else
    {
      current_depth = t.depth.at<unsigned short>(current_pt.y, current_pt.x) / para.depth_scale;
    }

    // current no 2D points are considered
    if (current_depth > 0 && current_depth < para.maximum_depth)
    {
      t.keypoints.push_back(undistort_feature_points[i]);
      cv::Point2f current_undistort_pt = undistort_feature_points[i].pt;
      t.descriptor.push_back(feature_desc.row(i));
      t.local_points.push_back(Eigen::Vector3d(
        (current_undistort_pt.x - para.c_cx) * current_depth / para.c_fx,
        (current_undistort_pt.y - para.c_cy) * current_depth / para.c_fy,
        current_depth));

    }
  }

  end = clock();
  double time_selectfeatures = (double)(end - start) / CLOCKS_PER_SEC * 1000;

  start = clock();
  return 1;
}


void refineDepthUseNormal(float *normal, float *depth,
                                 float fx, float fy, float cx, float cy,float width, float height)
{
    int numPixels = height * width;
    for(int i = 0; i < height; i++)
    {
        for(int j = 0; j < width; j++)
        {

            int pos = i * width + j;
            Eigen::Vector3f viewAngle = Eigen::Vector3f((j-cx)/fx,(i-cy)/fy,1);
            viewAngle.normalize();
            Eigen::Vector3f normalAngle = Eigen::Vector3f(normal[pos], normal[pos + numPixels],normal[pos + numPixels * 2]);
            float viewQuality = viewAngle.transpose() * normalAngle;
//            normal[pos] = viewQuality;
//            normal[pos + numPixels] = viewQuality;
//            normal[pos + numPixels * 2] = viewQuality;

            if(fabs(viewQuality) < 0.3)
            {
                depth[pos] = 0;
                normal[pos] = 0;
                normal[pos + numPixels] = 0;
                normal[pos + numPixels * 2] = 0;
            }
        }
    }
}

void findCubeCorner(Frame &frame_ref, CameraPara &cameraModel)
{
    int width = frame_ref.rgb.cols;
    int height = frame_ref.rgb.rows;

    const float fx = cameraModel.GetFx();
    const float fy = cameraModel.GetFy();
    const float cx = cameraModel.GetCx();
    const float cy = cameraModel.GetCy();
    float * filtered_depth = (float *)frame_ref.refined_depth.data;
    Eigen::MatrixXf transform = frame_ref.pose_sophus[0].matrix().block<3, 4>(0, 0).cast<float>();
    Eigen::Matrix3f rotation = transform.block<3,3>(0,0);
    Eigen::MatrixXf translation = transform.block<3,1>(0,3);

    __m256 maxX = _mm256_set1_ps(-1e8);
    __m256 maxY = _mm256_set1_ps(-1e8);
    __m256 maxZ = _mm256_set1_ps(-1e8);
    __m256 minX = _mm256_set1_ps(1e8);
    __m256 minY = _mm256_set1_ps(1e8);
    __m256 minZ = _mm256_set1_ps(1e8);

    vec8 inc = vec8(0,1,2,3,4,5,6,7);
    for(int i = 0; i < height; i ++)
    {
        for(int j = 0; j < width; j+=8)
        {
            int pos = i * width + j;
            vec8 depth_c = _mm256_loadu_ps(&filtered_depth[pos]);
            vec8 x = inc + vec8(j);
            vec8 y = vec8(i);
            vec8 refLocalVertexX = (x - vec8(cx)) / vec8(fx) * depth_c;
            vec8 refLocalVertexY = (y - vec8(cy)) / vec8(fy) * depth_c;
            vec8 refVX = vec8(rotation(0,0)) * refLocalVertexX + vec8(rotation(0,1)) * refLocalVertexY + vec8(rotation(0,2)) * depth_c + vec8(translation(0));
            vec8 refVY = vec8(rotation(1,0)) * refLocalVertexX + vec8(rotation(1,1)) * refLocalVertexY + vec8(rotation(1,2)) * depth_c + vec8(translation(1));
            vec8 refVZ = vec8(rotation(2,0)) * refLocalVertexX + vec8(rotation(2,1)) * refLocalVertexY + vec8(rotation(2,2)) * depth_c + vec8(translation(2));
            maxX = _mm256_max_ps(refVX.xmm,maxX);
            maxY = _mm256_max_ps(refVY.xmm,maxY);
            maxZ = _mm256_max_ps(refVZ.xmm,maxZ);

            minX = _mm256_min_ps(refVX.xmm,minX);
            minY = _mm256_min_ps(refVY.xmm,minY);
            minZ = _mm256_min_ps(refVZ.xmm,minZ);
        }
    }
    Eigen::Vector3f maxCorner = Eigen::Vector3f(-1e8,-1e8,-1e8);
    Eigen::Vector3f minCorner = Eigen::Vector3f(1e8,1e8,1e8);
    for(int i = 0; i < 8; i++ )
    {
        maxCorner(0) = fmax(maxCorner(0),maxX[i]);
        minCorner(0) = fmin(minCorner(0),minX[i]);

        maxCorner(1) = fmax(maxCorner(1),maxY[i]);
        minCorner(1) = fmin(minCorner(1),minY[i]);

        maxCorner(2) = fmax(maxCorner(2),maxX[i]);
        minCorner(2) = fmin(minCorner(2),minX[i]);
    }
//    std::cout << "find frame index: " << frame_ref.frame_index << std::endl;
//    std::cout << "max Corner :" << maxCorner<< std::endl;
//    std::cout << "min Corner :" << minCorner<< std::endl;
}

void refineNewframesSIMD(Frame &frame_ref, Frame &frame_new,CameraPara &cameraModel)
{
    int width = frame_ref.rgb.cols;
    int height = frame_ref.rgb.rows;

    const float fx = cameraModel.GetFx();
    const float fy = cameraModel.GetFy();
    const float cx = cameraModel.GetCx();
    const float cy = cameraModel.GetCy();

    float * filtered_depth_new = (float *)frame_new.refined_depth.data;
    float * filtered_depth = (float *)frame_ref.refined_depth.data;

    assert(filtered_depth_new != NULL);
    assert(filtered_depth != NULL);

    float threshold = 0.05;

    Eigen::MatrixXf transform_new_to_ref = (frame_ref.pose_sophus[0].inverse()*frame_new.pose_sophus[0]).matrix().block<3, 4>(0, 0).cast<float>();

    Eigen::Matrix3f rotation = transform_new_to_ref.block<3,3>(0,0);
    Eigen::MatrixXf translation = transform_new_to_ref.block<3,1>(0,3);

    vec8 inc = vec8(0,1,2,3,4,5,6,7);
    for(int i = 0; i < height; i ++)
    {
        for(int j = 0; j < width; j+=8)
        {
            int pos = i * width + j;
            vec8 depth_c = _mm256_loadu_ps(&filtered_depth_new[pos]);
            vec8 x = inc + vec8(j);
            vec8 y = vec8(i);
            vec8 refLocalVertexX = (x - vec8(cx)) / vec8(fx) * depth_c;
            vec8 refLocalVertexY = (y - vec8(cy)) / vec8(fy) * depth_c;
            vec8 refVX = vec8(rotation(0,0)) * refLocalVertexX + vec8(rotation(0,1)) * refLocalVertexY + vec8(rotation(0,2)) * depth_c + vec8(translation(0));
            vec8 refVY = vec8(rotation(1,0)) * refLocalVertexX + vec8(rotation(1,1)) * refLocalVertexY + vec8(rotation(1,2)) * depth_c + vec8(translation(1));
            vec8 refVZ = vec8(rotation(2,0)) * refLocalVertexX + vec8(rotation(2,1)) * refLocalVertexY + vec8(rotation(2,2)) * depth_c + vec8(translation(2));

            vec8 ref_coord_x = refVX / refVZ * vec8(fx) + vec8(cx+0.5);
            vec8 ref_coord_y = refVY / refVZ * vec8(fy) + vec8(cy+0.5);
            vec8 valid1 = (ref_coord_x > vec8(1)) & (ref_coord_x < vec8(width - 1));
            vec8 valid2 = (ref_coord_y > vec8(1)) & (ref_coord_y < vec8(height - 1));
            valid1 = valid1 & valid2;

            __m256i cameraPos = _mm256_cvtps_epi32((ref_coord_x.floor() + ref_coord_y.floor() * vec8(width)).xmm);
            vec8 new_depth = _mm256_mask_i32gather_ps(_mm256_set1_ps(0.0),filtered_depth,cameraPos,(valid1.xmm),4);
            valid1 = ((new_depth - refVZ) > (vec8(-threshold) * refVZ)) & ((new_depth - refVZ )< (vec8(threshold) * refVZ));

             __m256 filtered_depth_c = _mm256_blendv_ps(_mm256_set1_ps(0),depth_c.xmm,valid1.xmm);
            _mm256_storeu_ps(&filtered_depth_new[pos],filtered_depth_c);
        }
    }
}
//eliminate outliers
void refineNewframes(Frame &frame_ref, Frame &frame_new,CameraPara &cameraModel)
{

    int width = frame_ref.rgb.cols;
    int height = frame_ref.rgb.rows;

    const float fx = cameraModel.GetFx();
    const float fy = cameraModel.GetFy();
    const float cx = cameraModel.GetCx();
    const float cy = cameraModel.GetCy();

    float * filtered_depth_new = (float *)frame_new.refined_depth.data;
    float * filtered_depth = (float *)frame_ref.refined_depth.data;

    assert(filtered_depth_new != NULL);
    assert(filtered_depth != NULL);

    float depthFar = 10.0f;
    float depthNear = 0.01;

    Eigen::MatrixXf transform_new_to_ref = (frame_ref.pose_sophus[0].inverse()*frame_new.pose_sophus[0]).matrix().block<3, 4>(0, 0).cast<float>();
    Eigen::Matrix3f rotation = transform_new_to_ref.block<3,3>(0,0);
    Eigen::MatrixXf translation = transform_new_to_ref.block<3,1>(0,3);
        // depth refinement

    for(int i = 0; i < height; i ++)
    {
        for(int j = 0; j < width; j++)
        {
            int pos = i * width + j;
            float depth_c = filtered_depth_new[pos];

            if(depth_c >  depthFar || depth_c < depthNear )
            {
                continue;
            }


            Eigen::Vector3f ref_local_vertex = Eigen::Vector3f((j - cx) / fx, (i - cy) / fy, 1);
            Eigen::Vector3f ref_v = rotation * ref_local_vertex * depth_c + translation;
            float ref_coord_x = ref_v(0) / ref_v(2) * fx + cx;
            float ref_coord_y = ref_v(1) / ref_v(2) * fy + cy;
            if(ref_coord_x < 1 || ref_coord_x > width - 1 || ref_coord_y < 1 || ref_coord_y > height - 1)
            {
                continue;
            }
            int ulx = floor(ref_coord_x + 0.5);
            int uly = floor(ref_coord_y + 0.5);

            int ref_pos = uly * width + ulx;
            float depth_ul = (filtered_depth[ref_pos]);

            if(abs(depth_ul - ref_v(2)) >= 0.03 * ref_v(2) )
            {
                filtered_depth_new[pos] = 0;
            }
        }
    }
    return;
}

void refineKeyframesSIMD(Frame &frame_ref, Frame &frame_new,CameraPara &cameraModel)
{

    int width = frame_ref.rgb.cols;
    int height = frame_ref.rgb.rows;

    float fx = cameraModel.GetFx();
    float fy = cameraModel.GetFy();
    float cx = cameraModel.GetCx();
    float cy = cameraModel.GetCy();



    float * filtered_depth_new = (float *)frame_new.refined_depth.data;
    float * filtered_depth = (float *)frame_ref.refined_depth.data;
    float * vertex_weight = (float * ) frame_ref.weight.data;

    float threshold = 0.05;

    assert(filtered_depth_new != NULL);
    assert(filtered_depth != NULL);

    float depthFar = 10.0f;
    float depthNear = 0.01;

    Eigen::MatrixXd transform_ref_to_new;

    transform_ref_to_new = (frame_new.pose_sophus[0].inverse()*frame_ref.pose_sophus[0]).matrix().block<3, 4>(0, 0);



    Eigen::Matrix3f rotation = transform_ref_to_new.cast<float>().block<3,3>(0,0);
    Eigen::MatrixXf translation = transform_ref_to_new.cast<float>().block<3,1>(0,3);
    Eigen::Matrix3f rotationT = rotation.transpose();
        // depth refinement
    int validPixels = 0;
    vec8 inc = vec8(0,1,2,3,4,5,6,7);
    for(int i = 0; i < height; i ++)
    {
        for(int j = 0; j < width; j+=8)
        {
            int pos = i * width + j;
            vec8 depth_c = _mm256_loadu_ps(&filtered_depth[pos]);
            vec8 x = inc + vec8(j);
            vec8 y = vec8(i);
            vec8 refLocalVertexX = (x - vec8(cx)) / vec8(fx) * depth_c;
            vec8 refLocalVertexY = (y - vec8(cy)) / vec8(fy) * depth_c;
            vec8 refVX = vec8(rotation(0,0)) * refLocalVertexX + vec8(rotation(0,1)) * refLocalVertexY + vec8(rotation(0,2)) * depth_c + vec8(translation(0));
            vec8 refVY = vec8(rotation(1,0)) * refLocalVertexX + vec8(rotation(1,1)) * refLocalVertexY + vec8(rotation(1,2)) * depth_c + vec8(translation(1));
            vec8 refVZ = vec8(rotation(2,0)) * refLocalVertexX + vec8(rotation(2,1)) * refLocalVertexY + vec8(rotation(2,2)) * depth_c + vec8(translation(2));

            vec8 ref_coord_x = refVX / refVZ * vec8(fx) + vec8(cx);
            vec8 ref_coord_y = refVY / refVZ * vec8(fy) + vec8(cy);
            vec8 valid1 = (ref_coord_x > vec8(2)) & (ref_coord_x < vec8(width - 2));
            vec8 valid2 = (ref_coord_y > vec8(2)) & (ref_coord_y < vec8(height - 2));
            valid1 = valid1 & valid2;

            __m256i cameraPosUL = _mm256_cvtps_epi32((ref_coord_x.floor() + ref_coord_y.floor() * vec8(width)).xmm);
            __m256i cameraPosUR = _mm256_add_epi32(cameraPosUL, _mm256_set1_epi32(1));
            __m256i cameraPosBL = _mm256_add_epi32(cameraPosUL, _mm256_set1_epi32(width));
            __m256i cameraPosBR = _mm256_add_epi32(cameraPosUL, _mm256_set1_epi32(width + 1));


            vec8 newDepthUL = _mm256_mask_i32gather_ps(_mm256_set1_ps(0.0),filtered_depth_new,cameraPosUL,(valid1.xmm),4);
            vec8 newDepthUR = _mm256_mask_i32gather_ps(_mm256_set1_ps(0.0),filtered_depth_new,cameraPosUR,(valid1.xmm),4);
            vec8 newDepthBL = _mm256_mask_i32gather_ps(_mm256_set1_ps(0.0),filtered_depth_new,cameraPosBL,(valid1.xmm),4);
            vec8 newDepthBR = _mm256_mask_i32gather_ps(_mm256_set1_ps(0.0),filtered_depth_new,cameraPosBR,(valid1.xmm),4);

            vec8 deltaX = ref_coord_x - ref_coord_x.floor();
            vec8 deltaY = ref_coord_y - ref_coord_y.floor();

            vec8 test = (((newDepthUL - newDepthUR) < vec8(0.1)) & ((newDepthUL - newDepthUR) > vec8(-0.1))
                         & ((newDepthUL - newDepthBL) < vec8(0.1)) & ((newDepthUL - newDepthBL) > vec8(-0.1))
                         & ((newDepthUL - newDepthBR) < vec8(0.1)) & ((newDepthUL - newDepthBR) > vec8(-0.1)) );

            __m256i cameraPosNearest = _mm256_cvtps_epi32(((ref_coord_x + vec8(0.5)).floor() + (ref_coord_y+ vec8(0.5)).floor() * vec8(width)).xmm);
            vec8 newDepthNearest = _mm256_mask_i32gather_ps(_mm256_set1_ps(0.0),filtered_depth,cameraPosNearest,(valid1.xmm),4);
            vec8 bilinearDepth = (vec8(1) - vec8(deltaX)) * (vec8(1) - vec8(deltaY)) * newDepthUL +
                    (vec8(1) - vec8(deltaX)) * (vec8(deltaY)) * newDepthUR +
                    (vec8(deltaX)) * (vec8(1) - vec8(deltaY)) * newDepthBL +
                    (vec8(deltaX)) * (vec8(deltaY)) * newDepthBR;
            bilinearDepth.xmm = _mm256_blendv_ps(newDepthNearest.xmm,bilinearDepth.xmm,test.xmm);

            valid1 = (bilinearDepth - refVZ > vec8(-threshold) * refVZ) & (bilinearDepth - refVZ < vec8(threshold) * refVZ);


            vec8 scale = bilinearDepth / refVZ;
            refVX = refVX * scale - vec8(translation(0));
            refVY = refVY * scale - vec8(translation(1));
            refVZ = refVZ * scale - vec8(translation(2));

            vec8 vZ = vec8(rotationT(2,0)) * refVX + vec8(rotationT(2,1)) * refVY + vec8(rotationT(2,2)) * refVZ;

            vec8 weight = _mm256_loadu_ps(&vertex_weight[pos]);
            vec8 filteredDepth = (depth_c * weight + vZ) / (weight + vec8(1));
            vec8 filteredWeight = weight + vec8(1);
            filteredDepth.xmm = _mm256_blendv_ps(depth_c.xmm,filteredDepth.xmm,valid1.xmm);
            filteredWeight.xmm = _mm256_blendv_ps(weight.xmm,filteredWeight.xmm,valid1.xmm);

            _mm256_storeu_ps(&filtered_depth[pos],filteredDepth.xmm);
            _mm256_storeu_ps(&vertex_weight[pos],filteredWeight.xmm);

        }
    }
}

void refineKeyframes(Frame &frame_ref, Frame &frame_new,CameraPara &cameraModel)
{

    int width = frame_ref.rgb.cols;
    int height = frame_ref.rgb.rows;

    float fx = cameraModel.GetFx();
    float fy = cameraModel.GetFy();
    float cx = cameraModel.GetCx();
    float cy = cameraModel.GetCy();



    float * filtered_depth_new = (float *)frame_new.refined_depth.data;
    float * filtered_depth = (float *)frame_ref.refined_depth.data;
    float * vertex_weight = (float * ) frame_ref.weight.data;

    assert(filtered_depth_new != NULL);
    assert(filtered_depth != NULL);

    float depthFar = 10.0f;
    float depthNear = 0.01;




    Eigen::MatrixXd transform_ref_to_new;
    transform_ref_to_new = (frame_new.pose_sophus[0].inverse()*frame_ref.pose_sophus[0]).matrix().block<3, 4>(0, 0);


    Eigen::Matrix3f rotation = transform_ref_to_new.cast<float>().block<3,3>(0,0);
    Eigen::MatrixXf translation = transform_ref_to_new.cast<float>().block<3,1>(0,3);
        // depth refinement
    int validPixels = 0;
    for(int i = 0; i < height; i ++)
    {
        for(int j = 0; j < width; j++)
        {
            int pos = i * width + j;
            float depth_c = filtered_depth[pos];

            if(depth_c >  depthFar || depth_c < depthNear )
            {
                continue;
            }


            Eigen::Vector3f new_local_vertex = Eigen::Vector3f((j - cx) / fx, (i - cy) / fy, 1);
            Eigen::Vector3f new_v = rotation * new_local_vertex * depth_c + translation;
            float new_coord_x = new_v(0) / new_v(2) * fx + cx;
            float new_coord_y = new_v(1) / new_v(2) * fy + cy;
            if(new_coord_x < 1 || new_coord_x > width - 1 || new_coord_y < 1 || new_coord_y > height - 1)
            {
                continue;
            }
            int ulx = floor(new_coord_x);
            int uly = floor(new_coord_y);

            int new_pos = uly * width + ulx;
            float depth_ul = (filtered_depth_new[new_pos]);
            float depth_ur = (filtered_depth_new[new_pos+1]);
            float depth_bl = (filtered_depth_new[new_pos+width]);
            float depth_br = (filtered_depth_new[new_pos+width + 1]);
            float x = new_coord_x - ulx;
            float y = new_coord_y - uly;

            float bilinear_depth;
            if(abs(depth_ul - depth_ur) < 0.1 && abs(depth_ul - depth_bl) < 0.1 && abs(depth_ul - depth_br) < 0.1 )
            {
                bilinear_depth = (1-x)*(1-y) * depth_ul + (1-x)*y * depth_ur + x*(1-y) * depth_bl + x*y * depth_br;
            }
            else
            {
                bilinear_depth = filtered_depth_new[(int)(floor(new_coord_y + 0.5) * width + floor(new_coord_x + 0.5))];
            }
#if 1
            float new_depth_observation;
            Eigen::Vector3f v(0,0,0);

            if(abs(bilinear_depth - new_v(2)) < 0.015 * new_v(2) )
            {
                new_v = new_v * ( bilinear_depth / new_v(2));

                v =  rotation.transpose() * ( new_v - translation );
                float weight = vertex_weight[pos];
                new_depth_observation = (depth_c * weight + v(2)) / (weight + 1) ;
                filtered_depth[pos] = new_depth_observation;
                vertex_weight[pos]++;
                validPixels ++;
            }
#endif

        }
    }
    return;
}

void refineDepthUseNormalSIMD(float *normal, float *depth,
                                 float fx, float fy, float cx, float cy,float width, float height)
{
    int numPixels = height * width;
    vec8 inc = vec8(0,1,2,3,4,5,6,7);
    for(int i = 0; i < height; i++)
    {
        for(int j = 0; j < width; j += 8)
        {

            vec8 y = vec8(i);
            vec8 x = inc + vec8(j);
            int pos = i * width + j;
            vec8 depth_c = _mm256_loadu_ps(&depth[pos]);
            vec8 normalX = _mm256_loadu_ps(&normal[pos]);
            vec8 normalY = _mm256_loadu_ps(&normal[pos + numPixels]);
            vec8 normalZ = _mm256_loadu_ps(&normal[pos + numPixels * 2]);

            vec8 vX = (x - vec8(cx)) / vec8(fx);
            vec8 vY = (y - vec8(cy)) / vec8(fy);
            vec8 vZ = vec8(1);

            vec8 vRSQRT = _mm256_rsqrt_ps((vX * vX + vY * vY + vZ * vZ).xmm);

            vX = vX * vRSQRT;
            vY = vY * vRSQRT;
            vZ = vZ * vRSQRT;

            vec8 vQuality = vX * normalX + vY * normalY + vZ * normalZ;
            vec8 inValid = vQuality > vec8(-0.1) & vQuality < vec8(0.1);


            depth_c = _mm256_blendv_ps(depth_c.xmm,vec8(0.0).xmm,inValid.xmm);
            normalX = _mm256_blendv_ps(normalX.xmm,vec8(0.0).xmm,inValid.xmm);
            normalY = _mm256_blendv_ps(normalY.xmm,vec8(0.0).xmm,inValid.xmm);
            normalZ = _mm256_blendv_ps(normalZ.xmm,vec8(0.0).xmm,inValid.xmm);

            _mm256_storeu_ps(&normal[pos],normalX.xmm);
            _mm256_storeu_ps(&normal[pos + numPixels],normalY.xmm);
            _mm256_storeu_ps(&normal[pos + numPixels * 2],normalZ.xmm);
            _mm256_storeu_ps(&depth[pos],depth_c.xmm);

//            Eigen::Vector3f viewAngle = Eigen::Vector3f((j-cx)/fx,(i-cy)/fy,1);
//            viewAngle.normalize();
//            Eigen::Vector3f normalAngle = Eigen::Vector3f(normal[pos], normal[pos + numPixels],normal[pos + numPixels * 2]);
//            float viewQuality = viewAngle.transpose() * normalAngle;
//            if(fabs(viewQuality) < 0.3)
//            {
//                depth[pos] = 0;
//                normal[pos] = 0;
//                normal[pos + numPixels] = 0;
//                normal[pos + numPixels * 2] = 0;
//            }
        }
    }
}

// to be made into SIMD
void checkColorQuality(const cv::Mat &normalMap, cv::Mat &validColorFlag,
                       float fx, float fy, float cx, float cy)
{

    validColorFlag.release();
    int width = normalMap.cols;
    int height = normalMap.rows;
    validColorFlag.create(height,width,CV_8U);
    float *normal = (float *)normalMap.data;
    unsigned char * flag = (unsigned char *)validColorFlag.data;
    int totalPixelNum = height * width;
    for(int i = 0; i < height; i++)
    {
        for(int j = 0; j < width ; j++)
        {
            int pos = i * width + j;
            Eigen::Vector3f viewAngle = Eigen::Vector3f((j -cx)/fx,(i-cy)/fy,1);
            viewAngle.normalize();
            Eigen::Vector3f normalAngle = Eigen::Vector3f(normal[pos], normal[pos + totalPixelNum],normal[pos + totalPixelNum * 2]);
            float viewQuality = viewAngle.transpose() * normalAngle;
            if(fabs(viewQuality) >= 0.4)
            {
                flag[pos] = 1;
            }
        }
    }
}

void extractNormalMapSIMD(const cv::Mat &depthMap, cv::Mat &normalMap,
                             float fx, float fy, float cx, float cy)
{

    int width = depthMap.cols;
    int height = depthMap.rows;

    normalMap.release();
    normalMap.create(height,width,CV_32FC3);
    float normal_threshold = 0.3;

    float * depth = (float *)depthMap.data;
    float * normal = (float*) normalMap.data;

    int width_dst = width - 1;
    int height_dst = height - 1;

    vec8 inc = vec8(0,1,2,3,4,5,6,7);
    for(unsigned int i = 1; i < height_dst; i++)
    {
        for(unsigned int j = 1; j < width_dst - 9; j+= 8)
        {
            vec8 depth_r = _mm256_loadu_ps(&depth[i * width + j + 1]);
            vec8 depth_b = _mm256_loadu_ps(&depth[i * width + j + width]);
            vec8 depth_l = _mm256_loadu_ps(&depth[i * width + j - 1]);
            vec8 depth_t = _mm256_loadu_ps(&depth[i * width + j - width]);

            vec8 u1 = ((inc + vec8(j) - vec8(cx)) * (depth_r - depth_l) + depth_r + depth_l) / vec8(fx);
            vec8 u2 = vec8(i-cy) * (depth_r - depth_l) / vec8(fy);
            vec8 u3 = depth_r - depth_l;

            vec8 v1 = (inc + vec8(j) - vec8(cx))  * (depth_b - depth_t) / vec8(fx);
            vec8 v2 = (vec8(i-cy) * (depth_b - depth_t) + depth_b + depth_t) / vec8(fy);
            vec8 v3 = depth_b - depth_t;

            vec8 nX = u2*v3 - u3*v2;
            vec8 nY = u3*v1 - u1*v3;
            vec8 nZ = u1*v2 - u2*v1;

            vec8 nSquare = nX*nX + nY*nY + nZ*nZ;
            vec8 valid = (u3 < vec8(normal_threshold)) & (u3 > vec8(-normal_threshold))
                           & (v3 < vec8(normal_threshold)) & (v3 > vec8(-normal_threshold))
                           & (nSquare > vec8(1e-24));
            vec8 nRSQRT = _mm256_rsqrt_ps(nSquare.xmm);
            nX = nX * nRSQRT;
            nY = nY * nRSQRT;
            nZ = nZ * nRSQRT;
            nX = _mm256_blendv_ps(vec8(0.0).xmm,nX.xmm,valid.xmm);
            nY = _mm256_blendv_ps(vec8(0.0).xmm,nY.xmm,valid.xmm);
            nZ = _mm256_blendv_ps(vec8(0.0).xmm,nZ.xmm,valid.xmm);

            _mm256_storeu_ps(&normal[(i*width+j)],nX.xmm);
            _mm256_storeu_ps(&normal[(i*width+j) + width * height],nY.xmm);
            _mm256_storeu_ps(&normal[(i*width+j) + width * height * 2],nZ.xmm);
        }
    }
}


void saveRefinedFrame(std::string fileName, const Frame &frame_new, const  CameraPara &para)
{
    float fx,fy,cx,cy,width,height;
    fx = para.c_fx;
    fy = para.c_fy;
    cx = para.c_cx;
    cy = para.c_cy;
    width = para.width;
    height = para.height;

    std::vector<Eigen::Vector3f, Eigen::aligned_allocator<Eigen::Vector3d> > p;
    std::vector<Eigen::Vector3i, Eigen::aligned_allocator<Eigen::Vector3d> > color;
    for (int j = 0; j < height; j++)
    {

      for (int i = 0; i < width; i++)
      {
        if (frame_new.refined_depth.at<float>(j, i) > 0)
        {
          float x, y, z;
          x = (i - cx) / fx * (frame_new.refined_depth.at<float>(j, i)) ;
          y = (j - cy) / fy * (frame_new.refined_depth.at<float>(j, i)) ;
          z = frame_new.refined_depth.at<float>(j, i);

          Eigen::Vector3d v = applyPose(frame_new.pose_sophus[0],Eigen::Vector3d(x,y,z));
          p.push_back(Eigen::Vector3f(v(0),v(1),v(2)));


          color.push_back(Eigen::Vector3i(frame_new.rgb.at<cv::Vec3b>(j, i)[0], frame_new.rgb.at<cv::Vec3b>(j, i)[1], frame_new.rgb.at<cv::Vec3b>(j, i)[2]));
        }
      }
    }
    savePLYFiles(fileName,p,color);
}






void framePreprocess(Frame &t, MultiViewGeometry::CameraPara &camera)
{

    int height = t.depth.rows;
    int width = t.depth.cols;
    t.refined_depth.create(height,width,CV_32FC1);
    t.weight.create(height,width,CV_32FC1);
    for(int i = 0; i < height * width ; i++)
    {
        if(t.depth.at<unsigned short>(i) > camera.maximum_depth * camera.depth_scale)
        {
            t.depth.at<unsigned short>(i) = 0;
        }
        t.refined_depth.at<float>(i) = float(t.depth.at<unsigned short>(i)) / camera.depth_scale;
        t.weight.at<float>(i) = 0;
    }

    /***************bilateral filter***************/
#if 1
    cv::Mat filteredDepth;
    int bilateralFilterRange = 9;
#if MobileCPU
    bilateralFilterRange = 7;
#endif

    cv::bilateralFilter(t.refined_depth, filteredDepth, bilateralFilterRange, 0.03,4.5);
    t.refined_depth = filteredDepth;
    /***************remove boundary***************/

    float *refined_depth_data = (float *)t.refined_depth.data;
    unsigned short *depth_data = (unsigned short*)t.depth.data;
//    for(int i = 0; i < height * width; i++)
//    {
//        if(fabs(refined_depth_data[i] - float(depth_data[i]) / camera.depth_scale) > 0.02)
//        {
//             refined_depth_data[i] = 0;
//             depth_data[i] = 0;
//        }
//    }
//    removeBoundary(t.refined_depth);
#endif

    for(int i = 0; i < height * width ; i++)
    {
        t.depth.at<unsigned short>(i) = t.refined_depth.at<float>(i) * camera.depth_scale;
    }
    t.depth_scale = camera.depth_scale;

    //    if(t.frame_index % 10 == 0)
    //    {

    //        char fileName[256];
    //        memset(fileName, 0, 256);
    //        sprintf(fileName, "output/ply/%d.ply",t.frame_index);
    //        MultiViewGeometry::saveRefinedFrame(fileName,t,camera);
    //    }

}

int LoadOnlineOPENNIData(Frame &fc,
                   LogReader *liveLogReader,
                   MultiViewGeometry::CameraPara &camera)
{
    static int frame_index = 0;
    assert(liveLogReader != NULL);
    int width = camera.width;
    int height = camera.height;

    liveLogReader->getNext();

    fc.frame_index = frame_index;
    frame_index++;

    fc.rgb.release();
    fc.rgb.create(height,width,CV_8UC3);
    memcpy(fc.rgb.data,liveLogReader->rgb,width*height*3);

    fc.depth.release();
    fc.depth.create(height,width,CV_16UC1);
    memcpy(fc.depth.data,liveLogReader->depth,width*height*sizeof(short));

#if 0
    char fileName[256];
    memset(fileName,0,256);
    sprintf(fileName,"output/img/%04d_rgb.png",fc.frame_index);
    cv::imwrite(fileName,fc.rgb);
    memset(fileName,0,256);
    sprintf(fileName,"output/img/%04d_depth.png",fc.frame_index);
    cv::imwrite(fileName,fc.depth);
#endif

    framePreprocess(fc,camera);

    if(fc.frame_index > 10)
    {
        static_cast<LiveLogReader *>(liveLogReader)->setAuto(0);
    }


}

int LoadOnlineRS2Data(Frame &fc,
                      rs2::pipeline &pipe,
                   MultiViewGeometry::CameraPara &camera)
{


    rs2::pipeline_profile profile = pipe.get_active_profile();
    float depth_scale = get_depth_scale(profile.get_device());

    rs2::frameset frameset = pipe.wait_for_frames();

    auto depth_stream = profile.get_stream(RS2_STREAM_DEPTH);
    auto color_stream = profile.get_stream(RS2_STREAM_COLOR);
    rs2_extrinsics e = depth_stream.get_extrinsics_to(color_stream);
    rs2_intrinsics depthI = depth_stream.as<rs2::video_stream_profile>().get_intrinsics();
    rs2_intrinsics colorI = color_stream.as<rs2::video_stream_profile>().get_intrinsics();
    Eigen::Matrix3f R;
    Eigen::Vector3f t;


    R << e.rotation[0],e.rotation[1],e.rotation[2],
            e.rotation[3],e.rotation[4],e.rotation[5],
            e.rotation[6],e.rotation[7],e.rotation[8];
    t << e.translation[0], e.translation[1], e.translation[2];
    int width = depthI.width;
    int height = depthI.height;


    rs2::video_frame rs2color = frameset.get_color_frame();
    rs2::depth_frame rs2depth = frameset.get_depth_frame();
    const uint16_t* p_depth_frame = reinterpret_cast<const uint16_t*>(rs2depth.get_data());
    uint8_t* p_rgb = reinterpret_cast<uint8_t*>(const_cast<void*>(rs2color.get_data()));


    cv::Mat color_data;
    color_data.create(colorI.height,colorI.width,CV_8UC3);
    memcpy(color_data.data,p_rgb,width*height*3);

    fc.rgb.release();
    fc.rgb.create(height,width,CV_8UC3);
    for(int i = 0; i < height; i++)
    {
      for(int j = 0; j < width; j++)
      {
        fc.rgb.at<cv::Vec3b>(i,j) = cv::Vec3b(0,0,0);
      }
    }
    for(int i = 0; i < height; i++)
    {
      for(int j = 0; j < width; j++)
      {

        float d = (float) (p_depth_frame[i * width + j]) / camera.depth_scale;
        Eigen::Vector3f v = Eigen::Vector3f((j - depthI.ppx) / depthI.fx * d,
                             (i - depthI.ppy) / depthI.fy * d,
                             d);
        Eigen::Vector3f new_v = R * v + t;
        int x = new_v(0) * colorI.fx / new_v(2) + colorI.ppx;
        int y = new_v(1) * colorI.fy / new_v(2) + colorI.ppy;
        if(x < width && x > 0 && y < height && y > 0)
        {
           cv::Vec3b c = color_data.at<cv::Vec3b>(y,x);
           fc.rgb.at<cv::Vec3b>(i,j)[0] = c[0];
           fc.rgb.at<cv::Vec3b>(i,j)[1] = c[1];
           fc.rgb.at<cv::Vec3b>(i,j)[2] = c[2];
        }
      }
    }
    fc.depth.release();
    fc.depth.create(height,width,CV_16UC1);
    memcpy(fc.depth.data,p_depth_frame,width*height*sizeof(short));

//    auto processed = align.process(frameset);


    if (!rs2depth || !rs2color)
    {
        cout << "warning! no camera observations!" << endl;
        return 0;
    }




    static int frame_index = 0;
    fc.frame_index = frame_index;
    frame_index++;
#if 0
    char fileName[256];
    memset(fileName,0,256);
    sprintf(fileName,"output/img/%04d_rgb.png",fc.frame_index);
    cv::imwrite(fileName,fc.rgb);
    memset(fileName,0,256);
    sprintf(fileName,"output/img/%04d_depth.png",fc.frame_index);
    cv::imwrite(fileName,fc.depth);
#endif

    framePreprocess(fc,camera);
    return 1;
}
int LoadRawData(int index,
                Frame &t,
                const vector <string> &rgb_files,
                const vector<string > &depth_files,
                const vector<double> &time_stamp,
                MultiViewGeometry::CameraPara &camera)
{
    if (rgb_files.size() < index + 1 || depth_files.size() < index + 1)
    {
        cout << "no valid files exist !  files " << rgb_files.size() << ", required file index : " << index << endl;
        return 0;
    }
    t.frame_index = index;
    int image_id = index;
    t.time_stamp = time_stamp[image_id];
    t.rgb = cv::imread(rgb_files[image_id].c_str());
    t.depth = cv::imread(depth_files[image_id].c_str(), CV_LOAD_IMAGE_UNCHANGED);



    if (t.rgb.rows <= 0 || t.depth.rows <= 0)
    {
        cout << "load image error ! " << rgb_files[image_id] << " " << depth_files[image_id] << endl;
    }
    assert(t.rgb.rows > 0);
    assert(t.depth.rows > 0);

    framePreprocess(t,camera);

    return 1;
}

void  spilt_word(string ori, vector<string> &res)
{
    string buf; // Have a buffer string
    stringstream ss(ori); // Insert the string into a stream
    while (ss >> buf)
        res.push_back(buf);

}
void initOfflineData(string work_folder, vector <string> &rgb_files, vector<string > &depth_files, vector<double> &time_stamp,
    Eigen::MatrixXd &ground_truth)
{

    cout << "working folder: " << work_folder << endl;
    rgb_files.clear();
    depth_files.clear();
    string rgb_file, depth_file;
    char fileName[256];
    char line[1000];
    memset(fileName, '\0', 256);
    sprintf(fileName, "%s/associate.txt", work_folder.c_str());

    fstream fin;

    float average_time_delay = 0;
    int count = 0;
    fin.open(fileName, ios::in);
    while (fin.getline(line, sizeof(line), '\n'))
    {
        string input = line;
        vector<string> input_data;
        spilt_word(line, input_data);
        if (input_data.size() == 4)
        {
            double tRGB = stod(input_data[0]);
            double tDepth = stod(input_data[0]);
            average_time_delay += tRGB + tDepth;
            count++;
        }
    }
    fin.close();
    average_time_delay /= count;

    fin.open(fileName, ios::in);
    while (fin.getline(line, sizeof(line), '\n'))
    {
        string input = line;
        vector<string> input_data;
        spilt_word(line, input_data);
        if (input_data.size() == 4)
        {
            double tRGB = stod(input_data[0]);
            double tDepth = stod(input_data[2]);

            double time = (tRGB + tDepth) / 2;

            if(input_data[1][0]=='r')
            {
                rgb_file = work_folder + "/" + input_data[1];
                depth_file = work_folder + "/" + input_data[3];
            }
            else if(input_data[1][0]=='d')
            {
                rgb_file = work_folder + "/" + input_data[3];
                depth_file = work_folder + "/" + input_data[1];
            }
            else
            {
                rgb_file = work_folder + "/" + input_data[1];
                depth_file = work_folder + "/" + input_data[3];
            }

            rgb_files.push_back(rgb_file);
            depth_files.push_back(depth_file);
            time_stamp.push_back(time);
        }
    }
    fin.close();

    memset(fileName, '\0', 256);
    sprintf(fileName, "%s/groundtruth.txt", work_folder.c_str());

    fin.open(fileName, ios::in);
    int lineCnt = 0;
    while (fin.getline(line, sizeof(line), '\n'))
    {

        lineCnt++;
    }
    fin.close();
    fin.open(fileName, ios::in);
    ground_truth = Eigen::MatrixXd(lineCnt,8);
    lineCnt = 0;
    while (fin.getline(line, sizeof(line), '\n'))
    {
        string input = line;
        vector<string> input_data;
        spilt_word(line, input_data);
        if (input_data.size() == 8)
        {
            for (int cnt = 0; cnt < 8; cnt++)
            {
                ground_truth(lineCnt, cnt) = stod(input_data[cnt]);
            }
        }
        lineCnt++;
    }
    fin.close();


}



bool DirectoryExists( const char* pzPath )
{
    if ( pzPath == NULL) return false;

    DIR *pDir;
    bool bExists = false;

    pDir = opendir (pzPath);

    if (pDir != NULL)
    {
        bExists = true;
        (void) closedir (pDir);
    }

    return bExists;
}

void makeDir(const std::string & directory)
{
    if ( !boost::filesystem::exists( directory ) )
    {
        const int dir_err = mkdir(directory.c_str(), S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH);
        if (-1 == dir_err)
        {
            printf("Error creating directory!n");
            exit(1);
        }
    }
}

void printHelpFunctions()
{

}

/* sensorType: 0 for offline data
 *             1 for xtion, 2 for realsense
 *
 */
void parseInput(int argc, char **argv,
              int &showCaseMode,
              float &ipnutVoxelResolution,
              std::string &basepath,
              MultiViewGeometry::GlobalParameters &para,
              MultiViewGeometry::CameraPara &camera,
              int &sensorType)
{

    showCaseMode =0;

    sensorType = stod(argv[1]);
    string camera_param= argv[2];
    ipnutVoxelResolution =stod(argv[3]);

    if(sensorType == 0)
    {
        cout << "dataset data" << endl;
        basepath = argv[4];
    }
    else if(sensorType == 1)
    {
        cout << "openni online data scanning!" << endl;          
    }
    else if(sensorType == 2)
    {
        cout << "realsense online data scanning!" << endl;          
    }

    // makeDir(basepath+"/output");
    // makeDir(basepath+"/output/img");

    string global_parameters_file;
    global_parameters_file = "../param/settings.yaml";
    loadGlobalParameters(MultiViewGeometry::g_para,global_parameters_file);


    fstream fin;
    char line[1000];
    fin.open(camera_param.c_str(), ios::in);
    while (fin.getline(line, sizeof(line), '\n'))
    {
        string input = line;
        vector<string> input_data;
        spilt_word(line, input_data);
        if (input_data.size() != 13)
        {
            cout << "error in loading parameters" << endl;
        }
        else
        {
            camera.width = stod(input_data[0]);
            camera.height = stod(input_data[1]);
            camera.c_fx = stod(input_data[2]);
            camera.c_fy = stod(input_data[3]);
            camera.c_cx = stod(input_data[4]);
            camera.c_cy = stod(input_data[5]);
            camera.d[0] = stod(input_data[6]);
            camera.d[1] = stod(input_data[7]);
            camera.d[2] = stod(input_data[8]);
            camera.d[3] = stod(input_data[9]);
            camera.d[4] = stod(input_data[10]);
            camera.depth_scale = stod(input_data[11]);
            camera.maximum_depth = stod(input_data[12]);
        }
    }
    fin.close();


}

void initOpenNICamera(LogReader *logReader, CameraPara &camera)
{
    logReader = new LiveLogReader("", 0, LiveLogReader::CameraType::OpenNI2);
    bool good = ((LiveLogReader *)logReader)->cam->ok();
    if(logReader != NULL)
    {
        // for xtion, sensors need to be calibrated
        camera.c_fx = 540;
        camera.c_fy = 500;
        camera.c_cx = 315;
        camera.c_cy = 233;
        camera.height = 480;
        camera.width = 640;
        camera.depth_scale = 1000;
        camera.maximum_depth = 8;
        memset(camera.d,0,sizeof(float) * 5);
    }
    printf("%d\r\n", logReader != NULL);

}

int initRS2Camera(rs2::pipeline &pipe, CameraPara &camera)
{

    rs2::config cfg;
    cfg.enable_stream(RS2_STREAM_COLOR, 640, 480, RS2_FORMAT_RGB8, 60);
    cfg.enable_stream(RS2_STREAM_DEPTH, 640, 480, RS2_FORMAT_Z16, 60);

    pipe.start(cfg);
    rs2::stream_profile stream = pipe.get_active_profile().get_stream(RS2_STREAM_DEPTH);
    auto video_stream = stream.as<rs2::video_stream_profile>();
    rs2_intrinsics intrinsics = video_stream.get_intrinsics();
    auto principal_point = std::make_pair(intrinsics.ppx, intrinsics.ppy);
    auto focal_length = std::make_pair(intrinsics.fx, intrinsics.fy);
    rs2_distortion model = intrinsics.model;

    camera.c_fx = focal_length.first;
    camera.c_fy = focal_length.second;
    camera.c_cx = principal_point.first;
    camera.c_cy = principal_point.second;
    camera.height = 480;
    camera.width = 640;
    camera.depth_scale = 1000;
    camera.maximum_depth = 8;
    memset(camera.d,0,sizeof(float) * 5);
    return 1;
}
}

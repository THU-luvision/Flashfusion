#ifndef MAPMAINTAIN_HPP
#define MAPMAINTAIN_HPP


#include <opencv/cv.h>
#include <opencv2/opencv.hpp>
#include <cmath>
#include <Eigen/Eigen>
#include <Eigen/StdVector>
#include <immintrin.h>
#include <pmmintrin.h>
#include <vector>


template <class DataType>
inline void extractNormalMap(const cv::Mat &depthMap, cv::Mat &normalMap,
                             float fx, float fy, float cx, float cy, float depthScale)
{

    int width = depthMap.cols;
    int height = depthMap.rows;

    normalMap.release();
    normalMap.create(height,width,CV_32FC3);

    DataType * depth = (DataType *)depthMap.data;
    float * normal = (float*) normalMap.data;

    int width_dst = width - 1;
    int height_dst = height - 1;

    for(unsigned int i = 1; i < height_dst; i++)
    {
        for(unsigned int j = 1; j < width_dst - 1; j++)
        {
            float depth_r = float(depth[i * width + j + 1]) / depthScale;
            float depth_b = float(depth[i * width + j + width]) / depthScale;
            float depth_l = float(depth[i * width + j - 1]) / depthScale;
            float depth_t = float(depth[i * width + j - width]) / depthScale;

            Eigen::Vector3f vrl = Eigen::Vector3f( ((j-cx)*(depth_r -depth_l) + depth_r + depth_l) / fx,
                                    (i-cy) * (depth_r - depth_l) / fy,
                                    depth_r - depth_l
                                    );
            Eigen::Vector3f vtb = Eigen::Vector3f( (j-cx) * (depth_b - depth_t) /fx,
                                                   ((i-cy) * (depth_b - depth_t) + depth_b + depth_t) / fy,
                                                   depth_b - depth_t
                                                 );
            Eigen::Vector3f n = vrl.cross(vtb);
            if(std::isnan(n(0)) || std::isnan(n(1)) || std::isnan(n(2)) || fabs(n.transpose() * n) < 1e-36
                    || fabs(depth_r - depth_l) > 0.3 || fabs(depth_t - depth_b) > 0.3)
            {
                n(0) = 0;
                n(1) = 0;
                n(2) = 0;
            }
            else
            {
                n.normalize();
            }
            normal[(i * width + j) + 0] = n(0);
            normal[(i * width + j) + width * height] = n(1);
            normal[(i * width + j) + width * height * 2] = n(2);
        }
    }
}





#define SIGMA_SPACE 4.5
#define SIGMA_SPACE2_INV_HALF (0.5/(SIGMA_SPACE*SIGMA_SPACE))
#define BILATERAL_FILTER_RADIUS 3 // 6
#define BILATERAL_FILTER_DIAMETER (2*BILATERAL_FILTER_RADIUS + 1)
template <class DataType>
inline void bilateralFilterDepth(const cv::Mat &src, cv::Mat &dst, float sigma_color = 0.03)
{

    int width = src.cols;
    int height = src.rows;
    dst.release();

    dst.create(height,width,src.type());

    const int D = BILATERAL_FILTER_DIAMETER;
    float sigma_color2_inv_half = 0.5 / (sigma_color * sigma_color);

    for(int yidx = 0; yidx < height; yidx++)
    {
        for(int xidx = 0; xidx < width; xidx++)
        {
            int tx = fmin(xidx - D / 2 + D, width - 1);
            int ty = fmin(yidx - D / 2 + D, height - 1);


            float sum1 = 0;
            float sum2 = 0;

            int value = src.at<DataType>(yidx, xidx);
            int sx = fmax(xidx - D / 2, 0);
            int sy = fmax(yidx - D / 2, 0);

            for (int cy = sy; cy < ty; ++cy)
            {
                for (int cx = sx; cx < tx; ++cx)
                {
                    int tmp = src.at<DataType>(cy, cx);

                    float space2 = (xidx - cx) * (xidx - cx) + (yidx - cy) * (yidx - cy);
                    float color2 = (value - tmp) * (value - tmp);

                    float weight = exp(-(space2 * SIGMA_SPACE2_INV_HALF + color2 * sigma_color2_inv_half));

                    std::cout << weight << std::endl;
                    sum1 += tmp * weight;
                    sum2 += weight;

                }
            }
            DataType filtered_value = fmax(0, fmin((sum1 / sum2),40000));

            dst.at<DataType>(yidx,xidx) = filtered_value;
        }
    }
}



#define HALF_WIN_SIZE 1
#define DIST_LIMIT 0.10f
inline void removeBoundary(cv::Mat &depthMat)
{
    cv::Mat tmp = depthMat.clone();
    float * depth_pointer = (float *)depthMat.data;
    float * tmp_pointer = (float *)tmp.data;
    int height = depthMat.rows;
    int width = depthMat.cols;
    for(int yid = 0; yid < height; yid++)
    {
        for(int xid = 0; xid < width; xid++)
        {

            int left = std::max(0, xid - HALF_WIN_SIZE);
            int right = std::min(width - 1, xid + HALF_WIN_SIZE);
            int top = std::max(0, yid - HALF_WIN_SIZE);
            int bottom = std::min(height - 1, yid + HALF_WIN_SIZE);
            int pos = yid * width + xid;
            float depth = tmp_pointer[pos];
            bool on_boundary = (depth < 0.01);

            for (int ridx = top; ridx <= bottom; ++ridx)
            {
                for (int cidx = left; cidx <= right; ++cidx)
                {
                    float depth_around = tmp_pointer[ridx*width + cidx];
                    on_boundary = on_boundary || ( fabsf(depth - depth_around) > DIST_LIMIT);
                }
            }

            bool validFilter = fabs(tmp_pointer[pos] - depth_pointer[pos]) < DIST_LIMIT/2;
            if(!on_boundary && validFilter)
            {
                depth_pointer[pos] = tmp_pointer[pos];
            }
            else
            {
                depth_pointer[pos] = 0;
            }

        }
    }
}


inline void SelectLargestNValues(const int movingAverageLength,
                       std::vector<int> &keyframeIDList,
                       std::vector<float> &keyframeCostList,
                       std::vector<int> &keyframesToUpdate)
{
    std::vector<float> keyframeCostCumsum;
    if(keyframeIDList.size() < movingAverageLength)
    {
        return;
    }
    keyframeCostCumsum = std::vector<float>(keyframeIDList.size());
    keyframeCostCumsum[0] = keyframeCostList[0];
    for(int i = 0; i < keyframeIDList.size()-1; i++)
    {
        keyframeCostCumsum[i+1] = keyframeCostCumsum[i] + keyframeCostList[i+1];
    }
    std::vector<float> movingAverageList(keyframeIDList.size() - movingAverageLength + 1);
    for(int i = 0; i < keyframeIDList.size()-movingAverageLength+1; i++)
    {
        if(i == 0)
        {
            movingAverageList[i] = (keyframeCostCumsum[i + movingAverageLength - 1]) / float(movingAverageLength);
        }
        else
        {
            movingAverageList[i] = (keyframeCostCumsum[i + movingAverageLength - 1] - keyframeCostCumsum[i - 1]) / float(movingAverageLength);
        }
  //      cout << "average: " << i << " " << movingAverageList[i] << endl;
    }
    int max_index = 0;
    float max_value = -1;
    for(int i = 0; i < movingAverageList.size();i++)
    {
        if(movingAverageList[i] > max_value)
        {
            max_index = i;
            max_value = movingAverageList[i];
        }
    }
 //   cout << "max value: " << max_index << " " << max_value << endl;

    float threshold = 0.01 * 0.01;
    if(max_value > threshold)
    {
        std::vector<float> newKeyframeCostList;
        std::vector<int> newKeyframeIDList;
        for(int i = 0; i < keyframeCostList.size();i++)
        {
            if(i < max_index || i > max_index + movingAverageLength - 1)
            {
                newKeyframeCostList.push_back(keyframeCostList[i]);
                newKeyframeIDList.push_back(keyframeIDList[i]);
            }
            else
            {
 //               cout << "chosen max value: " << keyframeCostList[i] << " " << keyframeIDList[i] << endl;
                keyframesToUpdate.push_back(keyframeIDList[i]);
            }
        }
        keyframeCostList = newKeyframeCostList;
        keyframeIDList = newKeyframeIDList;
    }

}
inline float GetPoseDifference(const Eigen::Matrix4f &prePose, const Eigen::Matrix4f &curPose)
{
    Eigen::Matrix4f diffTransformation = prePose.inverse() * curPose;
    Eigen::MatrixXf diff(6,1);
    diff.block<3,1>(0,0) = diffTransformation.block<3,3>(0,0).eulerAngles(2,1,0);
    for(int k = 0; k < 3; k++)
    {
        float a = fabs(diff(k,0) - 3.1415926);
        float b = fabs(diff(k,0) + 3.1415926);
        float c = fabs(diff(k,0));
        diff(k,0) = fmin(a,b);
        diff(k,0) = fmin(diff(k,0),c);
    }
    diff.block<3,1>(3,0) = diffTransformation.block<3,1>(0,3);


    float cost = pow(diff(0,0),2)*9 + pow(diff(1,0),2)*9  + pow(diff(2,0),2)*9 + pow(diff(3,0),2)
            + pow(diff(4,0),2)  + pow(diff(5,0),2);
    return cost;
}


#endif // GEOMETRY3D_HPP

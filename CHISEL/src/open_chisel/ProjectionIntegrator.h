// The MIT License (MIT)
// Copyright (c) 2014 Matthew Klingensmith and Ivan Dryanovski
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

#ifndef PROJECTIONINTEGRATOR_H_
#define PROJECTIONINTEGRATOR_H_

#include <open_chisel/geometry/Geometry.h>
#include <open_chisel/geometry/Frustum.h>
#include <open_chisel/geometry/AABB.h>
#include <open_chisel/camera/PinholeCamera.h>
#include <open_chisel/camera/DepthImage.h>
#include <open_chisel/camera/ColorImage.h>
#include <open_chisel/Chunk.h>

#include <open_chisel/truncation/Truncator.h>
#include <open_chisel/weighting/Weighter.h>

#include <xmmintrin.h>
#include <smmintrin.h>
#include <immintrin.h>

namespace chisel
{

    class ProjectionIntegrator
    {
        public:
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW
            ProjectionIntegrator();
            ProjectionIntegrator(const TruncatorPtr& t,
                                 const WeighterPtr& w,
                                 float carvingDist,
                                 bool enableCarving,
                                 const Vec3List& centroids);

            ~ProjectionIntegrator();


            bool voxelUpdateSIMD(const float *depth_pointer, unsigned char *colorImage,
                                 const PinholeCamera& depthCamera,
                                 const Transform& depthCameraPose, int integrateFlag,
                                 Chunk *chunk, float *weight_pointer) const;
            inline bool IntegrateColor(float * depthImage,
                                       const PinholeCamera& depthCamera,
                                       const Transform& depthCameraPose,
                                       unsigned char * colorImage,
                                       Chunk * chunk,
                                       int integrate_flag,
                                       float *weight = NULL) const
            {
//                    assert(chunk != nullptr);

                    bool updated = false;
#if 1
                    updated = voxelUpdateSIMD(depthImage,
                                              colorImage,
                                              depthCamera,
                                              depthCameraPose,
                                              integrate_flag,
                                              chunk,
                                              weight);
#else
                    float resolution = chunk->GetVoxelResolutionMeters();
                    Eigen::Vector3i chunksNum = chunk->GetNumVoxels();
                    Vec3 origin = chunk->GetOrigin();
                    float resolutionDiagonal = sqrt(3.0f) * resolution;
                    float fx,fy,cx,cy,width,height;
                    fx = depthCamera.GetFx();
                    fy = depthCamera.GetFy();
                    cx = depthCamera.GetCx();
                    cy = depthCamera.GetCy();
                    width = depthCamera.GetWidth();
                    height = depthCamera.GetHeight();
                    Vec3 originInCamera = depthCameraPose.linear().transpose() * (origin - depthCameraPose.translation());
                    float truncation = -1;
                    truncation = truncator->GetTruncationDistance(originInCamera(2));
                    float depth_weight = weighter->GetWeight(1.0f, truncation);
                    float threshold_dist = truncation + resolutionDiagonal;
                    int NumOfVoxels = centroids.size();
                    Color<unsigned char> color;

                    // enable paralle
                    for (size_t i = 0; i < diff_centroids.size(); i++)
                    //parallel_for(indexes.begin(), indexes.end(), [&](const size_t& i)
                    {
                        Vec3 voxelCenterInCamera  = originInCamera + diff_centroids[i];
                        Vec3 cameraPos = Vec3(voxelCenterInCamera(0)/voxelCenterInCamera(2)*fx+cx,
                                              voxelCenterInCamera(1)/voxelCenterInCamera(2)*fy+cy,
                                              voxelCenterInCamera(2));
                        if (cameraPos(0) < 1 || cameraPos(0) > width -1 ||
                            cameraPos(1) < 1 || cameraPos(1) > height - 1 ||
                            voxelCenterInCamera(2) < 0 || voxelCenterInCamera(2) > 3.0)
                        {
                            continue;
                        }
                        int camera_plane_r = static_cast<int>(cameraPos(1)+0.5);
                        int camera_plane_c = static_cast<int>(cameraPos(0)+0.5);

                        float voxelDist = voxelCenterInCamera.z();

                        float depth = depthImage->DepthAt(camera_plane_r, camera_plane_c);
                        if(std::isnan(depth))
                        {
                            continue;
                        }
                        float surfaceDist = depth - voxelDist;
                        if (std::abs(surfaceDist) < truncation + resolutionDiagonal)
                        {
        #if 1
                            ColorVoxel& colorVoxel = chunk->GetColorVoxelMutable(i);
                            colorImage->At(camera_plane_r, camera_plane_c, &color);
                            colorVoxel.SetGreen(color.green);
                            colorVoxel.SetRed(color.red);
                            colorVoxel.SetBlue(color.blue);
                          //  if (colorVoxel.GetWeight() < 5)
                          //  {
                          //      colorImage->At(camera_plane_r, camera_plane_c, &color);
                          //      colorVoxel.Integrate(color.red, color.green, color.blue, 1);
                          //  }
        #endif

                            DistVoxel& voxel = chunk->voxels;
                            voxel.Integrate(surfaceDist, depth_weight,i);
                            updated = true;
                        }
#if 1
                        if (surfaceDist > truncation + carvingDist)
                        {
                            DistVoxel& voxel = chunk->voxels;
                            if (voxel.GetWeight(i) > 0 && voxel.GetSDF(i) < 1e-5)
                            {
                                voxel.Carve(i);
                                updated = true;
                            }
                        }
#endif

                    }
                    //);
#endif
                    return updated;
            }


            template<class DataType, class ColorType> bool DeintegrateColor(const std::shared_ptr<const DepthImage<DataType> >& depthImage, const PinholeCamera& depthCamera, const Transform& depthCameraPose, const std::shared_ptr<const ColorImage<ColorType> >& colorImage, const PinholeCamera& colorCamera, const Transform& colorCameraPose, Chunk* chunk) const
            {
                    assert(chunk != nullptr);
                    bool updated = false;
#if 0
                    float resolution = chunk->GetVoxelResolutionMeters();
                    Vec3 origin = chunk->GetOrigin();
                    float resolutionDiagonal = 2.0 * sqrt(3.0f) * resolution;
                    float fx,fy,cx,cy,width,height;
                    fx = depthCamera.GetFx();
                    fy = depthCamera.GetFy();
                    cx = depthCamera.GetCx();
                    cy = depthCamera.GetCy();
                    width = depthCamera.GetWidth();
                    height = depthCamera.GetHeight();
                    Vec3 originInCamera = depthCameraPose.linear().transpose() * (origin - depthCameraPose.translation());
                    float truncation = -1;
                    truncation = truncator->GetTruncationDistance(originInCamera(2));
                    float depth_weight = weighter->GetWeight(1.0f, truncation);
                    float threshold_dist = truncation + resolutionDiagonal;
                    Color<ColorType> color;

                    // enable paralle
                    for (size_t i = 0; i < centroids.size(); i++)
                    //parallel_for(indexes.begin(), indexes.end(), [&](const size_t& i)
                    {
                        Vec3 voxelCenterInCamera  = originInCamera + diff_centroids[i];
                        Vec3 cameraPos = Vec3(voxelCenterInCamera(0)/voxelCenterInCamera(2)*fx+cx,
                                              voxelCenterInCamera(1)/voxelCenterInCamera(2)*fy+cy,
                                              voxelCenterInCamera(2));
                        if (!depthCamera.IsPointOnImage(cameraPos) || voxelCenterInCamera.z() < 0)
                        {
                            continue;
                        }
                        int camera_plane_r = static_cast<int>(cameraPos(1)+0.5);
                        int camera_plane_c = static_cast<int>(cameraPos(0)+0.5);

                        float voxelDist = voxelCenterInCamera.z();

                        float depth = depthImage->DepthAt(camera_plane_r, camera_plane_c);
                        if(std::isnan(depth))
                        {
                            continue;
                        }
                        float surfaceDist = depth - voxelDist;
                        if (std::abs(surfaceDist) < truncation + resolutionDiagonal)
                        {
#if 1
                            ColorVoxel& colorVoxel = chunk->GetColorVoxelMutable(i);
                            colorImage->At(camera_plane_r, camera_plane_c, &color);
                            colorVoxel.SetGreen(color.green);
                            colorVoxel.SetRed(color.red);
                            colorVoxel.SetBlue(color.blue);
                          //  if (colorVoxel.GetWeight() < 5)
                          //  {
                          //      colorImage->At(camera_plane_r, camera_plane_c, &color);
                          //      colorVoxel.Integrate(color.red, color.green, color.blue, 1);
                          //  }
#endif

                            DistVoxel& voxel = chunk->GetDistVoxelMutable(i);
                            voxel.Deintegrate(surfaceDist, depth_weight);
                            updated = true;
                        }
                        if (surfaceDist > truncation + carvingDist)
                        {
                            DistVoxel& voxel = chunk->GetDistVoxelMutable(i);
                            if (voxel.GetWeight() > 0 && voxel.GetSDF() < 1e-5)
                            {
#if 1
                                voxel.SetWeight(0);
                                voxel.SetSDF(surfaceDist);
#else
                                voxel.Carve();
#endif
                                updated = true;
                            }
                        }

                    }
                    //);
#endif

                    return updated;
            }
            inline const TruncatorPtr& GetTruncator() const { return truncator; }
            inline void SetTruncator(const TruncatorPtr& value) { truncator = value; }
            inline const WeighterPtr& GetWeighter() const { return weighter; }
            inline void SetWeighter(const WeighterPtr& value) { weighter = value; }

            inline float GetCarvingDist() const { return carvingDist; }
            inline bool IsCarvingEnabled() const { return enableVoxelCarving; }
            inline void SetCarvingDist(float dist) { carvingDist = dist; }
            inline void SetCarvingEnabled(bool enabled) { enableVoxelCarving = enabled; }

            inline void SetCentroids(const Vec3List& c) { centroids = c; }


            __m256 * centroids_simd0;
            __m256 * centroids_simd1;
            __m256 * centroids_simd2;
            Vec3List diff_centroids;
            Vec3List centroids;
        protected:
            TruncatorPtr truncator;
            WeighterPtr weighter;
            float carvingDist;
            bool enableVoxelCarving;
    };

} // namespace chisel 

#endif // PROJECTIONINTEGRATOR_H_ 

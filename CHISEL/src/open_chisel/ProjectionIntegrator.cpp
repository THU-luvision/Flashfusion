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

#include <open_chisel/ProjectionIntegrator.h>
#include <open_chisel/geometry/Raycast.h>
#include <iostream>

namespace chisel
{

    ProjectionIntegrator::ProjectionIntegrator() :
        carvingDist(0), enableVoxelCarving(false)
    {
        // TODO Auto-generated constructor stub
        centroids_simd0 = (__m256*)_mm_malloc(64 * sizeof(__m256), 32);
        centroids_simd1 = (__m256*)_mm_malloc(64 * sizeof(__m256), 32);
        centroids_simd2 = (__m256*)_mm_malloc(64 * sizeof(__m256), 32);

    }

    ProjectionIntegrator::ProjectionIntegrator(const TruncatorPtr& t, const WeighterPtr& w, float crvDist, bool enableCrv, const Vec3List& centers) :
            truncator(t), weighter(w), carvingDist(crvDist), enableVoxelCarving(enableCrv), centroids(centers)
    {
        centroids_simd0 = (__m256*)_mm_malloc(64 * sizeof(__m256), 32);
        centroids_simd1 = (__m256*)_mm_malloc(64 * sizeof(__m256), 32);
        centroids_simd2 = (__m256*)_mm_malloc(64 * sizeof(__m256), 32);
    }


    void printUCharFrom256(__m256i data)
    {
        unsigned char * ucharData = (unsigned char *)&data;
        for(int i = 0; i < 32;i++)
        {
            printf("%d ",ucharData[i]);
        }
        printf("\r\n");
    }
    void printShortFrom256(__m256i data)
    {
        short * ucharData = (short *)&data;
        for(int i = 0; i < 16;i++)
        {
            printf("%d ",ucharData[i]);
        }
        printf("\r\n");
    }

    // integrateFlag: 1 for integration, -1 for deintegration
    bool ProjectionIntegrator::voxelUpdateSIMD(const float *depth_pointer,
                                               unsigned char * colorImage,
                                               const PinholeCamera& depthCamera,
                                               const Transform& depthCameraPose,
                                               int integrateFlag,
                                               Chunk * chunk,
                                               float *weight_pointer) const
    {

        bool updated = false;
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

        float depthFar = depthCamera.GetFarPlane();
        float depthNear = depthCamera.GetNearPlane();
        Vec3 originInCamera = depthCameraPose.linear().transpose() * (origin - depthCameraPose.translation());
        float truncation = -1;
        truncation = truncator->GetTruncationDistance(originInCamera(2));
        float depth_weight = weighter->GetWeight(1.0f, truncation);
        float color_weight = 1;
        __m256 weightSignSIMD = _mm256_set1_ps(1);
        if(!integrateFlag)
        {
            weightSignSIMD = _mm256_set1_ps(-1);
            depth_weight *= -1.0f;
            color_weight *= -1;

        }
        float threshold_dist = truncation + resolutionDiagonal;
        float threshold_color = resolutionDiagonal + 0.01;
        int NumOfVoxels = centroids.size();


        float originX,originY,originZ;
        originX = originInCamera(0);
        originY = originInCamera(1);
        originZ = originInCamera(2);
        __m256 origin_simd0 =_mm256_broadcast_ss(&(originX));
        __m256 origin_simd1 =_mm256_broadcast_ss(&(originY));
        __m256 origin_simd2 =_mm256_broadcast_ss(&(originZ));

        __m256 fx_simd =_mm256_set1_ps((fx));
        __m256 fy_simd =_mm256_set1_ps((fy));
        __m256 cx_simd =_mm256_set1_ps((cx + 0.5));
        __m256 cy_simd =_mm256_set1_ps((cy + 0.5));

        __m256i leftThreshold_simd =_mm256_set1_epi32(1);
        __m256i rightThreshold_simd =_mm256_set1_epi32(width - 1);
        __m256i topThreshold_simd =_mm256_set1_epi32(1);
        __m256i bottomThreshold_simd =_mm256_set1_epi32(height - 1);

        __m256i width_simd = _mm256_set_epi32(width,width,width,width,width,width,width,width);


        __m256 defaultDepth_simd = _mm256_set1_ps(0.0);
        __m256 sigma_simd = _mm256_set1_ps(1e-4);

        __m256 minDepth_simd = _mm256_set1_ps(depthNear);
        __m256 maxDepth_simd = _mm256_set1_ps(depthFar);
        __m256 inputWeight_simd = _mm256_set1_ps(depth_weight);

        bool updated_flag = false;

        DistVoxel& voxel = chunk->voxels;
        ColorVoxel &colorVoxel = chunk->colors;
        float * voxelSDFPointer = (float * )(&voxel.sdf[0]);
        float * voxelWeightPointer = (float * )(&voxel.weight[0]);
        unsigned short *voxelColorPointer = (unsigned short *)(&colorVoxel.colorData[0]);

        // use _mm256_mmask_i32gather_ps to load depth information
        // use _mm256_cmp_ps_mask to store the information
        // use floor to save data
        int pos = 0;
        for (int z = 0; z < chunksNum(2); z++)
        {
            for(int y = 0; y < chunksNum(1); y++)
            {
                for(int x = 0; x < chunksNum(0); x+=8)
                {
                    __m256 voxelCenterInCamera_SIMD0 = _mm256_add_ps(origin_simd0, centroids_simd0[pos]);
                    __m256 voxelCenterInCamera_SIMD1 = _mm256_add_ps(origin_simd1, centroids_simd1[pos]);
                    __m256 voxelCenterInCamera_SIMD2 = _mm256_add_ps(origin_simd2, centroids_simd2[pos]);

                    __m256 cameraPos_SIMD0 =
                            _mm256_add_ps(_mm256_mul_ps(_mm256_div_ps(voxelCenterInCamera_SIMD0,voxelCenterInCamera_SIMD2),fx_simd),cx_simd);
                    __m256 cameraPos_SIMD1 =
                            _mm256_add_ps(_mm256_mul_ps(_mm256_div_ps(voxelCenterInCamera_SIMD1,voxelCenterInCamera_SIMD2),fy_simd),cy_simd);
                    __m256i cameraX_SIMD = _mm256_cvtps_epi32(cameraPos_SIMD0);
                    __m256i cameraY_SIMD = _mm256_cvtps_epi32(cameraPos_SIMD1);
                    __m256i valid = _mm256_and_si256(_mm256_cmpgt_epi32(cameraX_SIMD,leftThreshold_simd),_mm256_cmpgt_epi32(rightThreshold_simd,cameraX_SIMD));
                    valid = _mm256_and_si256(valid,_mm256_cmpgt_epi32(cameraY_SIMD,topThreshold_simd));
                    valid = _mm256_and_si256(valid,_mm256_cmpgt_epi32(bottomThreshold_simd,cameraY_SIMD));
                    __m256 validf = _mm256_castsi256_ps(valid);


                    if(_mm256_testz_si256(valid, valid))
                    {
                        continue;
                    }
                    // load depth data and store them
                    __m256i camera_plane_pos_SIMD = _mm256_add_epi32(_mm256_mullo_epi32(cameraY_SIMD,width_simd),cameraX_SIMD);
                    __m256 depth_SIMD = _mm256_mask_i32gather_ps(defaultDepth_simd,depth_pointer,camera_plane_pos_SIMD,validf,4);



                    __m256 weight_SIMD = _mm256_set1_ps(depth_weight);
//                    if(weight_pointer != NULL)
//                    {
//                        weight_SIMD = _mm256_mask_i32gather_ps(defaultDepth_simd,weight_pointer,camera_plane_pos_SIMD,validf,4);
//                    }
                    __m256 surfaceDist_SIMD = _mm256_sub_ps(depth_SIMD,voxelCenterInCamera_SIMD2);
//                    _mm256_store_ps(loadSurfaceDistFromSIMD,surfaceDist_SIMD);
//                    _mm256_store_ps(loadDepthFromSIMD,depth_SIMD);


#if 1

#if 1
                    if(colorImage != NULL)
                    {
                        __m256i updateColor_SIMD =  _mm256_and_si256(valid,
                                                                    _mm256_cvtps_epi32(_mm256_and_ps(_mm256_cmp_ps(surfaceDist_SIMD,_mm256_set1_ps(-threshold_color),_CMP_GT_OS),
                                                                   _mm256_cmp_ps(_mm256_set1_ps(threshold_color),surfaceDist_SIMD,_CMP_GT_OS))));

                        if((!_mm256_testz_si256(updateColor_SIMD,updateColor_SIMD)) )
                        {
                            __m256i inputColorSIMD = _mm256_mask_i32gather_epi32(_mm256_set1_epi16(0),(const int*)colorImage,camera_plane_pos_SIMD,updateColor_SIMD,4);
                            __m256i voxelColorShortSIMD[2];
                            voxelColorShortSIMD[0] = _mm256_load_si256((__m256i *)&voxelColorPointer[pos * 32]);
                            voxelColorShortSIMD[1] = _mm256_load_si256((__m256i *)&voxelColorPointer[pos * 32 + 16]);
                            __m256i inputColorShortSIMD;
                            __m256i updatedColorSIMD;
                            if(integrateFlag)
                            {

                                for(int i = 0; i < 2; i++)
                                {
                                    inputColorShortSIMD = _mm256_cvtepu8_epi16(_mm256_extracti128_si256(inputColorSIMD,i));
                                    updatedColorSIMD = _mm256_add_epi16(voxelColorShortSIMD[i],inputColorShortSIMD);
                                    // boundary check!
                                   __m256i saturationFlag = _mm256_cmpgt_epi16(updatedColorSIMD,_mm256_set1_epi16(120));
                                   saturationFlag = _mm256_shufflehi_epi16(saturationFlag,255);
                                   saturationFlag = _mm256_shufflelo_epi16(saturationFlag,255);
                                   updatedColorSIMD = _mm256_blendv_epi8(updatedColorSIMD,
                                                                         _mm256_srli_epi16(updatedColorSIMD,2),
                                                                         saturationFlag);

                                    _mm256_store_si256((__m256i *)&voxelColorPointer[pos * 32 + 16*i],updatedColorSIMD);


                                }
                            }
                            else
                            {

                                for(int i = 0; i < 2; i++)
                                {
                                    inputColorShortSIMD = _mm256_cvtepu8_epi16(_mm256_extracti128_si256(inputColorSIMD,i));
                                    updatedColorSIMD = _mm256_sub_epi16(voxelColorShortSIMD[i],inputColorShortSIMD);
                                    // boundary check! yet should always larger than 0 by default.
                                    _mm256_store_si256((__m256i *)&voxelColorPointer[pos * 32 + 16*i],updatedColorSIMD);


                                }
                            }
                        }
                    }

#endif

                    validf = _mm256_and_ps(_mm256_cmp_ps(depth_SIMD,minDepth_simd,_CMP_GT_OS),_mm256_cmp_ps(maxDepth_simd,depth_SIMD,_CMP_GT_OS));
                    __m256 surfaceInside_SIMD =  _mm256_and_ps(_mm256_cmp_ps(surfaceDist_SIMD,_mm256_set1_ps(-0.03),_CMP_GT_OS),
                                                               _mm256_cmp_ps(_mm256_set1_ps(truncation + resolutionDiagonal),surfaceDist_SIMD,_CMP_GT_OS));
                    __m256 integrateFlag = _mm256_and_ps(validf,surfaceInside_SIMD);

                    if((!_mm256_testz_ps(integrateFlag,integrateFlag)) )
                    {
                        updated = true;
                        __m256 weightSIMD = _mm256_loadu_ps(&voxelWeightPointer[pos * 8]);
                        __m256 sdfSIMD = _mm256_loadu_ps(&voxelSDFPointer[pos*8]);
                        __m256 newWeightSIMD = _mm256_blendv_ps(defaultDepth_simd,weight_SIMD,integrateFlag);

                        __m256 updatedSdfSIMD = _mm256_div_ps(_mm256_add_ps(_mm256_mul_ps(sdfSIMD,weightSIMD),_mm256_mul_ps(surfaceDist_SIMD,newWeightSIMD)),
                                                _mm256_add_ps(_mm256_add_ps(weightSIMD,newWeightSIMD),sigma_simd));
                        __m256 updatedWeightSIMD = _mm256_add_ps(weightSIMD,newWeightSIMD);
                        __m256 weightValidSIMD =  _mm256_cmp_ps(updatedWeightSIMD,_mm256_set1_ps(0.5),_CMP_GT_OS);
                        updatedSdfSIMD = _mm256_blendv_ps(_mm256_set1_ps(999),updatedSdfSIMD,weightValidSIMD);
                        updatedWeightSIMD = _mm256_blendv_ps(_mm256_set1_ps(0.0f),updatedWeightSIMD,weightValidSIMD);

                        _mm256_storeu_ps(&voxelWeightPointer[pos*8], updatedWeightSIMD);
                        _mm256_storeu_ps(&voxelSDFPointer[pos*8],updatedSdfSIMD);

                    }
#if 0
                    for(int i = 0; i < 8; i++)
                    {
                        if(std::isnan(updatedSdfSIMD[i]) || ((updatedWeightSIMD[i]) < 1.5 && fabs(updatedSdfSIMD[i]) < 1e-3))
                        {
                            printf("weight: %f %f\r\n", weightSIMD[i], newWeightSIMD[i]);
                            printf("sdf: %f %f\r\n", sdfSIMD[i], surfaceDist_SIMD[i]);
                            printf("newWeight: %f %f\r\n", updatedSdfSIMD[i],updatedWeightSIMD[i]);
                            printf("pos: %d %d %d %d\r\n",
                                   chunk->GetID()(0),
                                   chunk->GetID()(1),
                                   chunk->GetID()(2),
                                   pos * 8 + i);

                            while(1)
                            {

                            }
                        }
                    }
#endif

#else
                    for(int i = 0; i < 8; i++)
                    {
                        int voxel_index = pos * 8 + i;
                        float surfaceDist = surfaceDist_SIMD[i];
                        float weight = weight_SIMD[i];
//                        std::cout << "depth_SIMD: " << depth_SIMD[i] << std::endl;




//                        if(chunk->GetID()(0) == 23 && chunk->GetID()(1) == 19 && chunk->GetID()(2) == 53 && voxel_index == 121)
//                        {
//                            printf("%f %f %d %f %f\r\n", surfaceDist,weight,integrateFlag,voxel.GetSDF(voxel_index), voxel.GetWeight(voxel_index));
//                        }

                        if ((surfaceDist < truncation + resolutionDiagonal && surfaceDist > -0.05)
                                && depth_SIMD[i] < depthFar
                                && depth_SIMD[i] > depthNear)
                        {
        #if 1
                            ColorVoxel& colorVoxel = chunk->colors;
                            int *cameraIndex = (int *)&camera_plane_pos_SIMD;
                            int cameraPose = cameraIndex[i];
                            if (cameraPose > 0 && cameraPose < width * height && std::abs(surfaceDist) < 0.005 + resolutionDiagonal)
                            {
                                colorVoxel.Integrate(colorImage[cameraPose * 4 + 2], colorImage[cameraPose * 4 + 1], colorImage[cameraPose * 4], 1, voxel_index);
                            }
        #endif

                            // decrease the weight of unknown part
                            if(surfaceDist < -0.02)
                            {
                                weight *= 0.1;
                            }

                            voxel.Integrate(surfaceDist, weight, voxel_index);

                            if(voxel.GetWeight(voxel_index) < 0.1)
                            {
                                voxel.SetSDF(999,voxel_index);
                                voxel.SetWeight(0,voxel_index);
                            }



                            if(voxel.GetWeight(voxel_index) > 0.01 && voxel.GetSDF(voxel_index) > 10)
                            {
                                voxel.SetSDF(999,voxel_index);
                                voxel.SetWeight(0,voxel_index);
//                                std::cout << "wrong voxel! " << voxel_index << " " << surfaceDist << " " << weight << std::endl;
                            }
                            updated = true;
                        }

//                        if(surfaceDist >= 0.2)
//                        {
//                            voxel.SetWeight(voxel.GetWeight(voxel_index)-weight, voxel_index);
//                            if(voxel.GetWeight(voxel_index) < 0.1)
//                            {
//                                voxel.SetSDF(999,voxel_index);
//                                voxel.SetWeight(0,voxel_index);
//                            }
//                            updated = true;
//                        }
                    }

#endif
                    // integrate voxel or color

                    // calculate surface distance

                    // do voxel integration and curve here.

                    //
                    pos ++;
                }
            }
        }
        return updated;
    }


    ProjectionIntegrator::~ProjectionIntegrator()
    {
        // TODO Auto-generated destructor stub
        _mm_free(centroids_simd0);
        _mm_free(centroids_simd1);
        _mm_free(centroids_simd2);
    }

} // namespace chisel 

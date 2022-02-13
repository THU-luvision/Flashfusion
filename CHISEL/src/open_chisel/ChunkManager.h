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

#ifndef CHUNKMANAGER_H_
#define CHUNKMANAGER_H_

#include <memory>
#include <unordered_map>
#include <mutex>
#include <open_chisel/geometry/Geometry.h>
#include <open_chisel/mesh/Mesh.h>
#include <open_chisel/ColorVoxel.h>
#include <open_chisel/DistVoxel.h>
#include <open_chisel/camera/DepthImage.h>
#include <open_chisel/camera/PinholeCamera.h>
#include <open_chisel/ProjectionIntegrator.h>
#include <iostream>

#include "Chunk.h"
#define TEST_GRADIENT 0
namespace chisel
{
    // Spatial hashing function from Matthias Teschner
    // Optimized Spatial Hashing for Collision Detection of Deformable Objects
    struct ChunkHasher
    {
            // Three large primes are used for spatial hashing.
            static constexpr size_t p1 = 73856093;
            static constexpr size_t p2 = 19349663;
            static constexpr size_t p3 = 83492791;

            std::size_t operator()(const ChunkID& key) const
            {
                return ( key(0) * p1 ^ key(1) * p2 ^ key(2) * p3);
            }
    };



    struct vec8
    {
      __m256 xmm;

      vec8 (__m256 v) : xmm (v) {}

      vec8 (float v) { xmm = _mm256_set1_ps(v); }

      vec8 (float a, float b, float c, float d, float e, float f, float g, float h)
      { xmm = _mm256_set_ps(h,g,f,e,d,c,b,a); }

      vec8 floor()
      {
          return _mm256_floor_ps(xmm);
      }

      vec8 (const float *v) { xmm = _mm256_load_ps(v); }

      vec8 operator & (const vec8 &v) const
      { return vec8(_mm256_and_ps(xmm,v.xmm)); }


      vec8 operator > (const vec8 &v) const
      { return vec8(_mm256_cmp_ps(xmm,v.xmm,_CMP_GT_OS)); }
      vec8 operator < (const vec8 &v) const
      { return vec8(_mm256_cmp_ps(v.xmm,xmm,_CMP_GT_OS)); }

      vec8 operator* (const vec8 &v) const
      { return vec8(_mm256_mul_ps(xmm, v.xmm)); }

      vec8 operator+ (const vec8 &v) const
      { return vec8(_mm256_add_ps(xmm, v.xmm)); }

      vec8 operator- (const vec8 &v) const
      { return vec8(_mm256_sub_ps(xmm, v.xmm)); }

      vec8 operator/ (const vec8 &v) const
      { return vec8(_mm256_div_ps(xmm, v.xmm)); }

      void operator*= (const vec8 &v)
      { xmm = _mm256_mul_ps(xmm, v.xmm); }

      void operator+= (const vec8 &v)
      { xmm = _mm256_add_ps(xmm, v.xmm); }

      void operator-= (const vec8 &v)
      { xmm = _mm256_sub_ps(xmm, v.xmm); }

      void operator/= (const vec8 &v)
      { xmm = _mm256_div_ps(xmm, v.xmm); }

      void operator>> (float *v)
      { _mm256_store_ps(v, xmm); }

    };


    typedef std::unordered_map<ChunkID, ChunkPtr, ChunkHasher> ChunkMap;
    typedef std::unordered_map<ChunkID, bool, ChunkHasher> ChunkSet;
    typedef std::unordered_map<ChunkID, MeshPtr, ChunkHasher> MeshMap;
    typedef std::unordered_map<ChunkID, std::vector<size_t>, ChunkHasher> ChunkPointMap;
    class Frustum;
    class AABB;
    class ProjectionIntegrator;
    class ChunkManager
    {
        public:
            ChunkManager();
            ChunkManager(const Eigen::Vector3i& chunkSize, float voxelResolution, bool color);
            virtual ~ChunkManager();

            inline const ChunkMap& GetChunks() const { return chunks; }
            inline ChunkMap& GetMutableChunks() { return chunks; }

            inline int round(float r)
            {
                return (r > 0.0) ? floor(r + 0.5) : ceil(r - 0.5);
            }


            inline bool HasChunk(const ChunkID& chunk) const
            {
                return chunks.find(chunk) != chunks.end();
            }

            inline ChunkPtr GetChunk(const ChunkID& chunk) const
            {
                return chunks.at(chunk);
            }

            Vec3 GetChunkCenter(const ChunkID& cID)
            {
                return Vec3(chunkSize(0) * cID(0) * voxelResolutionMeters,
                            chunkSize(1) * cID(1) * voxelResolutionMeters,
                            chunkSize(2) * cID(2) * voxelResolutionMeters);

            }

            inline void AddChunk(const ChunkPtr& chunk)
            {
                chunks.insert(std::make_pair(chunk->GetID(), chunk));
            }

            inline bool RemoveChunk(const ChunkID& chunk)
            {
                if(HasChunk(chunk))
                {
                    chunks.erase(chunk);

                    if (HasMesh(chunk))
                    {
                        allMeshes.erase(chunk);
                    }
                    return true;
                }
                return false;
            }

            inline bool RemoveChunk(const ChunkPtr& chunk)
            {
                return RemoveChunk(chunk->GetID());
            }

            inline bool HasChunk(int x, int y, int z) const { return HasChunk(ChunkID(x, y, z)); }
            inline ChunkPtr GetChunk(int x, int y, int z) const { return GetChunk(ChunkID(x, y, z)); }

            inline ChunkPtr GetChunkAt(const Vec3& pos)
            {
                ChunkID id = GetIDAt(pos);

                if (HasChunk(id))
                {
                    return GetChunk(id);
                }

                return ChunkPtr();
            }

            inline ChunkPtr GetOrCreateChunkAt(const Vec3& pos, bool* wasNew)
            {
                ChunkID id = GetIDAt(pos);

                if (HasChunk(id))
                {
                    *wasNew = false;
                    return GetChunk(id);
                }
                else
                {
                    *wasNew = true;
                    CreateChunk(id);
                    return GetChunk(id);
                }
            }

            inline ChunkID GetIDAt(const Vec3& pos) const
            {
                static const float roundingFactorX = 1.0f / (chunkSize(0) * voxelResolutionMeters);
                static const float roundingFactorY = 1.0f / (chunkSize(1) * voxelResolutionMeters);
                static const float roundingFactorZ = 1.0f / (chunkSize(2) * voxelResolutionMeters);
                return ChunkID(static_cast<int>(std::floor(pos(0) * roundingFactorX)),
                               static_cast<int>(std::floor(pos(1) * roundingFactorY)),
                               static_cast<int>(std::floor(pos(2) * roundingFactorZ)));
            }

            inline Vec3 GetCentroid(const Point3& globalVoxelID)
            {
                return globalVoxelID.cast<float>() * voxelResolutionMeters + halfVoxel;
            }
#if 0
            const DistVoxel* GetDistanceVoxel(const Vec3& pos);
            const ColorVoxel* GetColorVoxel(const Vec3& pos);
#endif

            template <class DataType> inline void GetChunkIDsIntersectCamera(ProjectionIntegrator& integrator,
                                                                             const Frustum &frustum, ChunkIDList * chunkList,
                                                                             const std::shared_ptr<const DepthImage<DataType> >& depthImage,
                                                                             const PinholeCamera& camera,
                                                                             const Transform& cameraPose)
            {
                assert(chunkList != nullptr);

                AABB frustumAABB;
                frustum.ComputeBoundingBox(&frustumAABB);

                ChunkID minID = GetIDAt(frustumAABB.min);
                ChunkID maxID = GetIDAt(frustumAABB.max) + Eigen::Vector3i(1, 1, 1);

                float resolutionDiagonal = chunkSize.norm() * voxelResolutionMeters;

                for (int x = minID(0) - 1; x <= maxID(0) + 1; x++)
                {
                    for (int y = minID(1) - 1; y <= maxID(1) + 1; y++)
                    {
                        for (int z = minID(2) - 1; z <= maxID(2) + 1; z++)
                        {
                            Vec3 min = Vec3(x * chunkSize(0), y * chunkSize(1), z * chunkSize(2)) * voxelResolutionMeters;
                            Vec3 max = min + chunkSize.cast<float>() * voxelResolutionMeters;


                            AABB chunkBox(min, max);

                            float corner[3][2];
                            for(int l = 0; l < 3; l++)
                            {
                                corner[l][0] = min(l);
                                corner[l][1] = max(l);
                            }

                            bool intersectFlag = 0;
                            int intersectCornersCnt = 0;
                            Vec3 corners[9];
                            for(int l = 0; l < 2; l++)
                            {
                                for(int m = 0; m < 2; m++)
                                {
                                    for(int n = 0; n < 2; n++)
                                    {
                                        corners[l*4+m*2+n] = Vec3(corner[0][l],corner[1][m],corner[2][n]);

                                    }
                                }
                            }
                            corners[8] = (min + max) / 2;
                            for(int i = 0; i < 9; i++)
                            {

                                Vec3 voxelCenter = corners[i];
                                Vec3 voxelCenterInCamera = cameraPose.linear().transpose() * (voxelCenter - cameraPose.translation());

                                Vec3 cameraPos = camera.ProjectPoint(voxelCenterInCamera);
                                if (!camera.IsPointOnImage(cameraPos) || voxelCenterInCamera.z() < 0)
                                    continue;
                                float voxelDist = voxelCenterInCamera.z();
                                float depth = depthImage->DepthAt((int)cameraPos(1), (int)cameraPos(0)); //depthImage->BilinearInterpolateDepth(cameraPos(0), cameraPos(1));

                                if(std::isnan(depth))
                                {
                                    continue;
                                }

                                float max_trancate_distance = integrator.GetTruncator()->GetTruncationDistance(depth);
                                float surfaceDist = depth - voxelDist;
                                if(voxelDist < 3 && fabs(surfaceDist) < max_trancate_distance + resolutionDiagonal)
                                {
                                    intersectCornersCnt += 1;
                                }
                            }


                                // transform current vertex to camera coordinate

                            if(intersectCornersCnt > 2)
                            {
                                chunkList->push_back(ChunkID(x, y, z));
                            }
                        }
                    }
                }

//                printf("%lu chunks intersect frustum\n", chunkList->size());
            }


            void findCubeCornerByMat(const float *filtered_depth,
                                     const PinholeCamera& depthCamera,
                                     const Transform& depthExtrinsic,
                                     Vec3 &maxCorner,
                                     Vec3 &minCorner)
            {
                int width = depthCamera.GetWidth();
                int height = depthCamera.GetHeight();

                const float fx = depthCamera.GetFx();
                const float fy = depthCamera.GetFy();
                const float cx = depthCamera.GetCx();
                const float cy = depthCamera.GetCy();
                Eigen::Matrix3f rotation = depthExtrinsic.linear();
                Eigen::MatrixXf translation = depthExtrinsic.translation();

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
                        depth_c = depth_c + vec8(0.2);
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
                maxCorner = Eigen::Vector3f(-1e8,-1e8,-1e8);
                minCorner = Eigen::Vector3f(1e8,1e8,1e8);
                for(int i = 0; i < 8; i++ )
                {
                    maxCorner(0) = fmax(maxCorner(0),maxX[i]);
                    minCorner(0) = fmin(minCorner(0),minX[i]);

                    maxCorner(1) = fmax(maxCorner(1),maxY[i]);
                    minCorner(1) = fmin(minCorner(1),minY[i]);

                    maxCorner(2) = fmax(maxCorner(2),maxZ[i]);
                    minCorner(2) = fmin(minCorner(2),minZ[i]);
                }
            }

            void GetBoundaryChunkID(const float *depthImage,
                                    const PinholeCamera& depthCamera,
                                    const Transform& depthExtrinsic,
                                    ChunkID &maxChunkID,
                                    ChunkID &minChunkID)
            {
                float minimum, maximum;
                minimum = 10;
                maximum = -1;
                Vec3 maxValue, minValue;
                findCubeCornerByMat(depthImage,depthCamera,depthExtrinsic,maxValue, minValue);
                maxChunkID = GetIDAt(maxValue);
                minChunkID = GetIDAt(minValue);
            }

            inline void GetChunkIDsObservedByCamera(ProjectionIntegrator& integrator,
                                                    const Frustum &frustum, ChunkIDList * chunkList,
                                                    const float* depthImage,
                                                    const PinholeCamera& camera,
                                                    const Transform& cameraPose)
            {
                assert(chunkList != nullptr);

#if 0
                AABB frustumAABB;
                frustum.ComputeBoundingBox(&frustumAABB);

                ChunkID minID = GetIDAt(frustumAABB.min);
                ChunkID maxID = GetIDAt(frustumAABB.max) + Eigen::Vector3i(1, 1, 1);
#else
                ChunkID minID, maxID;
                GetBoundaryChunkID(depthImage,camera,cameraPose,maxID,minID);
#endif
                float resolutionDiagonal = chunkSize(0) * voxelResolutionMeters / 2;
                int stepSize = 4;
                float negativeTruncation = 0.03;
                if(voxelResolutionMeters > 0.01)
                {
                    resolutionDiagonal = chunkSize(0) * voxelResolutionMeters * sqrt(3) ;
                    stepSize = 1;
                    negativeTruncation = 0.05 * voxelResolutionMeters / 0.005;
                }
                float fx,fy,cx,cy,width,height;
                fx = camera.GetFx();
                fy = camera.GetFy();
                cx = camera.GetCx();
                cy = camera.GetCy();
                width = camera.GetWidth();
                height = camera.GetHeight();
                int cornerCheckCnt = 0;

                float cornerIndex[8] = {0, chunkSize(0), (chunkSize(1) - 1) * chunkSize(0) - 1, chunkSize(1) * chunkSize(0) - 1,
                                        (chunkSize(2) - 1 ) * chunkSize(1) * chunkSize(0),
                                        (chunkSize(2) - 1 ) * chunkSize(1) * chunkSize(0) + chunkSize(0),
                                        (chunkSize(2) - 1 ) * chunkSize(1) * chunkSize(0) + (chunkSize(1) - 1) * chunkSize(0) - 1 ,
                                        (chunkSize(2) - 1 ) * chunkSize(1) * chunkSize(0) + chunkSize(1) * chunkSize(0) - 1};
                // approximately 100, 000 candidate chunks,
                // while 4, 000 chunks are candidate intersects

                Eigen::MatrixXf rotation = cameraPose.linear().transpose();
                Eigen::VectorXf translation = rotation * cameraPose.translation();
                Vec3 r0 = Vec3(rotation(0,0),rotation(1,0),rotation(2,0)) * float(chunkSize(0)) * voxelResolutionMeters;
                Vec3 r1 = Vec3(rotation(0,1),rotation(1,1),rotation(2,1)) * float(chunkSize(1)) * voxelResolutionMeters;
                Vec3 r2 = Vec3(rotation(0,2),rotation(1,2),rotation(2,2)) * float(chunkSize(2)) * voxelResolutionMeters;


                Vec3 halfVoxel = Vec3(GetResolution(), GetResolution(), GetResolution()) * 0.5f;
                Vec3 originX, originY, originZ;

                int candidateChunks = 0;
                int negativeChunks = 0;
                Vec3List diffCentroidCoarse = Vec3List(8);
                Vec3List diffCentroidRefine = Vec3List(8);
                for(int x = 0; x < 2; x++)
                {
                    for(int y = 0; y < 2; y++)
                    {
                        for(int z = 0; z < 2; z++)
                        {
                            Vec3 cur = Vec3(x * chunkSize(0),y * chunkSize(1), z * chunkSize(2));
                            diffCentroidCoarse[x + y*2 + z*4] =
                                    rotation * cur * voxelResolutionMeters * stepSize + halfVoxel;
                            diffCentroidRefine[x + y*2 + z*4] =
                                    rotation * cur * voxelResolutionMeters * 1 + halfVoxel;
                        }
                    }
                }
                __m256 diffCentroidCoarseSIMD[3];
                __m256 diffCentroidRefineSIMD[3];
                for(int i = 0; i < 3; i++)
                {
                    diffCentroidCoarseSIMD[i] = _mm256_set_ps(diffCentroidCoarse[0](i),diffCentroidCoarse[1](i),diffCentroidCoarse[2](i),diffCentroidCoarse[3](i),
                            diffCentroidCoarse[4](i),diffCentroidCoarse[5](i),diffCentroidCoarse[6](i),diffCentroidCoarse[7](i));
                    diffCentroidRefineSIMD[i] = _mm256_set_ps(diffCentroidRefine[0](i),diffCentroidRefine[1](i),diffCentroidRefine[2](i),diffCentroidRefine[3](i),
                            diffCentroidRefine[4](i),diffCentroidRefine[5](i),diffCentroidRefine[6](i),diffCentroidRefine[7](i));
                }

                for (int x = minID(0) - 1; x <= maxID(0) + 1; x += stepSize)
                {
                    originX = r0 * x - translation;
                    for (int y = minID(1) - 1; y <= maxID(1) + 1; y += stepSize)
                    {
                        originY = originX + r1 * y;
                        for (int z = minID(2) - 1; z <= maxID(2) + 1; z += stepSize)
                        {
                            candidateChunks++;
                            bool intersectFlag = 0;
                            Vec3 originInCamera = originY + z * r2;
//                            Vec3 origin = Vec3(x * chunkSize(0), y * chunkSize(1), z * chunkSize(2)) * voxelResolutionMeters;
//                            Vec3 originInCamera = rotation * (origin - translation);


                            float truncation = -1;
                            truncation = integrator.GetTruncator()->GetTruncationDistance(originInCamera(2));
                            float dtp = truncation + resolutionDiagonal * stepSize;
                            float dtn = negativeTruncation + resolutionDiagonal * stepSize;
#if 0
                            intersectFlag = CheckCornerIntersecting(camera,
                                                                    originInCamera,
                                                                    depthImage,
                                                                    dtp,dtn,
                                                                    diffCentroidCoarse);
#else

                            intersectFlag = CheckCornerIntersectingSIMD(camera,
                                                                        originInCamera,
                                                                        depthImage,
                                                                        dtp,
                                                                        dtn,
                                                                        diffCentroidCoarseSIMD[0],
                                                                        diffCentroidCoarseSIMD[1],
                                                                        diffCentroidCoarseSIMD[2]);
#endif

                                // transform current vertex to camera coordinate
//                            if(intersectFlag)
                            {
                                for(int i = x; i < x + stepSize; i++)
                                {
                                    for(int j = y; j < y + stepSize; j++)
                                    {
                                        for(int k = z; k < z + stepSize; k++)
                                        {
                                            bool refinedIntersectFlag = 0;

//                                            if(HasChunk(ChunkID(i,j,k)) && refinedIntersectFlag)
//                                            {
//                                                refinedIntersectFlag = 1;
//                                            }
                                            if(intersectFlag)
                                            {
                                                Vec3 origin = Vec3(i * chunkSize(0), j * chunkSize(1), k * chunkSize(2)) * voxelResolutionMeters;
                                                originInCamera = rotation * origin - translation;
                                                truncation = integrator.GetTruncator()->GetTruncationDistance(originInCamera(2));
                                                float dtp = truncation + resolutionDiagonal;
                                                float dtn = negativeTruncation + resolutionDiagonal;
#if 0
                                                refinedIntersectFlag = CheckCornerIntersecting(camera,
                                                                                               originInCamera,
                                                                                               depthImage,
                                                                                               dtp,dtn,
                                                                                               diffCentroidRefine);
#else
                                                refinedIntersectFlag = CheckCornerIntersectingSIMD(camera,
                                                                                            originInCamera,
                                                                                            depthImage,
                                                                                            dtp,
                                                                                            dtn,
                                                                                            diffCentroidRefineSIMD[0],
                                                                                            diffCentroidRefineSIMD[1],
                                                                                            diffCentroidRefineSIMD[2]);
#endif

                                            }
                                            if(refinedIntersectFlag )
                                            {
                                                chunkList->push_back(ChunkID(i, j, k));
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }

//                printf("select chunk statistics: %d %d\r\n", cornerCheckCnt,chunkList->size());

//                printf("%lu chunks intersect frustum\n", chunkList->size());
            }

            inline bool CheckCornerIntersectingSIMD(const PinholeCamera& camera,
                                                    const Vec3 &originInCamera,
                                                    const float *depthImage,
                                                    float distance_threshold_positive,
                                                    float distance_threshold_negative,
                                                    const __m256 &centroids_simd0,
                                                    const __m256 &centroids_simd1,
                                                    const __m256 &centroids_simd2
                                                    )
            {

                float fx,fy,cx,cy,width,height;
                fx = camera.GetFx();
                fy = camera.GetFy();
                cx = camera.GetCx();
                cy = camera.GetCy();
                width = camera.GetWidth();
                height = camera.GetHeight();
                bool intersectFlag = false;
                __m256 origin_simd0 = _mm256_set1_ps(originInCamera(0));
                __m256 origin_simd1 = _mm256_set1_ps(originInCamera(1));
                __m256 origin_simd2 = _mm256_set1_ps(originInCamera(2));

                __m256 voxelCenterInCamera_SIMD0 = _mm256_add_ps(origin_simd0, centroids_simd0);
                __m256 voxelCenterInCamera_SIMD1 = _mm256_add_ps(origin_simd1, centroids_simd1);
                __m256 voxelCenterInCamera_SIMD2 = _mm256_add_ps(origin_simd2, centroids_simd2);
                __m256 cameraPos_SIMD0 =
                        _mm256_add_ps(_mm256_mul_ps(_mm256_div_ps(voxelCenterInCamera_SIMD0,voxelCenterInCamera_SIMD2),_mm256_set1_ps(fx)),_mm256_set1_ps(cx));
                __m256 cameraPos_SIMD1 =
                        _mm256_add_ps(_mm256_mul_ps(_mm256_div_ps(voxelCenterInCamera_SIMD1,voxelCenterInCamera_SIMD2),_mm256_set1_ps(fy)),_mm256_set1_ps(cy));

                __m256i cameraX_SIMD = _mm256_cvtps_epi32(cameraPos_SIMD0);
                __m256i cameraY_SIMD = _mm256_cvtps_epi32(cameraPos_SIMD1);

                __m256 depthValid_SIMD = _mm256_and_ps(_mm256_cmp_ps(origin_simd2,_mm256_set1_ps(camera.GetNearPlane()),_CMP_GT_OS),
                                                       _mm256_cmp_ps(_mm256_set1_ps(camera.GetFarPlane()),origin_simd2,_CMP_GT_OS));
                __m256i valid = _mm256_and_si256(_mm256_cmpgt_epi32(cameraX_SIMD,_mm256_set1_epi32(1)),_mm256_cmpgt_epi32(_mm256_set1_epi32(width - 1),cameraX_SIMD));
                valid = _mm256_and_si256(valid,_mm256_cmpgt_epi32(cameraY_SIMD,_mm256_set1_epi32(1)));
                valid = _mm256_and_si256(valid,_mm256_cmpgt_epi32(_mm256_set1_epi32(height - 1),cameraY_SIMD));
                __m256 validf = _mm256_castsi256_ps(valid);
                if((_mm256_testz_ps(validf,validf)) )
                {
                    return false;
                }
                __m256i camera_plane_pos_SIMD = _mm256_add_epi32(_mm256_mullo_epi32(cameraY_SIMD,_mm256_set1_epi32(width)),cameraX_SIMD);
                __m256 depth_SIMD = _mm256_mask_i32gather_ps(_mm256_set1_ps(0.0),depthImage,camera_plane_pos_SIMD,validf,4);
                __m256 surfaceDist_SIMD = _mm256_sub_ps(depth_SIMD,voxelCenterInCamera_SIMD2);
                __m256 surfaceInside_SIMD =  _mm256_and_ps(_mm256_cmp_ps(surfaceDist_SIMD,_mm256_set1_ps(-distance_threshold_negative),_CMP_GT_OS),
                                                           _mm256_cmp_ps(_mm256_set1_ps(distance_threshold_positive),surfaceDist_SIMD,_CMP_GT_OS));

                __m256 integrateFlag = _mm256_and_ps(validf,surfaceInside_SIMD);
                integrateFlag = _mm256_and_ps(integrateFlag,depthValid_SIMD);
//                printf("%f %f %f %f %f %f %f %f\r\n",
//                       surfaceDist_SIMD[0],surfaceDist_SIMD[1],surfaceDist_SIMD[2],surfaceDist_SIMD[3],
//                       depth_SIMD[0],depth_SIMD[(1)],depth_SIMD[(2)],depth_SIMD[(3)]);
                if((!_mm256_testz_ps(integrateFlag,integrateFlag)) )
                {
                    intersectFlag = true;
                }
                return intersectFlag;

            }
            inline bool CheckCornerIntersecting(const PinholeCamera& camera,
                                                Vec3 originInCamera,
                                                const float *depthImage,
                                                float distance_threshold_positive,
                                                float distance_threshold_negative,
                                                const Vec3List &diff_centroids
                                                )
            {

                float fx,fy,cx,cy,width,height;
                fx = camera.GetFx();
                fy = camera.GetFy();
                cx = camera.GetCx();
                cy = camera.GetCy();
                width = camera.GetWidth();
                height = camera.GetHeight();
                bool intersectFlag = 0;

                for(int i = 0; i < 8; i++)
                {
                    Vec3 voxelCenterInCamera = originInCamera + diff_centroids[i];
                    Vec3 cameraPos = Vec3(voxelCenterInCamera(0)/voxelCenterInCamera(2)*fx+cx,
                                          voxelCenterInCamera(1)/voxelCenterInCamera(2)*fy+cy,
                                          voxelCenterInCamera(2));

                    if (cameraPos(0) < 1 || cameraPos(0) > width -1 ||
                        cameraPos(1) < 1 || cameraPos(1) > height - 1 ||
                        voxelCenterInCamera(2) < camera.GetNearPlane() || voxelCenterInCamera(2) > camera.GetFarPlane())
                    {
                        continue;
                    }

                    float voxelDist = voxelCenterInCamera.z();
                    int pixelPos = (int)cameraPos(1) * width + (int)cameraPos(0);
                    float depth = depthImage[pixelPos];
                    if(std::isnan(depth))
                    {
                        continue;
                    }
                    float surfaceDist = depth - voxelDist;


                    if(surfaceDist < distance_threshold_positive && surfaceDist > -distance_threshold_negative )
                    {
                        intersectFlag = 1;
                        return intersectFlag;
                    }
                }

                return intersectFlag;

            }

            void GetChunkIDsIntersecting(const AABB& box, ChunkIDList* chunkList);
            void GetChunkIDsIntersecting(const Frustum& frustum, ChunkIDList* chunkList);

            void CreateChunk(const ChunkID& id);

            void GenerateMeshEfficient(const ChunkPtr& chunk, Mesh* mesh);
            void GenerateMesh(const ChunkPtr& chunk, Mesh* mesh);
            void ColorizeMesh(Mesh* mesh);
#if 0
            Vec3 InterpolateColor(const Vec3& colorPos);
#endif


            inline Vec3 InterpolateColorNearest(const Vec3 & colorPos)
            {
                const ChunkID& chunkID = GetIDAt(colorPos);
                if(!HasChunk(chunkID))
                {
                    return Vec3(0, 0, 0);
                }
                else
                {
                    const ChunkPtr& chunk = GetChunk(chunkID);
                    return chunk->GetColorAt(colorPos);
                }
            }
            void CacheCentroids();
            void ExtractBorderVoxelMesh(const ChunkPtr& chunk, const Eigen::Vector3i& index, const Eigen::Vector3f& coordinates, VertIndex* nextMeshIndex, Mesh* mesh);
            void ExtractInsideVoxelMesh(const ChunkPtr& chunk, const Eigen::Vector3i& index, const Vec3& coords, VertIndex* nextMeshIndex, Mesh* mesh);

            inline const MeshMap& GetAllMeshes() const { return allMeshes; }
            inline MeshMap& GetAllMutableMeshes() { return allMeshes; }
            inline const MeshPtr& GetMesh(const ChunkID& chunkID) const { return allMeshes.at(chunkID); }
            inline MeshPtr& GetMutableMesh(const ChunkID& chunkID) { return allMeshes.at(chunkID); }
            inline bool HasMesh(const ChunkID& chunkID) const { return allMeshes.find(chunkID) != allMeshes.end(); }

            inline bool GetUseColor() { return useColor; }

            void RecomputeMeshes(const ChunkSet& chunks, const PinholeCamera &camera);
            void ComputeNormalsFromGradients(Mesh* mesh);

            void GetColorByProject(const PinholeCamera& camera, MeshPtr &mesh);
            inline const Eigen::Vector3i& GetChunkSize() const { return chunkSize; }
            inline float GetResolution() const { return voxelResolutionMeters; }

            inline const Vec3List& GetCentroids() const { return centroids; }

            void PrintMemoryStatistics();

            void Reset();

            bool GetSDFAndGradient(const Eigen::Vector3f& pos, Eigen::Vector3f &grad);

            // cubicVoxelID: the id (from 0~7) of the corner
            // cubicVoxelIndex: the index (from 0~511) of the corner
            bool extractGradientFromCubic(float *cubicSDFPointer,
                                          Point3 &currentVoxelID,
                                          int cubicVoxelID,
                                          int cubicVoxelIndex,
                                          float   *neighborSDFPointer,
                                          ChunkID &neighborChunkID,
                                          Eigen::Vector3f &grad);

            bool GetSDF(const Eigen::Vector3f& pos, double* dist);
            bool GetWeight(const Eigen::Vector3f & pos, double *weight);

            inline bool GetNeighborSDF(bool neighborFlag,
                                const Eigen::Vector3i &cid,
                                int voxelIndex,
                                const ChunkPtr &chunkCentral,
                                float &dd)
            {
#if TEST_GRADIENT
                std::cout << "normal selection: " << neighborFlag << " " << cid.transpose() << " " << voxelIndex << "   ";
#endif
                if(neighborFlag == 0)
                {
                    dd = chunkCentral->voxels.sdf[voxelIndex];
#if TEST_GRADIENT
                    std::cout << "normal selection value: " << dd << std::endl;
#endif
                    if(dd < 1)
                    {
                        return true;
                    }
                    else
                    {
                        return false;
                    }
                    return true;
                }
                else
                {
                    ChunkMap::iterator got = chunks.find(cid);
                    if(got != chunks.end())
                    {
                        chisel::ChunkPtr &chunk = got->second;
                        dd = chunk->voxels.sdf[voxelIndex];
#if TEST_GRADIENT
                        std::cout << "normal selection value: " << dd << std::endl;
#endif
                        if(dd < 1)
                        {
                            return true;
                        }
                        else
                        {
                            return false;
                        }
                        return true;

                    }

                }
                return false;
            }
            inline bool GetNeighborSDF(bool neighborFlag,
                                const Eigen::Vector3i &cid,
                                int voxelIndex,
                                float *currentChunkSDFPointer,
                                float &dd)
            {
#if TEST_GRADIENT
                std::cout << "smart selection: " << neighborFlag << " " << cid.transpose() << " " << voxelIndex << "   ";
#endif
                if(neighborFlag == 0)
                {
                    dd = currentChunkSDFPointer[voxelIndex];
#if TEST_GRADIENT
                    std::cout << "smart selection value: " << dd << std::endl;
#endif
                    if(dd < 1)
                    {
                        return true;
                    }
                    else
                    {
                        return false;
                    }
                }
                else
                {
                    ChunkMap::iterator got = chunks.find(cid);
                    if(got != chunks.end())
                    {
                        chisel::ChunkPtr &chunk = got->second;
                        dd = chunk->voxels.sdf[voxelIndex];
#if TEST_GRADIENT
                        std::cout << "smart selection value: " << dd << std::endl;
#endif
                        if(dd < 1)
                        {
                            return true;
                        }
                        else
                        {
                            return false;
                        }
                    }

                }
                return false;
            }

            inline void deepCopyChunks(ChunkMap &c,
                                       const std::vector<int> &PoseChanged)
            {
                for (const std::pair<ChunkID, ChunkPtr>& chunk:chunks)
                {
//                    if(!PoseChanged[anchorFrameIndex])
//                    {
//                        continue;
//                    }
                    c.insert(std::make_pair(chunk.first,std::allocate_shared<Chunk>(Eigen::aligned_allocator<Chunk>(), chunk.first, chunkSize, voxelResolutionMeters, useColor)));
                    *c.at(chunk.first) = *chunk.second;
#if 0
                    chunk.second->GetDistVoxelMutable(0).SetSDF(-10);
                    printf("if related: %f %f\r\n",c.at(chunk.first)->GetDistVoxelMutable(0).GetSDF(),
                           chunk.second->GetDistVoxelMutable(0).GetSDF(),
                           ori_depth);
                    chunk.second->GetDistVoxelMutable(0).SetSDF(ori_depth);
#endif

                }
            }
#if 0
            void reallocateChunks(const std::vector<Eigen::Matrix3f> &DynamicRotationList,
                                  const std::vector<Eigen::MatrixXf> &DynamicTranslationList,
                                  const std::vector<int> &PoseChanged,
                                  const ChunkMap &ori_chunks)

            {
                static const float roundingFactorX = 1.0f / (chunkSize(0) * voxelResolutionMeters);
                static const float roundingFactorY = 1.0f / (chunkSize(1) * voxelResolutionMeters);
                static const float roundingFactorZ = 1.0f / (chunkSize(2) * voxelResolutionMeters);
                int new_chunks_generated = 0;
                for (const std::pair<ChunkID, ChunkPtr>& chunk:ori_chunks)
                {
                   ChunkPtr cPtr = chunk.second;
                   ChunkID cID = chunk.first;

                   int anchorFrameIndex = cPtr->getAnchorFrameIndex();

                   Eigen::Matrix3f dynamicRotation = DynamicRotationList[anchorFrameIndex];
                   Eigen::MatrixXf dynamicTranslation = DynamicTranslationList[anchorFrameIndex];
                   if(PoseChanged[anchorFrameIndex])
                   {
                       GetChunk(cID)->resetChunk();
                   }
                   else
                   {
                       continue;
                   }

                   Eigen::Vector3f min = cPtr->GetOrigin();
                   Eigen::Vector3f max = min + chunkSize.cast<float>() * voxelResolutionMeters;


                   float corner[3][2];
                   for(int l = 0; l < 3; l++)
                   {
                       corner[l][0] = min(l);
                       corner[l][1] = max(l);
                   }

                   // determin if a new chunk needs to be allocated?
                   // Problems exist:
                   // 1.    Weather we need to allocate new chunks?
                   Vec3 chunkOrigin = cPtr->GetOrigin();
                   int cornerIndex[8] = {0,7,55,63,448,455,503,511};
#if 1
                   for(int k = 0; k < 8; k++)
                   {
                       Vec3 voxelCenter = centroids[cornerIndex[k]] + chunkOrigin;
                       Vec3 newVoxel = dynamicRotation * voxelCenter + dynamicTranslation;

                       if(cPtr->GetDistVoxel(cornerIndex[k]).GetWeight() < 1e-3)
                       {
                           continue;
                       }
                       chisel::ChunkID newChunkID = GetIDAt(newVoxel);
                       if (!HasChunk(newChunkID))
                       {
#if 0
                           printf("%f %f %f %f %f %f %d %d %d %d %d %d\r\n",
                                  cornerVoxel(0),cornerVoxel(1),cornerVoxel(2),
                                  rotatedCorner(0),rotatedCorner(1),cornerVoxel(2),
                                  cID(0),cID(1),cID(2),
                                  newChunkID(0),newChunkID(1),newChunkID(2));
#endif
                          CreateChunk(newChunkID);
                          GetChunk(newChunkID)->setAnchorFrameIndex(anchorFrameIndex);
                          new_chunks_generated++;
                       }
                   }
#else
                   for(int x = 0; x< 2;x++)
                   {
                       for(int y =0; y < 2; y++)
                       {
                           for(int z = 0; z < 2; z++)
                           {
                               Vec3 cornerVoxel = Vec3(corner[0][x],corner[1][y],corner[2][z]);
                               Vec3 rotatedCorner = dynamicRotation * cornerVoxel + dynamicTranslation;
#if 0
                               Vec3 diff = rotatedCorner - cornerVoxel;
                               if(diff.norm() < 0.05)
                               {
                                   continue;
                               }
#endif
                               chisel::ChunkID newChunkID = GetIDAt(rotatedCorner);
//                               chisel::ChunkID newChunkID = ChunkID(
//                                           round(rotatedCorner(0) * roundingFactorX),
//                                           round(rotatedCorner(1) * roundingFactorY),
//                                           round(rotatedCorner(2) * roundingFactorZ));

                               if (!HasChunk(newChunkID))
                               {
#if 0
                                   printf("%f %f %f %f %f %f %d %d %d %d %d %d\r\n",
                                          cornerVoxel(0),cornerVoxel(1),cornerVoxel(2),
                                          rotatedCorner(0),rotatedCorner(1),cornerVoxel(2),
                                          cID(0),cID(1),cID(2),
                                          newChunkID(0),newChunkID(1),newChunkID(2));
#endif
                                  CreateChunk(newChunkID);
                                  GetChunk(newChunkID)->setAnchorFrameIndex(anchorFrameIndex);
                                  new_chunks_generated++;
                               }
                           }
                       }
                   }
#endif
                }
                printf("new chunks generated: %d\r\n",new_chunks_generated);
            }
            template<class DataType, class ColorType>
            void reBufferChunks(const std::vector<Eigen::Matrix3f> &DynamicRotationList,
                                const std::vector<Eigen::MatrixXf> &DynamicTranslationList,
                                const std::vector<int> &PoseChanged,
                                const ChunkMap &ori_chunks)
            {

                Color<ColorType> color;
                int chunks_count = 0;


                const float roundingChunkX = 1.0f / (chunkSize(0) * voxelResolutionMeters);
                const float roundingChunkY = 1.0f / (chunkSize(1) * voxelResolutionMeters);
                const float roundingChunkZ = 1.0f / (chunkSize(2) * voxelResolutionMeters);
                const float roundingVoxelStep = 1.0f / voxelResolutionMeters;

                for (const std::pair<chisel::ChunkID, chisel::ChunkPtr>& chunk:ori_chunks)
                {
                   chisel::ChunkPtr cPtr = chunk.second;
                   chisel::ChunkID cID = chunk.first;
                   int anchorFrameIndex = cPtr->getAnchorFrameIndex();

                   if(!PoseChanged[anchorFrameIndex])
                   {
                       continue;
                   }

                   chunks_count++;


                   Vec3 dynamic;
                   Eigen::Matrix3f dynamicRotation = DynamicRotationList[anchorFrameIndex];
                   Eigen::MatrixXf dynamicTranslation = DynamicTranslationList[anchorFrameIndex];
                   Vec3 chunkOrigin = cPtr->GetOrigin();
                   Vec3 newOrigin = dynamicRotation * chunkOrigin + dynamicTranslation;

                   // localize the voxels that need to be updated, different centroids may have different requirements
                   // have to use bilinear interpolation for higher accuracy
                   // Once a chunk is computed, the others will move in the same way
                   for (size_t i = 0; i < centroids.size(); i++)
                   {
                       Vec3 voxelCenter = centroids[i] + chunkOrigin;
                       Vec3 newVoxel = dynamicRotation * voxelCenter + dynamicTranslation;

                        // to be done, update new voxel based on previous voxel
                        // locate the new chunk and the new voxelID

#if 0

                       chisel::ChunkID newChunkID = ChunkID(
                                   round(newVoxel(0) * roundingFactorX),
                                   round(newVoxel(1) * roundingFactorY),
                                   round(newVoxel(2) * roundingFactorZ));
                       ChunkPtr newChunk;
                       if(HasChunk(newChunkID))
                       {
                            newChunk = GetChunk(newChunkID);
                       }
                       else
                       {
                           continue;
                       }
#else
                       ChunkPtr newChunk = GetChunkAt(newVoxel);

#endif
//                       ChunkPtr newChunk = GetChunk(ChunkID(static_cast<int>(newVoxel(0) * roundingFactorX),
//                                                            static_cast<int>(newVoxel(1) * roundingFactorY),
//                                                            static_cast<int>(newVoxel(2) * roundingFactorZ)));
                       DistVoxel& oriDistVoxel = cPtr->GetDistVoxelMutable(i);
                       ColorVoxel& oriColorVoxel = cPtr->GetColorVoxelMutable(i);

                       if(oriDistVoxel.GetWeight() < 1e-3)
                       {
                           continue;
                       }
                       // trilinear interpolation is required.
                       if(newChunk.get())
                       {

                           // try trilinear interpolation

                           Vec3 rel = (newVoxel - newChunk->GetOrigin());
                           int voxelID = newChunk->GetVoxelID(rel);


                           // due to the upsampling issues
                           if(voxelID > 511)
                           {
                                printf("%d\r\n",voxelID);
                                voxelID = 511;
                           }
                           if(voxelID < 0)
                           {
                                printf("%d\r\n",voxelID);
                                voxelID = 0;
                           }



                           // check performance first, can be further acclerated!!!
                           Vec3 nominal_voxel = newChunk->GetOrigin() + centroids[voxelID];
                           // reproject back to calculate depth information
                           Vec3 reproject_voxel = dynamicRotation.transpose() * (nominal_voxel - dynamicTranslation);

                           // rasterization
                           Vec3 rasterizedPose = reproject_voxel * roundingVoxelStep;
                           float distX = rasterizedPose(0) - floor(rasterizedPose(0));
                           float distY = rasterizedPose(1) - floor(rasterizedPose(1));
                           float distZ = rasterizedPose(2) - floor(rasterizedPose(2));
                           Eigen::Vector3i V[8];
                           float spatialWeight[8];
                           V[0] = Eigen::Vector3i(floor(rasterizedPose(0)),floor(rasterizedPose(1)),floor(rasterizedPose(2)));
                           V[1] = Eigen::Vector3i(floor(rasterizedPose(0)),floor(rasterizedPose(1)),ceil(rasterizedPose(2)));
                           V[2] = Eigen::Vector3i(floor(rasterizedPose(0)),ceil(rasterizedPose(1)),floor(rasterizedPose(2)));
                           V[3] = Eigen::Vector3i(floor(rasterizedPose(0)),ceil(rasterizedPose(1)),ceil(rasterizedPose(2)));
                           V[4] = Eigen::Vector3i(ceil(rasterizedPose(0)),floor(rasterizedPose(1)),floor(rasterizedPose(2)));
                           V[5] = Eigen::Vector3i(ceil(rasterizedPose(0)),floor(rasterizedPose(1)),ceil(rasterizedPose(2)));
                           V[6] = Eigen::Vector3i(ceil(rasterizedPose(0)),ceil(rasterizedPose(1)),floor(rasterizedPose(2)));
                           V[7] = Eigen::Vector3i(ceil(rasterizedPose(0)),ceil(rasterizedPose(1)),ceil(rasterizedPose(2)));

                           spatialWeight[0] = (1 - distX) * (1 - distY) * (1 - distZ);
                           spatialWeight[1] = (1 - distX) * (1 - distY) * (distZ);
                           spatialWeight[2] = (1 - distX) * (distY) * (1 - distZ);
                           spatialWeight[3] = (1 - distX) * (distY) * (distZ);
                           spatialWeight[4] = (distX) * (1 - distY) * (1 - distZ);
                           spatialWeight[5] = (distX) * (1 - distY) * (distZ);
                           spatialWeight[6] = (distX) * (distY) * (1 - distZ);
                           spatialWeight[7] = (distX) * (distY) * (distZ);
                           float trilinear_depth = 0;
                           float trilinear_weight = 0;

                           for(int k = 0; k < 8; k++)
                           {
                               ChunkID cid = Eigen::Vector3i(
                                           floor((float)V[k](0) / (float)chunkSize(0)),
                                           floor((float)V[k](1) / (float)chunkSize(1)),
                                           floor((float)V[k](2) / (float)chunkSize(2)));

                               if(ori_chunks.find(cid) == ori_chunks.end())
                               {
                                    continue;
                               }

                               unsigned int vid = V[k](0) - cid(0) * chunkSize(0) +
                                       (V[k](1) - cid(1) * chunkSize(1)) * chunkSize(0) +
                                       (V[k](2) - cid(2) * chunkSize(2)) * chunkSize(0) * chunkSize(1);


                               ChunkPtr reprojectChunk = ori_chunks.at(cid);
                               if(!reprojectChunk.get())
                               {
                                   continue;
                               }
                               if(vid >= 512)
                               {
                                   printf("%d %d %d     %d %d %d    %d %d %d    %d %d %d\r\n",
                                          V[k](0) - cid(0) * chunkSize(0),
                                          (V[k](1) - cid(1) * chunkSize(1)),
                                          (V[k](2) - cid(2) * chunkSize(2)),
                                          cid(0),
                                          cid(1),
                                          cid(2),
                                          V[k](0),
                                          V[k](1),
                                          V[k](2),
                                          static_cast<int>((std::floor( (float)V[k](0) / (float)chunkSize(0)))),
                                          static_cast<int>((std::floor( (float)V[k](1) / (float)chunkSize(1)))),
                                          static_cast<int>((std::floor( (float)V[k](2) / (float)chunkSize(2))))
                                                                         );

                               }
                               assert(vid < 512);

                                       if(reprojectChunk->HasVoxels())
                               {

                                       if(reprojectChunk->GetDistVoxel(vid).GetWeight() > 1e-3)
                                       {
                                           trilinear_depth += reprojectChunk->GetDistVoxel(vid).GetSDF() * spatialWeight[k];
                                           trilinear_weight += spatialWeight[k];
                                       }
                               }
                           }


                           if(trilinear_weight  < 1e-3)
                           {

                               for(int k = 0; k < 8; k++)
                               {
                                   ChunkID cid = Eigen::Vector3i(
                                               floor((float)V[k](0) / (float)chunkSize(0)),
                                               floor((float)V[k](1) / (float)chunkSize(1)),
                                               floor((float)V[k](2) / (float)chunkSize(2)));

                                   if(ori_chunks.find(cid) == ori_chunks.end())
                                   {
                                        continue;
                                   }

                                   unsigned int vid = V[k](0) - cid(0) * chunkSize(0) +
                                           (V[k](1) - cid(1) * chunkSize(1)) * chunkSize(0) +
                                           (V[k](2) - cid(2) * chunkSize(2)) * chunkSize(0) * chunkSize(1);


                                   ChunkPtr reprojectChunk = ori_chunks.at(cid);

                                   if( k == 0)
                                   {

                                       printf("%d ori VS reproject: %f %f %f %d %d %d %d     %f %f %f %d %d %d %d    %f %f %f\r\n",
                                              k,
                                              voxelCenter(0),
                                              voxelCenter(1),
                                              voxelCenter(2),
                                              cID(0),
                                              cID(1),
                                              cID(2),
                                              i,
                                              reproject_voxel(0),
                                              reproject_voxel(1),
                                              reproject_voxel(2),
                                              cid(0),
                                              cid(1),
                                              cid(2),
                                              vid,
                                              reprojectChunk->GetDistVoxel(vid).GetWeight(),
                                              oriDistVoxel.GetWeight(),
                                              oriDistVoxel.GetSDF()
                                              );
                                   }




                                           if(reprojectChunk->HasVoxels())
                                   {

                                           if(reprojectChunk->GetDistVoxel(vid).GetWeight() > 1e-3)
                                           {
                                               trilinear_depth += reprojectChunk->GetDistVoxel(vid).GetSDF() * spatialWeight[k];
                                               trilinear_weight += spatialWeight[k];
                                           }
                                   }
                               }
                           }



                           DistVoxel& newDistVoxel = newChunk->GetDistVoxelMutable(voxelID);
                           ColorVoxel& newColorVoxel = newChunk->GetColorVoxelMutable(voxelID);

                           if(trilinear_weight > 0)
                           {
                               newColorVoxel.SetRed(oriColorVoxel.GetRed());
                               newColorVoxel.SetGreen(oriColorVoxel.GetGreen());
                               newColorVoxel.SetBlue(oriColorVoxel.GetBlue());
                               newColorVoxel.SetWeight(5);
                               newDistVoxel.Integrate(trilinear_depth/trilinear_weight,1);

//                               newDistVoxel.Integrate(oriDistVoxel.GetSDF(),oriDistVoxel.GetWeight());

                           }
                           else
                           {
                               printf("%d wrong interplation!\r\n", i);
                           }

//                           newDistVoxel.Integrate(oriDistVoxel.GetSDF(),oriDistVoxel.GetWeight());
#if 0
                           if(i < 2)
                           {
                               printf("new voxel: %f %f %f %f %f %f\r\n",
                                      voxelCenter[0],
                                      voxelCenter[1],
                                      voxelCenter[2],
                                      newVoxel[0],
                                      newVoxel[1],
                                      newVoxel[2]);
                               printf("voxel ID: %d %d %f %f %f %f\n",voxelID, i,oriDistVoxel.GetSDF(),oriDistVoxel.GetWeight(), newDistVoxel.GetSDF(), newDistVoxel.GetWeight());
                           }
#endif
                       }
                   }
                }
                printf("avilable chunks: %d\n",chunks_count);



                for (const std::pair<chisel::ChunkID, chisel::ChunkPtr>& chunk:ori_chunks)
                {
                   chisel::ChunkPtr cPtr = chunk.second;
                   chisel::ChunkID cID = chunk.first;
                   int anchorFrameIndex = cPtr->getAnchorFrameIndex();

                   if(!PoseChanged[anchorFrameIndex])
                   {
                       continue;
                   }

                   int cornerIndex[8] = {0,7,55,63,448,455,503,511};

                   bool redudentChunk = 1;
                   for(int k = 0; k < 8; k++)
                   {
                        if(cPtr->GetDistVoxel(cornerIndex[k]).GetWeight() > 1e-5)
                        {
                            redudentChunk = 0;
                            break;
                        }
                   }
                   if(redudentChunk)
                   {
                       RemoveChunk(cID);
                   }
                   else
                   {
                       std::mutex mutex;
                       RecomputeMesh(cID,mutex);
                   }




                }
            }

#endif

            EIGEN_MAKE_ALIGNED_OPERATOR_NEW


        protected:
            ChunkMap chunks;
            std::vector<int> voxelNeighborIndex;
            Eigen::Vector3i chunkSize;
            float voxelResolutionMeters;
            Vec3 halfVoxel;
            Vec3List centroids;
            Eigen::Matrix<int, 3, 8> cubeIndexOffsets;
            std::vector<int,Eigen::aligned_allocator<Eigen::Vector4f> > neighborVoxelIndexChunkWise;
            std::vector<int,Eigen::aligned_allocator<Eigen::Vector4f> > neighborChunkIndexChunkWise;
            std::vector<Vec3, Eigen::aligned_allocator<Eigen::Vector4f> > interpolatedPoints;
            int edgeIndexPairs[12][2];
            MeshMap allMeshes;
            bool useColor;

    };

    typedef std::shared_ptr<ChunkManager> ChunkManagerPtr;
    typedef std::shared_ptr<const ChunkManager> ChunkManagerConstPtr;


} // namespace chisel 

#endif // CHUNKMANAGER_H_ 

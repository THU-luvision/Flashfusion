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

#ifndef CHISEL_H_
#define CHISEL_H_

#include <open_chisel/threading/Threading.h>
#include <open_chisel/ChunkManager.h>
#include <open_chisel/ProjectionIntegrator.h>
#include <open_chisel/geometry/Geometry.h>
#include <open_chisel/camera/PinholeCamera.h>
#include <open_chisel/camera/DepthImage.h>
#include <open_chisel/geometry/Frustum.h>
#include "Stopwatch.h"
#include <chrono>
#include <thread>

namespace chisel
{

    class Chisel
    {
        public:
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW
            Chisel();
            Chisel(const Eigen::Vector3i& chunkSize, float voxelResolution, bool useColor);
            virtual ~Chisel();

            inline const ChunkManager& GetChunkManager() const { return chunkManager; }
            inline ChunkManager& GetMutableChunkManager() { return chunkManager; }
            inline void SetChunkManager(const ChunkManager& manager) { chunkManager = manager; }


            int GetFullMeshes(float *vertices);


            void bufferIntegratorSIMDCentroids(ProjectionIntegrator& integrator, const Transform &depthExtrinsic);


            void GetSearchRegion(float *corners,
                                 const PinholeCamera& depthCamera,
                                 const Transform& depthExtrinsic)
            {


                float minX = minChunkID(0) * float(chunkManager.GetChunkSize()(0)) * chunkManager.GetResolution();
                float minY = minChunkID(1) * float(chunkManager.GetChunkSize()(1)) * chunkManager.GetResolution();
                float minZ = minChunkID(2) * float(chunkManager.GetChunkSize()(2)) * chunkManager.GetResolution();

                float maxX = maxChunkID(0) * float(chunkManager.GetChunkSize()(0)) * chunkManager.GetResolution();
                float maxY = maxChunkID(1) * float(chunkManager.GetChunkSize()(1)) * chunkManager.GetResolution();
                float maxZ = maxChunkID(2) * float(chunkManager.GetChunkSize()(2)) * chunkManager.GetResolution();



                std::cout << "min: " << minX << " " << minY << " " << minZ << " max: ";
                std::cout << maxX << " " << maxY << " " << maxZ << std::endl;

                float vertexList[24] =
                {
                    minX, minY, minZ,
                    maxX, minY, minZ,
                    minX, maxY, minZ,
                    maxX, maxY, minZ,
                    minX, minY, maxZ,
                    maxX, minY, maxZ,
                    minX, maxY, maxZ,
                    maxX, maxY, maxZ
                };
                memcpy(corners, vertexList, 24 * sizeof(float));
            }

            ChunkID maxChunkID;
            ChunkID minChunkID;
            std::vector<float> candidateCubes;


            float EvaluateReferenceFrame(const ChunkID &cID, const PinholeCamera& camera,
                                         const Eigen::Matrix4f &pos)
            {

                float threshold = 0.2;
                if(!chunkManager.HasMesh(cID))
                {
                    return 0.2;
                }
                const MeshPtr &mesh = chunkManager.GetMesh(cID);
                float fx = camera.GetFx();
                float fy = camera.GetFy();
                float avgLoc = 0;
                Vec3 chunkCenter = chunkManager.GetChunkCenter(cID);

                Vec3 vertex = chunkCenter;
                Vec4 localVertex = pos * Vec4(vertex(0),vertex(1),vertex(2),1);
                localVertex = localVertex / localVertex(2);
//                float x = localVertex(0) / fx;
//                float y = localVertex(1) / fy;
//                avgLoc = fmax(0.2,fmax(x,y));

                Vec3 viewAngle = Vec3(localVertex(0),localVertex(1),localVertex(2));
                viewAngle.normalize();
                float viewQuality = mesh->averageNoraml.dot(viewAngle);

                return viewQuality;
            }
            void UpdateReferenceFrame(ChunkIDList &chunksIntersecting,
                                   int keyframeIndex,
                                   const Mat4x4 pos,
                                   const float * normalPointer,
                                   const unsigned char * colorPointer,
                                   const float * depthPointer,
                                      const PinholeCamera &camera
                                   )
            {
                for ( const ChunkID& chunkID : chunksIntersecting)
                {
                    if (chunkManager.HasChunk(chunkID))
                    {

                        ChunkPtr chunk = chunkManager.GetChunk(chunkID);
                        if(chunk->GetReferenceFrameIndex() < 0 || EvaluateReferenceFrame(chunkID, camera,pos) <= EvaluateReferenceFrame(chunkID,camera,chunk->GetReferenceFramePose()))
                        {
                            chunk->UpdateReferenceFrame(keyframeIndex, pos, normalPointer, colorPointer,depthPointer);
                        }
                    }
                }
            }

            void PrepareIntersectChunks(ProjectionIntegrator& integrator,
                                                              float *depthImage,
                                                              const Transform& depthExtrinsic,
                                                              const PinholeCamera& depthCamera,
                                                              ChunkIDList &chunksIntersecting,
                                                              std::vector<bool> &needsUpdateFlag,
                                                              std::vector<bool> &newChunkFlag)
            {
                TICK("CHISEL::Reintegration::1::prepareIntersectChunks::1::prepareChunks");
                bufferIntegratorSIMDCentroids(integrator, depthExtrinsic);
                chunksIntersecting.clear();
                needsUpdateFlag.clear();
                newChunkFlag.clear();

                chunkManager.GetBoundaryChunkID(depthImage,depthCamera,depthExtrinsic,maxChunkID,minChunkID);

                Frustum frustum;
                depthCamera.SetupFrustum(depthExtrinsic, &frustum);
                TOCK("CHISEL::Reintegration::1::prepareIntersectChunks::1::prepareChunks");

                TICK("CHISEL::Reintegration::1::prepareIntersectChunks::2::collectChunks");
                chunkManager.GetChunkIDsObservedByCamera(integrator,frustum,&chunksIntersecting, depthImage,depthCamera,depthExtrinsic);
                TOCK("CHISEL::Reintegration::1::prepareIntersectChunks::2::collectChunks");
                TICK("CHISEL::Reintegration::1::prepareIntersectChunks::3::createChunks");
                GetChunkCubes(candidateCubes,chunksIntersecting);
                for ( const ChunkID& chunkID : chunksIntersecting)
                {
                    bool chunkNew = false;
                    if (!chunkManager.HasChunk(chunkID))
                    {
                       chunkNew = true;
                       chunkManager.CreateChunk(chunkID);
                    }
                    newChunkFlag.push_back(chunkNew);
                    needsUpdateFlag.push_back(false);
                }
                TOCK("CHISEL::Reintegration::1::prepareIntersectChunks::3::createChunks");


            }

            void PrepareIntersectChunks(ProjectionIntegrator& integrator,
                                                              float *depthImage,
                                                              const Transform& depthExtrinsic,
                                                              const PinholeCamera& depthCamera,
                                                              std::vector<void *> &candidateChunks,
                                                              std::vector<bool> &needsUpdateFlag,
                                                              std::vector<bool> &newChunkFlag)
            {

                TICK("CHISEL::Reintegration::1::prepareIntersectChunks::1::prepareChunks");
                bufferIntegratorSIMDCentroids(integrator, depthExtrinsic);
                ChunkIDList chunksIntersecting;
                chunksIntersecting.clear();
                needsUpdateFlag.clear();
                newChunkFlag.clear();

                chunkManager.GetBoundaryChunkID(depthImage,depthCamera,depthExtrinsic,maxChunkID,minChunkID);

                Frustum frustum;
                depthCamera.SetupFrustum(depthExtrinsic, &frustum);
                TOCK("CHISEL::Reintegration::1::prepareIntersectChunks::1::prepareChunks");

                TICK("CHISEL::Reintegration::1::prepareIntersectChunks::2::collectChunks");
                chunkManager.GetChunkIDsObservedByCamera(integrator,frustum,&chunksIntersecting, depthImage,depthCamera,depthExtrinsic);
                TOCK("CHISEL::Reintegration::1::prepareIntersectChunks::2::collectChunks");
                TICK("CHISEL::Reintegration::1::prepareIntersectChunks::3::createChunks");
                GetChunkCubes(candidateCubes,chunksIntersecting);

                for ( const ChunkID& chunkID : chunksIntersecting)
                {
                    bool chunkNew = false;
                    if (!chunkManager.HasChunk(chunkID))
                    {
                       chunkNew = true;
                       chunkManager.CreateChunk(chunkID);
                    }
                    candidateChunks.push_back((void *)chunkManager.GetChunk(chunkID).get());
                    newChunkFlag.push_back(chunkNew);
                    needsUpdateFlag.push_back(false);
                }
                TOCK("CHISEL::Reintegration::1::prepareIntersectChunks::3::createChunks");

            }

            void FinalizeIntegrateChunks(ChunkIDList &chunksIntersecting,
                                         std::vector<bool> &needsUpdateFlag,
                                         std::vector<bool> &newChunkFlag,
                                         ChunkIDList &validChunks)
            {

                validChunks.clear();

                ChunkIDList garbageChunks;

                TICK("CHISEL::Reintegration::4::FinalizeIntegrateChunks::1::findMeshesToUpdate");
                for(int i = 0; i < chunksIntersecting.size();i++)
                {
                    const ChunkID& chunkID = chunksIntersecting[i];
                    bool chunkNew = newChunkFlag[i];
                    bool needsUpdate = needsUpdateFlag[i];
                    if (needsUpdate)
                    {
                        meshesToUpdate[chunkID] = true;
                        meshesToUpdate[chunkID + ChunkID(-1, 0, 0)] = true;
                        meshesToUpdate[chunkID + ChunkID( 1, 0, 0)] = true;
                        meshesToUpdate[chunkID + ChunkID( 0, -1, 0)] = true;
                        meshesToUpdate[chunkID + ChunkID(0, 1, 0)] = true;
                        meshesToUpdate[chunkID + ChunkID(0, 0, -1)] = true;
                        meshesToUpdate[chunkID + ChunkID(0, 0, 1)] = true;
//                        for (int dx = -1; dx <= 1; dx++)
//                        {
//                            for (int dy = -1; dy <= 1; dy++)
//                            {
//                                for (int dz = -1; dz <= 1; dz++)
//                                {
//                                    meshesToUpdate[chunkID + ChunkID(dx, dy, dz)] = true;
//                                }
//                            }
//                        }
                        validChunks.push_back(chunkID);
                    }
                    else if(chunkNew)
                    {
                        garbageChunks.push_back(chunkID);
                    }
                }
                TOCK("CHISEL::Reintegration::4::FinalizeIntegrateChunks::1::findMeshesToUpdate");


//              printf("chunk list size: %d, garbage chunks: %d\r\n",chunkNewList.size(),garbageChunks.size());
                TICK("CHISEL::Reintegration::4::FinalizeIntegrateChunks::2::GarbageCollect");
                GarbageCollect(garbageChunks);
                TICK("CHISEL::Reintegration::4::FinalizeIntegrateChunks::2::GarbageCollect");

            }

            void IntegrateDepthScanColor(ProjectionIntegrator& integrator,
                                         float * depthImage,
                                         unsigned char* colorImage,
                                         const Transform& depthExtrinsic,
                                         const PinholeCamera& depthCamera,
                                         ChunkIDList &chunksIntersecting,
                                         std::vector<bool> &needsUpdateFlag,
                                         int integrate_flag,
                                         float *weight = NULL)
            {

                bufferIntegratorSIMDCentroids(integrator, depthExtrinsic);


                if(chunksIntersecting.size() < 1)
                {
                    return;
                }
                std::vector<int> threadIndex;
                for(int i = 0; i < chunksIntersecting.size();i++)
                {
                    threadIndex.push_back(i);
                }

                parallel_for(threadIndex.begin(),threadIndex.end(),[&](const int& i)
                {
                    ChunkID chunkID = chunksIntersecting[i];
                    if(chunkManager.GetUseColor())
                    {
                        const ChunkPtr &chunk = chunkManager.GetChunk(chunkID);
                        bool needsUpdate = integrator.IntegrateColor(depthImage, depthCamera, depthExtrinsic, colorImage, chunk.get(),integrate_flag,weight);
                        needsUpdateFlag[i] = (needsUpdateFlag[i] || needsUpdate);
                    }
                });
            }

            float GetDistanceFromSurface(Eigen::Vector3f global_vertex, float &tsdfWeight)
            {

                tsdfWeight = 0;

                global_vertex -= Eigen::Vector3f(1,1,1) * (chunkManager.GetResolution() / 2);
                const float roundingVoxelStep = 1.0f / chunkManager.GetResolution();
                Eigen::Vector3i chunkSize = chunkManager.GetChunkSize();
                Vec3 rasterizedPose = global_vertex*roundingVoxelStep;


                Eigen::Vector3i V[8];
                float spatialWeight[8];
                float distX = rasterizedPose(0) - floor(rasterizedPose(0));
                float distY = rasterizedPose(1) - floor(rasterizedPose(1));
                float distZ = rasterizedPose(2) - floor(rasterizedPose(2));
                spatialWeight[0] = (1 - distX) * (1 - distY) * (1 - distZ);
                spatialWeight[1] = (1 - distX) * (1 - distY) * (distZ);
                spatialWeight[2] = (1 - distX) * (distY) * (1 - distZ);
                spatialWeight[3] = (1 - distX) * (distY) * (distZ);
                spatialWeight[4] = (distX) * (1 - distY) * (1 - distZ);
                spatialWeight[5] = (distX) * (1 - distY) * (distZ);
                spatialWeight[6] = (distX) * (distY) * (1 - distZ);
                spatialWeight[7] = (distX) * (distY) * (distZ);
                V[0] = Eigen::Vector3i(floor(rasterizedPose(0)),floor(rasterizedPose(1)),floor(rasterizedPose(2)));
                V[1] = Eigen::Vector3i(floor(rasterizedPose(0)),floor(rasterizedPose(1)),ceil(rasterizedPose(2)));
                V[2] = Eigen::Vector3i(floor(rasterizedPose(0)),ceil(rasterizedPose(1)),floor(rasterizedPose(2)));
                V[3] = Eigen::Vector3i(floor(rasterizedPose(0)),ceil(rasterizedPose(1)),ceil(rasterizedPose(2)));
                V[4] = Eigen::Vector3i(ceil(rasterizedPose(0)),floor(rasterizedPose(1)),floor(rasterizedPose(2)));
                V[5] = Eigen::Vector3i(ceil(rasterizedPose(0)),floor(rasterizedPose(1)),ceil(rasterizedPose(2)));
                V[6] = Eigen::Vector3i(ceil(rasterizedPose(0)),ceil(rasterizedPose(1)),floor(rasterizedPose(2)));
                V[7] = Eigen::Vector3i(ceil(rasterizedPose(0)),ceil(rasterizedPose(1)),ceil(rasterizedPose(2)));

                float weight = 0;
                float distance = 0;
                for(int k = 0; k < 8; k++)
                {
                    ChunkID cid = Eigen::Vector3i(floor((float)V[k](0) / (float)chunkSize(0)),
                                                  floor((float)V[k](1) / (float)chunkSize(1)),
                                                  floor((float)V[k](2) / (float)chunkSize(2)));

                    if(chunkManager.HasChunk(cid))
                    {

                        ChunkPtr chunk = chunkManager.GetChunk(cid);
                        unsigned int vid = V[k](0) - cid(0) * chunkSize(0) +
                                (V[k](1) - cid(1) * chunkSize(1)) * chunkSize(0) +
                                (V[k](2) - cid(2) * chunkSize(2)) * chunkSize(0) * chunkSize(1);

                        const DistVoxel &voxels = chunk->GetVoxels();
                        weight += spatialWeight[k] * voxels.weight[vid];
                        distance += voxels.sdf[vid] * spatialWeight[k] * voxels.weight[vid];
                        tsdfWeight += voxels.weight[vid] * spatialWeight[k];
                    }
                }
                if(weight > 0)
                {
                    distance = distance / weight;
                    tsdfWeight = tsdfWeight / weight;
                }
#if 0
                if(distance > 0.1)
                {
                    for(int k = 0; k < 8; k++)
                    {
                        ChunkID cid = Eigen::Vector3i(floor((float)V[k](0) / (float)chunkSize(0)),
                                                      floor((float)V[k](1) / (float)chunkSize(1)),
                                                      floor((float)V[k](2) / (float)chunkSize(2)));

                        if(chunkManager.HasChunk(cid))
                        {

                            ChunkPtr chunk = chunkManager.GetChunk(cid);
                            unsigned int vid = V[k](0) - cid(0) * chunkSize(0) +
                                    (V[k](1) - cid(1) * chunkSize(1)) * chunkSize(0) +
                                    (V[k](2) - cid(2) * chunkSize(2)) * chunkSize(0) * chunkSize(1);

                            const DistVoxel &voxels = chunk->GetVoxels();

                            std::cout << "weight/distance/k: " << spatialWeight[k] << "/"
                                      << voxels.sdf[vid] << "/" << k << std::endl;
                        }
                    }
                }
#endif
                return distance;

            }


            void GetChunkCubes(std::vector<float> &cubes,
                               ChunkIDList &chunksIntersecting)
            {
                cubes.clear();
                Vec3 offset[8];
                Vec3 resolution = chunkManager.GetResolution() * chunkManager.GetChunkSize().cast<float>();
                offset[0] = Vec3(0,0,0);
                offset[1] = Vec3(1,0,0);
                offset[2] = Vec3(0,1,0);
                offset[3] = Vec3(1,1,0);
                offset[4] = Vec3(0,0,1);
                offset[5] = Vec3(1,0,1);
                offset[6] = Vec3(0,1,1);
                offset[7] = Vec3(1,1,1);

                Eigen::Vector3i numVoxels = chunkManager.GetChunkSize();
                float voxelResolutionMeters = chunkManager.GetResolution();
                for(int i = 0; i < chunksIntersecting.size(); i++)
                {
                    ChunkID & chunkID = chunksIntersecting[i];
                    Vec3 origin = Vec3(numVoxels(0) * chunkID(0) * voxelResolutionMeters,
                                       numVoxels(1) * chunkID(1) * voxelResolutionMeters,
                                       numVoxels(2) * chunkID(2) * voxelResolutionMeters);
                    for(int j = 0; j < 8;j++)
                    {
                        Vec3 corner = origin + Vec3(offset[j](0)*resolution(0),offset[j](1)*resolution(1),offset[j](2)*resolution(2));
                        cubes.push_back(corner(0));
                        cubes.push_back(corner(1));
                        cubes.push_back(corner(2));
                    }
                }

            }

            void RefineFrameInVoxel(ProjectionIntegrator& integrator,
                                    float * depthImage,
                                    float * weight,
                                    const Transform& depthExtrinsic,
                                    const PinholeCamera& depthCamera)
            {

                int height = depthCamera.GetHeight();
                int width = depthCamera.GetWidth();
                float cx = depthCamera.GetCx();
                float cy = depthCamera.GetCy();
                float fx = depthCamera.GetFx();
                float fy = depthCamera.GetFy();

                Eigen::Matrix3f rotationRef = depthExtrinsic.linear();
                Eigen::Vector3f translationRef = depthExtrinsic.translation();

                float tsdfWeight;
                for(int i = 0; i < height; i ++)
                {
                    for(int j = 0; j < width; j++)
                    {
                        float depth = depthImage[i*width+j];
                        if(depth < 0.05 || depth > 3)
                        {
                            continue;
                        }

                        Eigen::Vector3f dir = Eigen::Vector3f((j-cx)/fx,(i-cy)/fy,1);
                        Eigen::Vector3f global_vertex = rotationRef * dir * depth + translationRef;
                        float updated_distance = GetDistanceFromSurface(global_vertex,tsdfWeight);
                        float first_update = updated_distance;
                        float depth_init = depth;
                        depth += updated_distance;
                        global_vertex = rotationRef * dir * depth + translationRef;
                        updated_distance = GetDistanceFromSurface(global_vertex,tsdfWeight);

                        depth += updated_distance;
                        global_vertex = rotationRef * dir * depth + translationRef;
                        updated_distance = GetDistanceFromSurface(global_vertex,tsdfWeight);

                        depth += updated_distance;
                        global_vertex = rotationRef * dir * depth + translationRef;
                        updated_distance = GetDistanceFromSurface(global_vertex,tsdfWeight);

                        depth += updated_distance;
                        global_vertex = rotationRef * dir * depth + translationRef;
                        updated_distance = GetDistanceFromSurface(global_vertex,tsdfWeight);

                        depth += updated_distance;
                        global_vertex = rotationRef * dir * depth + translationRef;
                        updated_distance = GetDistanceFromSurface(global_vertex,tsdfWeight);

                        depth += updated_distance;
                        depthImage[i*width + j] = depth;
                        weight[i*width +j] = tsdfWeight;
                        if(fabs(updated_distance) > 5e-3 )
                        {
                            depthImage[i*width + j] = 0;
                            weight[i*width +j] = 0;
                        }

                        if(depthImage[i*width + j] > depthCamera.GetFarPlane() || depthImage[i*width+j] < depthCamera.GetNearPlane() )
                        {
                            depthImage[i*width + j] = 0;
                            weight[i*width +j] = 0;
                        }
                        if(fabs(depthImage[i * width + j] - depth_init) > 0.1 )
                        {
                            depthImage[i*width + j] = 0;
                            weight[i*width +j] = 0;
                        }
#if 0
                        if(fabs(updated_distance) > 1e-3)
                        {
                            std::cout << "warning! " << depth << " " << depth_init << " " << updated_distance << " " << first_update << std::endl;
                        }
#endif
                        // search along dir until iso surface is found

                    }
                }

            }


            void IntegrateDepthScanColor(ProjectionIntegrator& integrator,
                                         float * depthImage,
                                         unsigned char* colorImage,
                                         const Transform& depthExtrinsic,
                                         const PinholeCamera& depthCamera)
            {

                ChunkIDList chunksIntersecting;
                std::vector<bool> needsUpdateFlag;
                std::vector<bool> newChunkFlag;
                ChunkIDList validChunks;
                PrepareIntersectChunks(integrator,depthImage,depthExtrinsic,depthCamera,chunksIntersecting,needsUpdateFlag,newChunkFlag);
                IntegrateDepthScanColor(integrator,depthImage,colorImage,depthExtrinsic,depthCamera,chunksIntersecting,needsUpdateFlag,1);
                FinalizeIntegrateChunks(chunksIntersecting,needsUpdateFlag,newChunkFlag,validChunks);
            }


            bool SaveTSDFFiles(const std::string& fileName);


            void GarbageCollect(const ChunkIDList& chunks)
            {
                       for (const ChunkID& chunkID : chunks)
                       {
                           chunkManager.RemoveChunk(chunkID);
                           meshesToUpdate.erase(chunkID);
                       }
            }

            void UpdateMeshes(const PinholeCamera& camera)
            {
                chunkManager.RecomputeMeshes(meshesToUpdate,camera);
                printf("begin to clear meshes\r\n");
                meshesToUpdate.clear();
                printf("finish to clear meshes\r\n");
            }

            bool SaveAllMeshesToPLY(const std::string& filename);
            void Reset();

            const ChunkSet& GetMeshesToUpdate() const { return meshesToUpdate; }


            ChunkManager chunkManager;
            ChunkSet meshesToUpdate;
        protected:

    };
    typedef std::shared_ptr<Chisel> ChiselPtr;
    typedef std::shared_ptr<const Chisel> ChiselConstPtr;

} // namespace chisel 

#endif // CHISEL_H_ 

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

#include <assert.h>
#include <open_chisel/threading/Threading.h>
#include <open_chisel/ChunkManager.h>
#include <open_chisel/geometry/Frustum.h>
#include <open_chisel/geometry/AABB.h>
#include <open_chisel/marching_cubes/MarchingCubes.h>
#include <open_chisel/geometry/Raycast.h>
#include <open_chisel/ProjectionIntegrator.h>
#include <open_chisel/truncation/Truncator.h>
#include <iostream>
#include "Stopwatch.h"
using namespace std;

namespace chisel
{

    ChunkManager::ChunkManager() :
            chunkSize(16, 16, 16), voxelResolutionMeters(0.03)
    {
        CacheCentroids();
    }

    ChunkManager::~ChunkManager()
    {

    }

    ChunkManager::ChunkManager(const Eigen::Vector3i& size, float res, bool color) :
            chunkSize(size), voxelResolutionMeters(res), useColor(color)
    {
        CacheCentroids();
    }

    void ChunkManager::CacheCentroids()
    {
        halfVoxel = Vec3(voxelResolutionMeters, voxelResolutionMeters, voxelResolutionMeters) * 0.5f;
        centroids.resize(static_cast<size_t>(chunkSize(0) * chunkSize(1) * chunkSize(2)));
        int i = 0;
        for (int z = 0; z < chunkSize(2); z++)
        {
            for(int y = 0; y < chunkSize(1); y++)
            {
                for(int x = 0; x < chunkSize(0); x++)
                {
                    centroids[i] = Vec3(x, y, z) * voxelResolutionMeters + halfVoxel;
                    i++;
                }
            }
        }

        cubeIndexOffsets << 0, 1, 1, 0, 0, 1, 1, 0,
                            0, 0, 1, 1, 0, 0, 1, 1,
                            0, 0, 0, 0, 1, 1, 1, 1;

        int tempEdgeIndexPairs[12][2] =
        {
            { 0, 1 },
            { 1, 2 },
            { 2, 3 },
            { 3, 0 },
            { 4, 5 },
            { 5, 6 },
            { 6, 7 },
            { 7, 4 },
            { 0, 4 },
            { 1, 5 },
            { 2, 6 },
            { 3, 7 }
        };
        for(int i = 0; i < 12; i++)
        {
            for(int j = 0; j < 2; j++)
            {
                edgeIndexPairs[i][j] = tempEdgeIndexPairs[i][j];
            }
        }

        neighborVoxelIndexChunkWise.assign(chunkSize(2) * chunkSize(1) * chunkSize(0) * 8, 0);
        neighborChunkIndexChunkWise.assign(chunkSize(2) * chunkSize(1) * chunkSize(0) * 8, 0);
        for (int z = 0; z < chunkSize(2); z++)
        {
            for(int y = 0; y < chunkSize(1); y++)
            {
                for(int x = 0; x < chunkSize(0); x++)
                {
                    for(int k = 0; k < 8; k++)
                    {

                        int cornerX = x + cubeIndexOffsets(0,k);
                        int cornerY = y + cubeIndexOffsets(1,k);
                        int cornerZ = z + cubeIndexOffsets(2,k);

                        int cornerVoxelIndex = (int)(cornerX & 0x07) +
                                               ((int)(cornerY & 0x07)) * chunkSize(0) +
                                               ((int)(cornerZ & 0x07)) * chunkSize(1) * chunkSize(0);

                        int index = (cornerX == chunkSize(0)) + (cornerY == chunkSize(1)) * 2 + (cornerZ == chunkSize(2)) * 4;
                        neighborVoxelIndexChunkWise[(z*chunkSize(1) * chunkSize(0)+y*chunkSize(0)+x) * 8 + k] = cornerVoxelIndex;
                        neighborChunkIndexChunkWise[(z*chunkSize(1) * chunkSize(0)+y*chunkSize(0)+x) * 8 + k] = index;
                    }
                }
            }
        }
        voxelNeighborIndex = std::vector<int>(6 * chunkSize(2) * chunkSize(1) * chunkSize(0));

        int voxelNum = chunkSize(2) * chunkSize(1) * chunkSize(0);
        i = 0;
        for (int z = 0; z < chunkSize(2); z++)
        {
            for(int y = 0; y < chunkSize(1); y++)
            {
                for(int x = 0; x < chunkSize(0); x++)
                {
                    Eigen::Vector3i vid;
                    int voxelIndex;

                    vid = Eigen::Vector3i(x - 1,y,z);
                    voxelNeighborIndex[i + voxelNum * 0] =  ((((vid(0) + chunkSize(0) )% chunkSize(0)))  +
                                                             ((vid(1) + chunkSize(1)) % chunkSize(1))  * chunkSize(0) +
                                                             ((vid(2) + chunkSize(2))% chunkSize(2))  * chunkSize(0) * chunkSize(1));
                    vid = Eigen::Vector3i(x + 1,y,z);
                    voxelNeighborIndex[i + voxelNum * 1] =  ((((vid(0) + chunkSize(0) )% chunkSize(0)))  +
                                                             ((vid(1) + chunkSize(1)) % chunkSize(1))  * chunkSize(0) +
                                                             ((vid(2) + chunkSize(2))% chunkSize(2))  * chunkSize(0) * chunkSize(1));
                    vid = Eigen::Vector3i(x,y - 1,z);
                    voxelNeighborIndex[i + voxelNum * 2] =  ((((vid(0) + chunkSize(0) )% chunkSize(0)))  +
                                                             ((vid(1) + chunkSize(1)) % chunkSize(1))  * chunkSize(0) +
                                                             ((vid(2) + chunkSize(2))% chunkSize(2))  * chunkSize(0) * chunkSize(1));
                    vid = Eigen::Vector3i(x,y + 1,z);
                    voxelNeighborIndex[i + voxelNum * 3] =  ((((vid(0) + chunkSize(0) )% chunkSize(0)))  +
                                                             ((vid(1) + chunkSize(1)) % chunkSize(1))  * chunkSize(0) +
                                                             ((vid(2) + chunkSize(2))% chunkSize(2))  * chunkSize(0) * chunkSize(1));
                    vid = Eigen::Vector3i(x,y,z - 1);
                    voxelNeighborIndex[i + voxelNum * 4] =  ((((vid(0) + chunkSize(0) )% chunkSize(0)))  +
                                                             ((vid(1) + chunkSize(1)) % chunkSize(1))  * chunkSize(0) +
                                                             ((vid(2) + chunkSize(2))% chunkSize(2))  * chunkSize(0) * chunkSize(1));
                    vid = Eigen::Vector3i(x,y,z+1);
                    voxelNeighborIndex[i + voxelNum * 5] =  ((((vid(0) + chunkSize(0) )% chunkSize(0)))  +
                                                             ((vid(1) + chunkSize(1)) % chunkSize(1))  * chunkSize(0) +
                                                             ((vid(2) + chunkSize(2))% chunkSize(2))  * chunkSize(0) * chunkSize(1));
                    i++;
                }
            }
        }
    }

    void ChunkManager::GetChunkIDsIntersecting(const AABB& box, ChunkIDList* chunkList)
    {
        assert(chunkList != nullptr);

        ChunkID minID = GetIDAt(box.min);
        ChunkID maxID = GetIDAt(box.max) + Eigen::Vector3i(1, 1, 1);

        for (int x = minID(0); x < maxID(0); x++)
        {
            for (int y = minID(1); y < maxID(1); y++)
            {
                for (int z = minID(2); z < maxID(2); z++)
                {
                    chunkList->push_back(ChunkID(x, y, z));
                }
            }
        }
    }



    void ChunkManager::GetColorByProject(const PinholeCamera& camera, MeshPtr &mesh)
    {

        float cx = camera.GetCx();
        float cy = camera.GetCy();
        float fx = camera.GetFx();
        float fy = camera.GetFy();
        int width = camera.GetWidth();
        int height = camera.GetHeight();

#if 1
        const ChunkPtr &chunk = GetChunk(mesh->chunkID);
        int frameIndex;
        Mat4x4 transformation;
        const float * normalMap;
        const unsigned char * colorMap;
        const float * depthMap;
        chunk->GetReferenceFrame(frameIndex, transformation,normalMap,colorMap,depthMap);


        mesh->colors.clear();
        mesh->colors.resize(mesh->vertices.size());
        if(frameIndex  < 0)
        {
            return;
        }
//            cout << "begin update meshes!" << endl;
        for (size_t i = 0; i < mesh->vertices.size(); i++)
        {
            Vec3 vertex = mesh->vertices[i];
            Vec4 localVertex = transformation * Vec4(vertex(0),vertex(1),vertex(2),1);
            float localDepth = localVertex(2);
            localVertex = localVertex / localVertex(2);
            int cameraX = (localVertex(0)* fx + cx ) ;
            int cameraY = (localVertex(1)* fy + cy ) ;
            if(cameraX < 0 || cameraX > width - 1 || cameraY < 0 || cameraY > height - 1 )
            {
                continue;
            }
            int cameraPos = cameraY * width + cameraX;

//            if(fabs(depthMap[cameraPos] - localDepth) < 0.1)
            {

                Vec3 color = Vec3(colorMap[cameraPos * 3] / 255.0f, colorMap[cameraPos * 3 + 1] / 255.0f, colorMap[cameraPos * 3 + 2] / 255.0f);
                mesh->colors[i] = color;
            }
//                Vec3 normal = Vec3(normalMap[cameraPos * 3], normalMap[cameraPos * 3 + 1], normalMap[cameraPos * 3 + 2]);
//                mesh->normals[i] = normal;
        }
#else
            ColorizeMesh(mesh.get());
#endif

    }

    void ChunkManager::RecomputeMeshes(const ChunkSet& chunkMeshes, const PinholeCamera& camera)
    {

        if (chunkMeshes.empty())
        {
            return;
        }
        std::vector<MeshPtr> meshes;
        for (const std::pair<ChunkID, bool>& chunk : chunkMeshes)
        {

            if (!chunk.second)
            {
                continue;
            }
            ChunkID chunkID = ChunkID(chunk.first);
            if (!HasChunk(chunkID))
            {
                continue;
            }

            MeshPtr mesh;
            if (!HasMesh(chunkID))
            {
                mesh = std::allocate_shared<Mesh>(Eigen::aligned_allocator<Mesh>());
            }
            else
            {
                mesh = GetMesh(chunkID);
            }
            mesh->chunkID = chunkID;
            meshes.push_back(mesh);
        }



        TICK("CHISEL_MESHING::UpdateMeshes::GenerateMeshes");
        parallel_for(meshes.begin(),meshes.end(),[&](MeshPtr &mesh)
        {
            ChunkPtr chunk = GetChunk(mesh->chunkID);
            GenerateMeshEfficient(chunk, mesh.get());
        });
        TOCK("CHISEL_MESHING::UpdateMeshes::GenerateMeshes");
        for(MeshPtr & mesh: meshes)
        {
            if(!mesh->vertices.empty())
                allMeshes[mesh->chunkID] = mesh;
        }
    }

    void ChunkManager::CreateChunk(const ChunkID& id)
    {
        AddChunk(std::allocate_shared<Chunk>(Eigen::aligned_allocator<Chunk>(), id, chunkSize, voxelResolutionMeters, useColor));
    }

    void ChunkManager::Reset()
    {
        allMeshes.clear();
        chunks.clear();
    }


    bool ChunkManager::extractGradientFromCubic(float *cubicSDFPointer,
                                  Point3 &currentVoxelID,
                                  int cubicVoxelID,
                                  int cubicVoxelIndex,
                                  float   *neighborSDFPointer,
                                  ChunkID &neighborChunkID,
                                  Eigen::Vector3f &grad)
    {
        if(neighborSDFPointer == NULL)
        {
            return false;
        }
        static int voxelsInChunk = chunkSize(0) * chunkSize(1) * chunkSize(2);
        Point3 nearestVoxelCurrentPoint = currentVoxelID + cubeIndexOffsets.col(cubicVoxelID);
        int cornerVoxelIndex = cubicVoxelIndex;
        ChunkID nCC[6];
        bool edgeFlag[6];
        int nVI[6];
        nCC[0] = ChunkID(neighborChunkID(0)-1,neighborChunkID(1),neighborChunkID(2));
        nCC[1] = ChunkID(neighborChunkID(0)+1,neighborChunkID(1),neighborChunkID(2));
        nCC[2] = ChunkID(neighborChunkID(0),neighborChunkID(1)-1,neighborChunkID(2));
        nCC[3] = ChunkID(neighborChunkID(0),neighborChunkID(1)+1,neighborChunkID(2));
        nCC[4] = ChunkID(neighborChunkID(0),neighborChunkID(1),neighborChunkID(2)-1);
        nCC[5] = ChunkID(neighborChunkID(0),neighborChunkID(1),neighborChunkID(2)+1);
        edgeFlag[0] = (nearestVoxelCurrentPoint(0) == 0);
        edgeFlag[1] = (nearestVoxelCurrentPoint(0) == chunkSize(0) - 1);
        edgeFlag[2] = (nearestVoxelCurrentPoint(1) == 0);
        edgeFlag[3] = (nearestVoxelCurrentPoint(1) == chunkSize(1) - 1);
        edgeFlag[4] = (nearestVoxelCurrentPoint(2) == 0);
        edgeFlag[5] = (nearestVoxelCurrentPoint(2) == chunkSize(2) - 1);
        nVI[0] = voxelNeighborIndex[cornerVoxelIndex + voxelsInChunk * 0];
        nVI[1] = voxelNeighborIndex[cornerVoxelIndex + voxelsInChunk * 1];
        nVI[2] = voxelNeighborIndex[cornerVoxelIndex + voxelsInChunk * 2];
        nVI[3] = voxelNeighborIndex[cornerVoxelIndex + voxelsInChunk * 3];
        nVI[4] = voxelNeighborIndex[cornerVoxelIndex + voxelsInChunk * 4];
        nVI[5] = voxelNeighborIndex[cornerVoxelIndex + voxelsInChunk * 5];
        float *nSP = neighborSDFPointer;
        float dd[6];
        int k = 0;
#if TEST_GRADIENT
        std::cout << "cubic ID: " << cubicVoxelID << std::endl;
        std::cout << "voxel id: " << nearestVoxelCurrentPoint.transpose() << std::endl;
#endif

        switch(cubicVoxelID){
        case 0:
            dd[1] = cubicSDFPointer[1];
            dd[3] = cubicSDFPointer[3];
            dd[5] = cubicSDFPointer[4];
            k = 0; if(!GetNeighborSDF(edgeFlag[k], nCC[k],nVI[k],nSP,dd[k])) return false;
            k = 2; if(!GetNeighborSDF(edgeFlag[k], nCC[k],nVI[k],nSP,dd[k])) return false;
            k = 4; if(!GetNeighborSDF(edgeFlag[k], nCC[k],nVI[k],nSP,dd[k])) return false;
            break;
        case 1:
            dd[0] = cubicSDFPointer[0];
            dd[3] = cubicSDFPointer[2];
            dd[5] = cubicSDFPointer[5];
            k = 1; if(!GetNeighborSDF(edgeFlag[k], nCC[k],nVI[k],nSP,dd[k])) return false;
            k = 2; if(!GetNeighborSDF(edgeFlag[k], nCC[k],nVI[k],nSP,dd[k])) return false;
            k = 4; if(!GetNeighborSDF(edgeFlag[k], nCC[k],nVI[k],nSP,dd[k])) return false;
            break;
        case 2:
            dd[0] = cubicSDFPointer[3];
            dd[2] = cubicSDFPointer[1];
            dd[5] = cubicSDFPointer[6];
            k = 1; if(!GetNeighborSDF(edgeFlag[k], nCC[k],nVI[k],nSP,dd[k])) return false;
            k = 3; if(!GetNeighborSDF(edgeFlag[k], nCC[k],nVI[k],nSP,dd[k])) return false;
            k = 4; if(!GetNeighborSDF(edgeFlag[k], nCC[k],nVI[k],nSP,dd[k])) return false;
            break;
        case 3:
            dd[1] = cubicSDFPointer[2];
            dd[2] = cubicSDFPointer[0];
            dd[5] = cubicSDFPointer[7];
            k = 0; if(!GetNeighborSDF(edgeFlag[k], nCC[k],nVI[k],nSP,dd[k])) return false;
            k = 3; if(!GetNeighborSDF(edgeFlag[k], nCC[k],nVI[k],nSP,dd[k])) return false;
            k = 4; if(!GetNeighborSDF(edgeFlag[k], nCC[k],nVI[k],nSP,dd[k])) return false;
            break;
        case 4:
            dd[1] = cubicSDFPointer[5];
            dd[3] = cubicSDFPointer[7];
            dd[4] = cubicSDFPointer[0];
            k = 0; if(!GetNeighborSDF(edgeFlag[k], nCC[k],nVI[k],nSP,dd[k])) return false;
            k = 2; if(!GetNeighborSDF(edgeFlag[k], nCC[k],nVI[k],nSP,dd[k])) return false;
            k = 5; if(!GetNeighborSDF(edgeFlag[k], nCC[k],nVI[k],nSP,dd[k])) return false;
            break;
        case 5:
            dd[0] = cubicSDFPointer[4];
            dd[3] = cubicSDFPointer[6];
            dd[4] = cubicSDFPointer[1];
            k = 1; if(!GetNeighborSDF(edgeFlag[k], nCC[k],nVI[k],nSP,dd[k])) return false;
            k = 2; if(!GetNeighborSDF(edgeFlag[k], nCC[k],nVI[k],nSP,dd[k])) return false;
            k = 5; if(!GetNeighborSDF(edgeFlag[k], nCC[k],nVI[k],nSP,dd[k])) return false;
            break;
        case 6:
            dd[0] = cubicSDFPointer[7];
            dd[2] = cubicSDFPointer[5];
            dd[4] = cubicSDFPointer[2];
            k = 1; if(!GetNeighborSDF(edgeFlag[k], nCC[k],nVI[k],nSP,dd[k])) return false;
            k = 3; if(!GetNeighborSDF(edgeFlag[k], nCC[k],nVI[k],nSP,dd[k])) return false;
            k = 5; if(!GetNeighborSDF(edgeFlag[k], nCC[k],nVI[k],nSP,dd[k])) return false;
            break;
        case 7:
            dd[1] = cubicSDFPointer[6];
            dd[2] = cubicSDFPointer[4];
            dd[4] = cubicSDFPointer[3];
            k = 0; if(!GetNeighborSDF(edgeFlag[k], nCC[k],nVI[k],nSP,dd[k])) return false;
            k = 3; if(!GetNeighborSDF(edgeFlag[k], nCC[k],nVI[k],nSP,dd[k])) return false;
            k = 5; if(!GetNeighborSDF(edgeFlag[k], nCC[k],nVI[k],nSP,dd[k])) return false;
            break;
        }
#if TEST_GRADIENT
        cout << dd[0] << endl << dd[1] << endl << dd[2] << endl
             << dd[3] << endl << dd[4] << endl << dd[5] << endl << endl;
#endif


        grad = Eigen::Vector3f(dd[1] - dd[0], dd[3] - dd[2], dd[5] - dd[4]);
        float g = grad.norm();
        grad.normalize();
        if(g > voxelResolutionMeters *100)
        {
            return false;
        }
        return true;
    }
    void ChunkManager::GetChunkIDsIntersecting(const Frustum& frustum, ChunkIDList* chunkList)
    {
        assert(chunkList != nullptr);

        AABB frustumAABB;
        frustum.ComputeBoundingBox(&frustumAABB);

        ChunkID minID = GetIDAt(frustumAABB.min);
        ChunkID maxID = GetIDAt(frustumAABB.max) + Eigen::Vector3i(1, 1, 1);

        printf("FrustumAABB: %f %f %f %f %f %f\n", frustumAABB.min.x(), frustumAABB.min.y(), frustumAABB.min.z(), frustumAABB.max.x(), frustumAABB.max.y(), frustumAABB.max.z());
        printf("Frustum min: %d %d %d max: %d %d %d\n", minID.x(), minID.y(), minID.z(), maxID.x(), maxID.y(), maxID.z());
        for (int x = minID(0) - 1; x <= maxID(0) + 1; x++)
        {
            for (int y = minID(1) - 1; y <= maxID(1) + 1; y++)
            {
                for (int z = minID(2) - 1; z <= maxID(2) + 1; z++)
                {
                    Vec3 min = Vec3(x * chunkSize(0), y * chunkSize(1), z * chunkSize(2)) * voxelResolutionMeters;
                    Vec3 max = min + chunkSize.cast<float>() * voxelResolutionMeters;
                    AABB chunkBox(min, max);
                    if(frustum.Intersects(chunkBox))
                    {
                        chunkList->push_back(ChunkID(x, y, z));
                    }
                }
            }
        }

        printf("%lu chunks intersect frustum\n", chunkList->size());
    }



    void ChunkManager::ExtractInsideVoxelMesh(const ChunkPtr& chunk, const Eigen::Vector3i& index, const Vec3& coords, VertIndex* nextMeshIndex, Mesh* mesh)
    {
        assert(mesh != nullptr);
        Eigen::Matrix<float, 3, 8> cubeCoordOffsets = cubeIndexOffsets.cast<float>() * voxelResolutionMeters;
        Eigen::Matrix<float, 3, 8> cornerCoords;
        Eigen::Matrix<float, 8, 1> cornerSDF;
        bool allNeighborsObserved = true;


        for (int i = 0; i < 8; ++i)
        {
            Eigen::Vector3i corner_index = index + cubeIndexOffsets.col(i);
            int voxelIndex = chunk->GetVoxelID(corner_index.x(), corner_index.y(), corner_index.z());
            const DistVoxel& thisVoxel = chunk->voxels;

            // Do not extract a mesh here if one of the corner is unobserved and
            // outside the truncation region.
            if (thisVoxel.GetWeight(voxelIndex) <= 1e-15)
            {
                allNeighborsObserved = false;
                break;
            }
            cornerCoords.col(i) = coords + cubeCoordOffsets.col(i);
            cornerSDF(i) = thisVoxel.GetSDF(voxelIndex);
        }

        if (allNeighborsObserved)
        {
            MarchingCubes::MeshCube(cornerCoords, cornerSDF, nextMeshIndex, mesh);
        }
    }

    void ChunkManager::ExtractBorderVoxelMesh(const ChunkPtr& chunk, const Eigen::Vector3i& index, const Eigen::Vector3f& coordinates, VertIndex* nextMeshIndex, Mesh* mesh)
    {
        const Eigen::Matrix<float, 3, 8> cubeCoordOffsets = cubeIndexOffsets.cast<float>() * voxelResolutionMeters;
        Eigen::Matrix<float, 3, 8> cornerCoords;
        Eigen::Matrix<float, 8, 1> cornerSDF;
        bool allNeighborsObserved = true;
        for (int i = 0; i < 8; ++i)
        {
            Eigen::Vector3i cornerIDX = index + cubeIndexOffsets.col(i);

            if (chunk->IsCoordValid(cornerIDX.x(), cornerIDX.y(), cornerIDX.z()))
            {
                const DistVoxel& thisVoxel = chunk->voxels;
                int voxelIndex = chunk->GetVoxelID(cornerIDX.x(), cornerIDX.y(), cornerIDX.z());
                // Do not extract a mesh here if one of the corners is unobserved
                // and outside the truncation region.

                if (thisVoxel.GetWeight(voxelIndex) <= 1e-15)
                {
                    allNeighborsObserved = false;
                    break;
                }
                cornerCoords.col(i) = coordinates + cubeCoordOffsets.col(i);
                cornerSDF(i) = thisVoxel.GetSDF(voxelIndex);
            }
            else
            {
                Eigen::Vector3i chunkOffset = Eigen::Vector3i::Zero();


                for (int j = 0; j < 3; j++)
                {
                    if (cornerIDX(j) < 0)
                    {
                        chunkOffset(j) = -1;
                        cornerIDX(j) = chunkSize(j) - 1;

                    }
                    else if(cornerIDX(j) >= chunkSize(j))
                    {
                        chunkOffset(j) = 1;
                        cornerIDX(j) = 0;
                    }
                }

                ChunkID neighborID = chunkOffset + chunk->GetID();

                if (HasChunk(neighborID))
                {
                    const ChunkPtr& neighborChunk = GetChunk(neighborID);
                    if(!neighborChunk->IsCoordValid(cornerIDX.x(), cornerIDX.y(), cornerIDX.z()))
                    {
                        allNeighborsObserved = false;
                        break;
                    }

                    const DistVoxel& thisVoxel = neighborChunk->voxels;
                    int voxelIndex = neighborChunk->GetVoxelID(cornerIDX.x(), cornerIDX.y(), cornerIDX.z());
                    // Do not extract a mesh here if one of the corners is unobserved
                    // and outside the truncation region.
                    if (thisVoxel.GetWeight(voxelIndex) <= 1e-15)
                    {
                        allNeighborsObserved = false;
                        break;
                    }
                    cornerCoords.col(i) = coordinates + cubeCoordOffsets.col(i);
                    cornerSDF(i) = thisVoxel.GetSDF(voxelIndex);
                }
                else
                {
                    allNeighborsObserved = false;
                    break;
                }

            }

        }

        if (allNeighborsObserved)
        {
            MarchingCubes::MeshCube(cornerCoords, cornerSDF, nextMeshIndex, mesh);
        }
    }


    // can be acclerated using SIMD, which will lead to further improvements
    void ChunkManager::GenerateMeshEfficient(const ChunkPtr& chunk, Mesh* mesh)
    {
        assert(mesh != nullptr);

        mesh->Clear();
        const int maxX = chunkSize(0);
        const int maxY = chunkSize(1);
        const int maxZ = chunkSize(2);

        VoxelID i = 0;
        VertIndex nextIndex = 0;
        int voxelsInChunk = maxX * maxY * maxZ;

        int chunkExsitFlag[8];
        float * sdfPointer[8];
        float *weightPointer[8];
        unsigned short *colorPointer[8];
        chunkExsitFlag[0] = 1;
        ChunkID cID = chunk->GetID();
        ChunkID neighborChunkID[8];
        sdfPointer[0] = chunk->voxels.sdf.data();
        weightPointer[0] = chunk->voxels.weight.data();
        colorPointer[0] = chunk->colors.colorData;
        neighborChunkID[0] = cID;
        for(int i = 1; i < 8; i++)
        {
            ChunkID newChunkID = cID + ChunkID((i%2), (i%4)/2, i / 4);
            neighborChunkID[i] = newChunkID;
            chunkExsitFlag[i] =  HasChunk(newChunkID);
            if(chunkExsitFlag[i])
            {
                const ChunkPtr & neighborChunk = GetChunk(newChunkID);
                sdfPointer[i] = neighborChunk->voxels.sdf.data();
                weightPointer[i] = neighborChunk->voxels.weight.data();
                colorPointer[i] = neighborChunk->colors.colorData;
            }
            else
            {
                colorPointer[i] = NULL;
                sdfPointer[i] = NULL;
                weightPointer[i] = NULL;
            }
        }

        __m256i shiftCountSIMD = _mm256_set_epi32(7,6,5,4,3,2,1,0);
        static Eigen::Matrix<float, 3, 8> cubeCoordOffsets = cubeIndexOffsets.cast<float>() * voxelResolutionMeters;
        __m256 cornerSDF;

        // For voxels not bordering the outside, we can use a more efficient function.
//        Vec3 origin =
        int x,y,z;
// Timing result: 40ms for inside voxels and 30ms for outside voxels
// 38ms for datapreparation and 30ms for mesh extraction


        float cornerWeight[8];
        __m256i *voxelNeighborIndexPtr = (__m256i *)&neighborVoxelIndexChunkWise[0];
        for (z = 0; z < maxZ; z++)
        {
            for (y = 0; y < maxY; y++)
            {
                for (x = 0; x < maxX; x++)
                {

                    Point3 current_voxel = Point3(x,y,z);
                    int voxelIndex = chunk->GetVoxelID(current_voxel);
                    memset(cornerWeight, 0, 32);
#if 0
                    if(fabs(chunk->voxels.sdf[i]) > 0.1)
                    {
                        continue;
                    }
#endif
                    bool allNeighborsObserved = true;
//                    unsigned char * colorData = &chunk->colors.colorData[voxelIndex * 4];
                    Vec3 origin = chunk->GetOrigin() + centroids[voxelIndex];
                    int colorChangeFlag = 0;
//                    int marchCubeIndex = 0;

                    if(z == maxZ - 1 || y == maxY - 1 || x == maxX - 1)
                    {
                        for(int k = 0; k < 8; k++)
                        {
                            int index = neighborChunkIndexChunkWise[voxelIndex*8 + k];
                            if(!chunkExsitFlag[index])
                            {
                                allNeighborsObserved = false;
                                break;
                            }
                            int cornerVoxelIndex = neighborVoxelIndexChunkWise[voxelIndex*8 + k];
                            float sdf = sdfPointer[index][cornerVoxelIndex];
                            cornerWeight[k] = weightPointer[index][cornerVoxelIndex];

                            if(sdf > 1)
                            {
                                allNeighborsObserved = false;
                                break;
                            }
                            cornerSDF[k] = sdf;
                            colorChangeFlag += sdf > 0;
                        }

                    }
                    else
                    {

#if 0
                        __m256i cornerVoxelIndex = _mm256_load_si256(&voxelNeighborIndexPtr[voxelIndex]);
                        cornerSDF = _mm256_i32gather_ps(sdfPointer[0],cornerVoxelIndex,4);
                        __m256 validFlag = _mm256_cmp_ps(cornerSDF,_mm256_set1_ps(1),_CMP_GT_OS);
                        // only if all of validFlag is zero, valid
                        //
                        if(!_mm256_testz_ps(validFlag,validFlag))
                        {
                            allNeighborsObserved = false;
                        }
                        validFlag = _mm256_cmp_ps(cornerSDF,_mm256_set1_ps(0),_CMP_GT_OS);
                        colorChangeFlag = 1;
                        if(_mm256_testz_ps(validFlag,validFlag) || _mm256_testc_ps(validFlag,_mm256_castsi256_ps(_mm256_set1_epi64x(-1))))
                        {
                            colorChangeFlag = 0;
                        }
#else
                        for(int k = 0; k < 8; k++)
                        {
                            int cornerVoxelIndex = neighborVoxelIndexChunkWise[voxelIndex*8 +k];
                            float sdf = sdfPointer[0][cornerVoxelIndex];
                            cornerWeight[k] = weightPointer[0][cornerVoxelIndex];
                            if(sdf > 1)
                            {
                                allNeighborsObserved = false;
                                break;
                            }
                            cornerSDF[k] = sdf;
                            colorChangeFlag += sdf > 0;
                        }
#endif
                    }

                    if(allNeighborsObserved && colorChangeFlag %8 > 0)
                    {
#if 0
                          nextIndex += 3;mesh->normals.push_back(Vec3(0,0,0));
#else
                        __m256i negativeFlag = _mm256_castps_si256(_mm256_cmp_ps(_mm256_set1_ps(0),cornerSDF,_CMP_GT_OS));
                        negativeFlag = _mm256_and_si256(negativeFlag,_mm256_set1_epi32(1));
                        negativeFlag = _mm256_sllv_epi32(negativeFlag,shiftCountSIMD);

                        __m256i hsum = _mm256_hadd_epi32(negativeFlag,negativeFlag);
                        hsum = _mm256_add_epi32(hsum,_mm256_permute2x128_si256(hsum,hsum,1));
                        hsum = _mm256_hadd_epi32(hsum,hsum);
                        int index = _mm256_extract_epi32(hsum,0);


                        Eigen::Matrix<float, 3, 12> edge_vertex_coordinates;
                        Eigen::Matrix<float, 3, 12> edge_normal_coordinates;
                        Eigen::Matrix<float, 3, 12> edge_color_coordinates;
                        int normalValidFlag[12];

                        const int* table_row = MarchingCubes::triangleTable[index];

                        int table_col = 0;
                        float cubicSDFValue[8];
                        _mm256_storeu_ps(cubicSDFValue,cornerSDF);
                        if(table_row[table_col] != -1)
                        {
                            for (std::size_t i = 0; i < 12; ++i)
                            {
                                normalValidFlag[i] = 0;
                                int normal_exist = 0;
                                const int* pairs = edgeIndexPairs[i];
                                const int edge0 = pairs[0];
                                const int edge1 = pairs[1];
                                float sdf0 = cubicSDFValue[edge0];
                                float sdf1 = cubicSDFValue[edge1];
                                // Only interpolate along edges where there is a zero crossing.
                                if ((sdf0 > 0 && sdf1 < 0) || (sdf0 < 0 && sdf1 > 0))
                                {

                                    float t = sdf0 / (sdf0 - sdf1);
                                    edge_vertex_coordinates.col(i) = cubeCoordOffsets.col(edge0) + t * (cubeCoordOffsets.col(edge1) - cubeCoordOffsets.col(edge0));
                                    // the normal of a voxel
                                    int cubicVoxelID = edgeIndexPairs[i][fabs(sdf0) > fabs(sdf1)];

                                    int neighborChunkIndex = neighborChunkIndexChunkWise[voxelIndex*8 + cubicVoxelID];
                                    Eigen::Vector3f grad;

                                    int neighborVoxelIndex = neighborVoxelIndexChunkWise[voxelIndex*8 +cubicVoxelID];
                                    normal_exist = extractGradientFromCubic(cubicSDFValue,
                                                             current_voxel,
                                                             cubicVoxelID,
                                                             neighborVoxelIndex,
                                                             sdfPointer[neighborChunkIndex],
                                                             neighborChunkID[neighborChunkIndex],
                                                             grad);

                                    float weight_threshold = 50;
                                    if(cornerWeight[cubicVoxelID] > weight_threshold)
                                    {
                                        edge_normal_coordinates.col(i) = grad;
#if 0
                                        int interpolateColorFlag = -1;
                                        Eigen::Vector4f colorInterpoloated[2];
                                        for(int cnt = 0; cnt < 2; cnt++)
                                        {
                                            int cubicVoxelIDColor = edgeIndexPairs[i][cnt];
                                            int neighborChunkIndexColor = neighborChunkIndexChunkWise[voxelIndex*8 + cubicVoxelID];
                                            int neighborVoxelIndexColor = neighborVoxelIndexChunkWise[voxelIndex*8 +cubicVoxelID];
                                            colorInterpoloated[cnt] = Eigen::Vector4f(
                                                                colorPointer[neighborChunkIndexColor][neighborVoxelIndexColor * 4] / 255.0f,
                                                                colorPointer[neighborChunkIndexColor][neighborVoxelIndexColor * 4 + 1] / 255.0f,
                                                                colorPointer[neighborChunkIndexColor][neighborVoxelIndexColor * 4 + 2] / 255.0f,
                                                                colorPointer[neighborChunkIndexColor][neighborVoxelIndexColor * 4 + 3]);

                                        }
                                        Eigen::Vector4f c;
                                        c = colorInterpoloated[0] * (1-t) + colorInterpoloated[1] * t;
                                        if(c(3) > 0)
                                        {
                                            c = c/c(3);
                                            edge_color_coordinates.col(i) = Eigen::Vector3f(c(0),c(1),c(2));
                                        }
                                        else
                                        {
                                            //cornerWeight[cubicVoxelID] = 0;
                                            edge_color_coordinates.col(i) = Eigen::Vector3f(0.0f, 0.0f, 0.0f);
                                        }
#else
                                        float colorWeight = colorPointer[neighborChunkIndex][neighborVoxelIndex * 4+3];
                                        if(colorWeight > 0)
                                        {
                                            edge_color_coordinates.col(i) = Eigen::Vector3f(
                                                        colorPointer[neighborChunkIndex][neighborVoxelIndex * 4] / 255.0f / colorWeight,
                                                        colorPointer[neighborChunkIndex][neighborVoxelIndex * 4 + 1] / 255.0f / colorWeight,
                                                        colorPointer[neighborChunkIndex][neighborVoxelIndex * 4 + 2] / 255.0f / colorWeight);
                                        }
                                        else
                                        {
                                            edge_color_coordinates.col(i) = Eigen::Vector3f(1.0f, 1.0f, 1.0f);
                                        }
#endif
                                        normalValidFlag[i] = normal_exist;
                                    }
                                }
                            }
                        }

                        while (table_row[table_col] != -1)
                        {
                            int s2 = table_row[table_col + 2];
                            int s1 = table_row[table_col + 1];
                            int s0 = table_row[table_col];

                            if(!normalValidFlag[s2] || !normalValidFlag[s1] || !normalValidFlag[s0])
                            {
                                table_col += 3;
                                continue;
                            }

                            mesh->vertices.emplace_back(edge_vertex_coordinates.col(s2)+origin);
                            mesh->vertices.emplace_back(edge_vertex_coordinates.col(s1)+origin);
                            mesh->vertices.emplace_back(edge_vertex_coordinates.col(s0)+origin);
                            mesh->indices.push_back(nextIndex);
                            mesh->indices.push_back((nextIndex) + 1);
                            mesh->indices.push_back((nextIndex) + 2);
                            mesh->normals.push_back(edge_normal_coordinates.col(s2));
                            mesh->normals.push_back(edge_normal_coordinates.col(s1));
                            mesh->normals.push_back(edge_normal_coordinates.col(s0));
                            mesh->colors.push_back(edge_color_coordinates.col(s2));
                            mesh->colors.push_back(edge_color_coordinates.col(s1));
                            mesh->colors.push_back(edge_color_coordinates.col(s0));
                            nextIndex += 3;
                            table_col += 3;
                        }
#endif
                    }
                }
            }
        }
#if 0
        for (z = 0; z < maxZ; z++)
        {
            for (y = 0; y < maxY; y++)
            {
                for (x = 0; x < maxX; x++)
                {
                    int voxelIndex = chunk->GetVoxelID(x,y,z);
                    bool allNeighborsObserved = true;
                    unsigned char * colorData = &chunk->colors.colorData[voxelIndex * 4];
                    Vec3 origin = chunk->GetOrigin() + centroids[voxelIndex];
                    int colorChangeFlag = 0;

#if 0
/************************SIMD*******************************/
                    __m256i cornerX = _mm256_set1_epi32(x) + cubeIndexOffsetsSIMD[0];
                    __m256i cornerY = _mm256_set1_epi32(y) + cubeIndexOffsetsSIMD[1];
                    __m256i cornerZ = _mm256_set1_epi32(z) + cubeIndexOffsetsSIMD[2];

                    __m256i indexX = _mm256_and_si256(_mm256_cmpeq_epi32(cornerX, maxX), _mm256_set1_epi32(1));
                    __m256i indexY = _mm256_and_si256(_mm256_cmpeq_epi32(cornerY, maxY), _mm256_set1_epi32(2));
                    __m256i indexZ = _mm256_and_si256(_mm256_cmpeq_epi32(cornerZ, maxZ), _mm256_set1_epi32(4));
                    __m256i index = _mm256_add_epi32(indexX,indexY);
                    index = _mm256_add_epi32(index,indexZ);
                    // use shuffle to check if all chunks are valid
                    __m256i validSIMD = _mm256_permutevar8x32_epi32(chunksExsitingFlagSIMD,index);
                    // to calculate cornerVoxelIndex
                    indexX = _mm256_and_si256(cornerX, _mm256_set1_epi32(7));
                    indexY = _mm256_slli_epi32(_mm256_and_si256(cornerY, _mm256_set1_epi32(7)),3);
                    indexZ = _mm256_slli_epi32(_mm256_and_si256(cornerZ, _mm256_set1_epi32(7)),6);
                    __m256i cornerVoxelIndex = _mm256_add_epi32(indexX, indexY);
                    cornerVoxelIndex = _mm256_add_epi32(cornerVoxelIndex, indexZ);


#endif
                    for(int k = 0; k < 8; k++)
                    {
                        cornerX = x + cubeIndexOffsets(0,k);
                        cornerY = y + cubeIndexOffsets(1,k);
                        cornerZ = z + cubeIndexOffsets(2,k);

                        int index = (cornerX == maxX) + (cornerY == maxY) * 2 + (cornerZ == maxZ) * 4;

                        if(!chunkExsitFlag[index])
                        {
                            allNeighborsObserved = false;
                            break;
                        }

#if 1
                        int cornerVoxelIndex = cornerX % 8 + cornerY % 8 * 8 + cornerZ % 8 * 64;

                        float sdf = sdfPointer[index][cornerVoxelIndex];
                        if(fabs(sdf) > 0.06)
                        {
                            allNeighborsObserved = false;
                            break;
                        }

                        cornerSDF[k] = sdf;
                        cornerCoords(0,k) = origin(0) + cubeCoordOffsets(0,k);
                        cornerCoords(1,k) = origin(1) + cubeCoordOffsets(1,k);
                        cornerCoords(2,k) = origin(2) + cubeCoordOffsets(2,k);
                        colorChangeFlag += sdf > 0;

#endif
                    }

                    if(allNeighborsObserved && colorChangeFlag % 8 > 0)
                    {
                        MarchingCubes::MeshCube(cornerCoords, cornerSDF, &nextIndex, mesh,colorData);
                    }
                }
            }
        }
#endif

//        mesh->vertices.push_back(Vec3(cornerSDF[0],cornerSDF[1],cornerSDF[2]));
//        printf("Generated a new mesh with %lu verts, %lu norm, and %lu idx\n", mesh->vertices.size(), mesh->normals.size(), mesh->indices.size());

        assert(mesh->vertices.size() == mesh->normals.size());
    }
    void ChunkManager::GenerateMesh(const ChunkPtr& chunk, Mesh* mesh)
    {
        assert(mesh != nullptr);

        mesh->Clear();
        const int maxX = chunkSize(0);
        const int maxY = chunkSize(1);
        const int maxZ = chunkSize(2);


        Eigen::Vector3i index;
        VoxelID i = 0;
        VertIndex nextIndex = 0;

        // For voxels not bordering the outside, we can use a more efficient function.
        for (index.z() = 0; index.z() < maxZ; index.z()++)
        {
            for (index.y() = 0; index.y() < maxY; index.y()++)
            {
                for (index.x() = 0; index.x() < maxX; index.x()++)
                {
                    i = chunk->GetVoxelID(index.x(), index.y(), index.z());
                    if(index.x() == maxX - 1 || index.y() == maxY - 1 || index.z() == maxZ - 1)
                    {
                        ExtractBorderVoxelMesh(chunk, index, centroids.at(i) + chunk->GetOrigin(), &nextIndex, mesh);
                    }
                    else
                    {
                        ExtractInsideVoxelMesh(chunk, index, centroids.at(i) + chunk->GetOrigin(), &nextIndex, mesh);
                    }
                }
            }
        }

//        printf("Generated a new mesh with %lu verts, %lu norm, and %lu idx\n", mesh->vertices.size(), mesh->normals.size(), mesh->indices.size());

        assert(mesh->vertices.size() == mesh->normals.size());
        assert(mesh->vertices.size() == mesh->indices.size());
    }

    bool ChunkManager::GetSDFAndGradient(const Eigen::Vector3f& pos, Eigen::Vector3f& grad)\
    {


#if 0
        Eigen::Vector3f posf = Eigen::Vector3f(std::floor(pos.x() / voxelResolutionMeters) * voxelResolutionMeters + voxelResolutionMeters / 2.0f,
                std::floor(pos.y() / voxelResolutionMeters) * voxelResolutionMeters + voxelResolutionMeters / 2.0f,
                std::floor(pos.z() / voxelResolutionMeters) * voxelResolutionMeters + voxelResolutionMeters / 2.0f);
        if (!GetSDF(posf, dist)) return false;
        double ddxplus, ddyplus, ddzplus = 0.0;
        double ddxminus, ddyminus, ddzminus = 0.0;
        if (!GetSDF(posf + Eigen::Vector3f(voxelResolutionMeters, 0, 0), &ddxplus)) return false;
        if (!GetSDF(posf + Eigen::Vector3f(0, voxelResolutionMeters, 0), &ddyplus)) return false;
        if (!GetSDF(posf + Eigen::Vector3f(0, 0, voxelResolutionMeters), &ddzplus)) return false;
        if (!GetSDF(posf - Eigen::Vector3f(voxelResolutionMeters, 0, 0), &ddxminus)) return false;
        if (!GetSDF(posf - Eigen::Vector3f(0, voxelResolutionMeters, 0), &ddyminus)) return false;
        if (!GetSDF(posf - Eigen::Vector3f(0, 0, voxelResolutionMeters), &ddzminus)) return false;

        *grad = Eigen::Vector3f(ddxplus - ddxminus, ddyplus - ddyminus, ddzplus - ddzminus);
        grad->normalize();
#else
        Eigen::Vector3f posf = Eigen::Vector3f(std::floor(pos.x() / voxelResolutionMeters) * voxelResolutionMeters + voxelResolutionMeters / 2.0f,
                std::floor(pos.y() / voxelResolutionMeters) * voxelResolutionMeters + voxelResolutionMeters / 2.0f,
                std::floor(pos.z() / voxelResolutionMeters) * voxelResolutionMeters + voxelResolutionMeters / 2.0f);


        int voxelsInChunk = chunkSize(0) * chunkSize(1) * chunkSize(2);
        static const float roundingVoxelX = 1.0f / (voxelResolutionMeters);
        static const float roundingVoxelY = 1.0f / (voxelResolutionMeters);
        static const float roundingVoxelZ = 1.0f / (voxelResolutionMeters);
        static const float roundingChunkX = 1.0f / (voxelResolutionMeters * chunkSize(0));
        static const float roundingChunkY = 1.0f / (voxelResolutionMeters * chunkSize(1));
        static const float roundingChunkZ = 1.0f / (voxelResolutionMeters * chunkSize(2));
        Eigen::Vector3i vidGlobal = Eigen::Vector3i(static_cast<int>(std::floor(posf(0) * roundingVoxelX)),
                              static_cast<int>(std::floor(posf(1) * roundingVoxelY )),
                              static_cast<int>(std::floor(posf(2) * roundingVoxelZ )));

        Eigen::Vector3i cid = Eigen::Vector3i(static_cast<int>(std::floor(posf(0) * roundingChunkX)),
                                              static_cast<int>(std::floor(posf(1) * roundingChunkY)),
                                              static_cast<int>(std::floor(posf(2) * roundingChunkZ)));


        Eigen::Vector3i vid = Eigen::Vector3i(vidGlobal(0) - cid(0) * chunkSize(0),
                                              vidGlobal(1) - cid(1) * chunkSize(1),
                                              vidGlobal(2) - cid(2) * chunkSize(2));

#if TEST_GRADIENT
        cout << "normal voxel id: " << vid.transpose() << endl;
#endif

        // look up table for 512 neighbors
        if(!HasChunk(cid))  return false;
        // check if chunk exsit

        const chisel::ChunkPtr &chunkCentral = GetChunk(cid);

        float ddxplus, ddyplus, ddzplus = 0.0;
        float ddxminus, ddyminus, ddzminus = 0.0;
        int voxelIndex = vid(0) + vid(1) * chunkSize(0) + vid(2) * chunkSize(0) * chunkSize(1);


        //for current voxel
        //for the xminus
#if TEST_GRADIENT
        cout << "normal selection: cid " << cid.transpose() << " " << voxelIndex << endl;
#endif
        if(!GetNeighborSDF(vid(0) == 0,
                           Eigen::Vector3i(cid(0)-1,cid(1),cid(2)),
                           voxelNeighborIndex[voxelIndex + voxelsInChunk * 0],
                           chunkCentral,
                           ddxminus)) return false;
        if(!GetNeighborSDF(vid(0) == chunkSize(0) - 1,
                           Eigen::Vector3i(cid(0)+1,cid(1),cid(2)),
                           voxelNeighborIndex[voxelIndex + voxelsInChunk * 1],
                           chunkCentral,
                           ddxplus)) return false;
        if(!GetNeighborSDF(vid(1) == 0,
                           Eigen::Vector3i(cid(0),cid(1) - 1,cid(2)),
                           voxelNeighborIndex[voxelIndex + voxelsInChunk * 2],
                           chunkCentral,
                           ddyminus)) return false;
        if(!GetNeighborSDF(vid(1) == chunkSize(1) - 1,
                           Eigen::Vector3i(cid(0),cid(1) + 1,cid(2)),
                           voxelNeighborIndex[voxelIndex + voxelsInChunk * 3],
                           chunkCentral,
                           ddyplus)) return false;
        if(!GetNeighborSDF(vid(2) == 0,
                           Eigen::Vector3i(cid(0),cid(1),cid(2) - 1),
                           voxelNeighborIndex[voxelIndex + voxelsInChunk * 4],
                           chunkCentral,
                           ddzminus)) return false;
        if(!GetNeighborSDF(vid(2) == chunkSize(2) - 1,
                           Eigen::Vector3i(cid(0),cid(1),cid(2) + 1),
                           voxelNeighborIndex[voxelIndex + voxelsInChunk * 5],
                           chunkCentral,
                           ddzplus)) return false;

        grad = Eigen::Vector3f(ddxplus - ddxminus, ddyplus - ddyminus, ddzplus - ddzminus);
//        grad->normalize();


#endif
        return true;
    }


    bool ChunkManager::GetWeight(const Eigen::Vector3f& posf, double* weight)
    {
        chisel::ChunkPtr chunk = GetChunkAt(posf);
        if(chunk)
        {
            Eigen::Vector3f relativePos = posf - chunk->GetOrigin();
            Eigen::Vector3i coords = chunk->GetVoxelCoords(relativePos);
            chisel::VoxelID id = chunk->GetVoxelID(coords);
            if(id >= 0 && id < chunk->GetTotalNumVoxels())
            {
                const chisel::DistVoxel& voxel = chunk->voxels;
                *weight = voxel.GetWeight((id));
                return true;
            }
            return false;
        }
        else
        {
            return false;
        }
    }

    bool ChunkManager::GetSDF(const Eigen::Vector3f& posf, double* dist)
    {
        chisel::ChunkPtr chunk = GetChunkAt(posf);
        if(chunk)
        {
            Eigen::Vector3f relativePos = posf - chunk->GetOrigin();
            Eigen::Vector3i coords = chunk->GetVoxelCoords(relativePos);
            chisel::VoxelID id = chunk->GetVoxelID(coords);
            if(id >= 0 && id < chunk->GetTotalNumVoxels())
            {
                const chisel::DistVoxel& voxel = chunk->voxels;
                if(voxel.GetWeight(id) > 1e-12)
                {
                    *dist = voxel.GetSDF(id);
                    return true;
                }
            }
            return false;
        }
        else
        {
//            std::cout << "warning! chunks do not exist for isosurface!" << std::endl;
            return false;
        }
    }


    //trilinear interpolation
#if 0
    Vec3 ChunkManager::InterpolateColor(const Vec3& colorPos)
    {
        const float& x = colorPos(0);
        const float& y = colorPos(1);
        const float& z = colorPos(2);
        const int x_0 = static_cast<int>(std::floor(x / voxelResolutionMeters));
        const int y_0 = static_cast<int>(std::floor(y / voxelResolutionMeters));
        const int z_0 = static_cast<int>(std::floor(z / voxelResolutionMeters));
        const int x_1 = x_0 + 1;
        const int y_1 = y_0 + 1;
        const int z_1 = z_0 + 1;

        const ColorVoxel* v_000 = GetColorVoxel(Vec3(x_0, y_0, z_0));
        float red, green, blue = 0.0f;

        const ColorVoxel* v_001 = GetColorVoxel(Vec3(x_0, y_0, z_1));
        const ColorVoxel* v_011 = GetColorVoxel(Vec3(x_0, y_1, z_1));
        const ColorVoxel* v_111 = GetColorVoxel(Vec3(x_1, y_1, z_1));
        const ColorVoxel* v_110 = GetColorVoxel(Vec3(x_1, y_1, z_0));
        const ColorVoxel* v_100 = GetColorVoxel(Vec3(x_1, y_0, z_0));
        const ColorVoxel* v_010 = GetColorVoxel(Vec3(x_0, y_1, z_0));
        const ColorVoxel* v_101 = GetColorVoxel(Vec3(x_1, y_0, z_1));

        if(!v_000 || !v_001 || !v_011 || !v_111 || !v_110 || !v_100 || !v_010 || !v_101)
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

        float xd = (x - x_0) / (x_1 - x_0);
        float yd = (y - y_0) / (y_1 - y_0);
        float zd = (z - z_0) / (z_1 - z_0);
        {
            float c_00 = v_000->GetRed() * (1 - xd) + v_100->GetRed() * xd;
            float c_10 = v_010->GetRed() * (1 - xd) + v_110->GetRed() * xd;
            float c_01 = v_001->GetRed() * (1 - xd) + v_101->GetRed() * xd;
            float c_11 = v_011->GetRed() * (1 - xd) + v_111->GetRed() * xd;
            float c_0 = c_00 * (1 - yd) + c_10 * yd;
            float c_1 = c_01 * (1 - yd) + c_11 * yd;
            float c = c_0 * (1 - zd) + c_1 * zd;
            red = c / 255.0f;
        }
        {
            float c_00 = v_000->GetGreen() * (1 - xd) + v_100->GetGreen() * xd;
            float c_10 = v_010->GetGreen() * (1 - xd) + v_110->GetGreen() * xd;
            float c_01 = v_001->GetGreen() * (1 - xd) + v_101->GetGreen() * xd;
            float c_11 = v_011->GetGreen() * (1 - xd) + v_111->GetGreen() * xd;
            float c_0 = c_00 * (1 - yd) + c_10 * yd;
            float c_1 = c_01 * (1 - yd) + c_11 * yd;
            float c = c_0 * (1 - zd) + c_1 * zd;
            green = c / 255.0f;
        }
        {
            float c_00 = v_000->GetBlue() * (1 - xd) + v_100->GetBlue()  * xd;
            float c_10 = v_010->GetBlue() * (1 - xd) + v_110->GetBlue()  * xd;
            float c_01 = v_001->GetBlue() * (1 - xd) + v_101->GetBlue()  * xd;
            float c_11 = v_011->GetBlue() * (1 - xd) + v_111->GetBlue()  * xd;
            float c_0 = c_00 * (1 - yd) + c_10 * yd;
            float c_1 = c_01 * (1 - yd) + c_11 * yd;
            float c = c_0 * (1 - zd) + c_1 * zd;
            blue = c / 255.0f;
         }

        return Vec3(red, green, blue);
    }
#endif
#if 0
    const DistVoxel* ChunkManager::GetDistanceVoxel(const Vec3& pos)
    {
        ChunkPtr chunk = GetChunkAt(pos);

        if(chunk.get())
        {
            Vec3 rel = (pos - chunk->GetOrigin());
            return &(chunk->GetDistVoxel(chunk->GetVoxelID(rel)));
        }
        else return nullptr;
    }


    const ColorVoxel* ChunkManager::GetColorVoxel(const Vec3& pos)
    {
        ChunkPtr chunk = GetChunkAt(pos);

        if(chunk.get())
        {
            Vec3 rel = (pos - chunk->GetOrigin());
            const VoxelID& id = chunk->GetVoxelID(rel);
            if (id >= 0 && id < chunk->GetTotalNumVoxels())
            {
                return &(chunk->GetColorVoxel(id));
            }
            else
            {
                return nullptr;
            }
        }
        else return nullptr;
    }
#endif



    void ChunkManager::ComputeNormalsFromGradients(Mesh* mesh)
    {
        assert(mesh != nullptr);
        if(mesh->vertices.size() < 3)
        {
            return;
        }
        double dist;
        Vec3 grad;
        for (size_t i = 0; i < mesh->vertices.size(); i+=3)
        {
#if 0

            if(GetSDFAndGradient(mesh->vertices.at(i), &dist, &grad))
            {
                float mag = grad.norm();
                if(mag> 1e-12)
                {
                    grad.normalize();
                    mesh->normals[i+0] = grad;
                    mesh->normals[i+1] = grad;
                    mesh->normals[i+2] = grad;
                }
            }
#else

            int reset_flag = 0;
            for(int j = 0; j < 3; j++)
            {
                if(GetSDFAndGradient(mesh->vertices.at(i+j), grad))
                {
                    float mag = grad.norm();
                    float normal_threshold = voxelResolutionMeters * 6;
                    if(mag> 1e-12 && mag < normal_threshold)
                    {
                        grad.normalize();
                        mesh->normals[i+j] = grad;
                        mesh->averageNoraml += grad;
                        double weight;
                        double sdf;
                        GetWeight(mesh->vertices.at(i+j), &weight);
                        float threshold = 1000;
                        weight = weight > threshold ? threshold : weight;
                        mesh->colors[i+j] = Vec3(weight / threshold, weight / threshold, weight / threshold);

                        if(weight < 200)
                        {
                            reset_flag = 1;
                        }
//                        GetSDF(mesh->vertices.at(i+j), &sdf);
//                        if(weight < 100 || fabs(sdf) > 0.5)
//                        {
//                            reset_flag = 1;
//                        }
                    }
                    else
                    {
                        reset_flag = 1;
                    }
                }
                else
                {
                    reset_flag = 1;
                }
            }
            if(reset_flag)
            {
                mesh->vertices[i+0] = Vec3(0,0,0);
                mesh->vertices[i+1] = Vec3(0,0,0);
                mesh->vertices[i+2] = Vec3(0,0,0);
            }
#endif


        }
        if(mesh->vertices.size() > 0)
        {
            mesh->averageNoraml /= mesh->vertices.size();
            mesh->averageNoraml.normalize();
        }
    }

    void ChunkManager::ColorizeMesh(Mesh* mesh)
    {
        assert(mesh != nullptr);

        mesh->colors.clear();
        mesh->colors.resize(mesh->vertices.size());
        for (size_t i = 0; i < mesh->vertices.size(); i++)
        {
            const Vec3& vertex = mesh->vertices.at(i);
            mesh->colors[i] = InterpolateColorNearest(vertex);
        }
    }


    void ChunkManager::PrintMemoryStatistics()
    {
        float bigFloat = std::numeric_limits<float>::max();

        chisel::AABB totalBounds;
        totalBounds.min = chisel::Vec3(bigFloat, bigFloat, bigFloat);
        totalBounds.max = chisel::Vec3(-bigFloat, -bigFloat, -bigFloat);

        ChunkStatistics stats;
        stats.numKnownInside = 0;
        stats.numKnownOutside = 0;
        stats.numUnknown = 0;
        stats.totalWeight = 0.0f;
        for (const std::pair<ChunkID, ChunkPtr>& chunk : chunks)
        {
            AABB bounds = chunk.second->ComputeBoundingBox();
            for (int i = 0; i < 3; i++)
            {
                totalBounds.min(i) = std::min(totalBounds.min(i), bounds.min(i));
                totalBounds.max(i) = std::max(totalBounds.max(i), bounds.max(i));
            }

            chunk.second->ComputeStatistics(&stats);
        }


        Vec3 ext = totalBounds.GetExtents();
        Vec3 numVoxels = ext * 2 / voxelResolutionMeters;
        float totalNum = numVoxels(0) * numVoxels(1) * numVoxels(2);

        float maxMemory = totalNum * sizeof(DistVoxel) / 1000000.0f;

        size_t currentNum = chunks.size() * (chunkSize(0) * chunkSize(1) * chunkSize(2));
        float currentMemory = currentNum * sizeof(DistVoxel) / 1000000.0f;

        printf("Num Unknown: %lu, Num KnownIn: %lu, Num KnownOut: %lu Weight: %f\n", stats.numUnknown, stats.numKnownInside, stats.numKnownOutside, stats.totalWeight);
        printf("Bounds: %f %f %f %f %f %f\n", totalBounds.min.x(), totalBounds.min.y(), totalBounds.min.z(), totalBounds.max.x(), totalBounds.max.y(), totalBounds.max.z());
        printf("Theoretical max (MB): %f, Current (MB): %f\n", maxMemory, currentMemory);

    }

} // namespace chisel 

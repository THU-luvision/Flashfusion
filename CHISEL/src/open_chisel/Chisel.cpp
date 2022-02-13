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

#include <open_chisel/Chisel.h>

#include <open_chisel/io/PLY.h>

#include <open_chisel/geometry/Raycast.h>

#include <iostream>
#include <fstream>
namespace chisel
{

    Chisel::Chisel()
    {
        // TODO Auto-generated constructor stub
    }

    Chisel::Chisel(const Eigen::Vector3i& chunkSize, float voxelResolution, bool useColor) :
            chunkManager(chunkSize, voxelResolution, useColor)
    {
    }

    Chisel::~Chisel()
    {
        // TODO Auto-generated destructor stub
    }

    void Chisel::Reset()
    {
        chunkManager.Reset();
        meshesToUpdate.clear();
    }

    void Chisel::bufferIntegratorSIMDCentroids(ProjectionIntegrator& integrator,const Transform& depthExtrinsic)
    {
        //set updated centroids
        Vec3 halfVoxel = Vec3(chunkManager.GetResolution(), chunkManager.GetResolution(), chunkManager.GetResolution()) * 0.5f;
        Vec3List diff_centroids;
        diff_centroids.resize(static_cast<size_t>(chunkManager.GetChunkSize()(0) * chunkManager.GetChunkSize()(1) * chunkManager.GetChunkSize()(2)));
        int i = 0;
        for (int z = 0; z < chunkManager.GetChunkSize()(2); z++)
        {
            for(int y = 0; y < chunkManager.GetChunkSize()(1); y++)
            {
                for(int x = 0; x < chunkManager.GetChunkSize()(0); x++)
                {
                    diff_centroids[i] = depthExtrinsic.linear().transpose() *Vec3(x, y, z) * chunkManager.GetResolution() + halfVoxel;
                    i++;
                }
            }
        }

        integrator.diff_centroids = diff_centroids;

        int NumVoxels = chunkManager.GetChunkSize()(2) * chunkManager.GetChunkSize()(0) * chunkManager.GetChunkSize()(0);


        float data0[8],data1[8],data2[8];

        for (int z = 0; z <  chunkManager.GetChunkSize()(2); z++)
        {
            for(int y = 0; y <  chunkManager.GetChunkSize()(1); y++)
            {
                for(int x = 0; x <  chunkManager.GetChunkSize()(0); x+=8)
                {

                    int pos = z * chunkManager.GetChunkSize()(0) * chunkManager.GetChunkSize()(1)
                            + y * chunkManager.GetChunkSize()(0) + x;
                    // 8 * float3 vectors are converted to 3 * float8 vectors
                    //(f00 f01 f02 f10 f11 f12) to (f00 f10 f20 f30 ...)
                    for(int a = 0; a < 8; a++)
                    {
                        int local_pos = a;
                        data0[local_pos] = integrator.diff_centroids[pos + local_pos](0);
                        data1[local_pos] = integrator.diff_centroids[pos + local_pos](1);
                        data2[local_pos] = integrator.diff_centroids[pos + local_pos](2);
                    }

#if 1
                    int centroids_pos_simd = z * chunkManager.GetChunkSize()(0) * chunkManager.GetChunkSize()(1)
                            + y * chunkManager.GetChunkSize()(0) + x;
                    centroids_pos_simd /= 8;
                    integrator.centroids_simd0[centroids_pos_simd] = _mm256_loadu_ps(data0);
                    integrator.centroids_simd1[centroids_pos_simd] = _mm256_loadu_ps(data1);
                    integrator.centroids_simd2[centroids_pos_simd] = _mm256_loadu_ps(data2);
#endif

                }
            }
        }
    }
    int Chisel::GetFullMeshes(float *vertices)
    {


        size_t v = 0;
        float *cur_vert;
        for (const std::pair<ChunkID, MeshPtr>& it : chunkManager.GetAllMeshes())
        {
            int anchorFrameID = chunkManager.GetChunk(it.first)->GetReferenceFrameIndex();
            if(it.second->vertices.size() != it.second->colors.size() || it.second->vertices.size() != it.second->normals.size())
            {
                std::cout << "mesh vertex error!" <<std::endl;
                while(1)
                {
                }
            }
            for (int i =0; i < it.second->vertices.size();i+=3)
            {
//                if(color(0) + color(1) + color(2) < 1e-12)
//                {
//                    continue;
//                }

                for(int j = 0; j < 3; j++)
                {
                    const Vec3& vert = it.second->vertices[i+j];
                    const Vec3& color = it.second->colors[i+j];
                    const Vec3& normal = it.second->normals[i+j];

                    cur_vert = &vertices[12*v];
                    cur_vert[0] = vert(0);
                    cur_vert[1] = vert(1);
                    cur_vert[2] = vert(2);
                    cur_vert[3] = 50;


                    int rgb_value = (color(0) * 255);
                    rgb_value = (rgb_value << 8) + int(color(1) * 255);
                    rgb_value = (rgb_value << 8) + int(color(2) * 255);
                    cur_vert[4] = rgb_value;
                    cur_vert[5] = anchorFrameID;
                    cur_vert[6] = anchorFrameID;
                    cur_vert[7] = anchorFrameID;

                    cur_vert[8] = normal(0);
                    cur_vert[9] = normal(1);
                    cur_vert[10] = normal(2);
                    cur_vert[11] = 5;
                    v++;
                }

            }
        }
        return v;
    }

    bool Chisel::SaveAllMeshesToPLY(const std::string& filename)
    {
        printf("Saving all meshes to PLY file...\n");

        chisel::MeshPtr fullMesh(new chisel::Mesh());

        size_t v = 0;
        for (const std::pair<ChunkID, MeshPtr>& it : chunkManager.GetAllMeshes())
        {
            for (const Vec3& vert : it.second->vertices)
            {
                fullMesh->vertices.push_back(vert);
                fullMesh->indices.push_back(v);
                v++;
            }

            for (const Vec3& color : it.second->colors)
            {
                fullMesh->colors.push_back(color);
            }

            for (const Vec3& normal : it.second->normals)
            {
                fullMesh->normals.push_back(normal);
            }
        }

        printf("Full mesh has %lu verts\n", v);
        bool success = SaveMeshPLYASCII(filename, fullMesh);

        if (!success)
        {
            printf("Saving failed!\n");
        }

        return success;
    }

    bool Chisel::SaveTSDFFiles(const std::string& fileName)
    {
        printf("Saving tsdf to PLY file...\n");
        chisel::MeshPtr fullMesh(new chisel::Mesh());
        Vec3List V;
        Point3List C;
        Vec3List N;
        size_t v = 0;

        int X = chunkManager.GetChunkSize()(0);
        int Y = chunkManager.GetChunkSize()(1);
        int Z = chunkManager.GetChunkSize()(2);
        const ChunkMap& chunks = chunkManager.GetChunks();

        Vec3 halfVoxel = Vec3(chunkManager.GetResolution(), chunkManager.GetResolution(), chunkManager.GetResolution()) * 0.5f;
        for(const std::pair<ChunkID, ChunkPtr>& chunk:chunks)
        {
            ChunkPtr cPtr = chunk.second;
            const DistVoxel &voxels = cPtr->voxels;
            Vec3 ori = cPtr->GetOrigin();
            for (int z = 0; z < chunkManager.GetChunkSize()(2); z++)
            {
                for(int y = 0; y < chunkManager.GetChunkSize()(1); y++)
                {
                    for(int x = 0; x < chunkManager.GetChunkSize()(0); x++)
                    {
                        Vec3 pos = ori  + Vec3(x, y, z) * chunkManager.GetResolution() + halfVoxel;
                        int voxelIndex = x  + y * X + z * X * Y;


                        Point3 color = Point3(cPtr->colors.GetBlue(voxelIndex) * 255,
                                              cPtr->colors.GetGreen(voxelIndex) * 255,
                                              cPtr->colors.GetRed(voxelIndex) * 255) ;

                        Vec3 normal = Vec3((voxels.sdf[voxelIndex] + 0.2) * 30, voxels.weight[voxelIndex],voxels.sdf[voxelIndex] * 100 );

                        V.push_back(pos);
                        C.push_back(color);
                        N.push_back(normal);
                    }
                }
            }
        }

        std::ofstream output_file(fileName.c_str(), std::ios::out | std::ios::trunc);
        int pointNum = fmin(V.size(), C.size());
        output_file << "ply" << std::endl;
        output_file << "format ascii 1.0           { ascii/binary, format version number }" << std::endl;
        output_file << "comment made by Greg Turk  { comments keyword specified, like all lines }" << std::endl;
        output_file << "comment this file is a cube" << std::endl;
        output_file << "element vertex " << pointNum << "           { define \"vertex\" element, 8 of them in file }" << std::endl;
        output_file << "property float x" << std::endl;
        output_file << "property float y" << std::endl;
        output_file << "property float z" << std::endl;
        output_file << "property float nx" << std::endl;
        output_file << "property float ny" << std::endl;
        output_file << "property float nz" << std::endl;
        output_file << "property float intensity" << std::endl;
        output_file << "property uchar red" << std::endl;
        output_file << "property uchar green" << std::endl;
        output_file << "property uchar blue" << std::endl;

        output_file << "end_header" << std::endl;
        for (int i = 0; i < V.size(); i++)
        {
          output_file << V[i](0) << " " << V[i](1) << " " << V[i](2) << " "
                      << N[i](0) << " " << N[i](1) << " " << N[i](2) << " " << 1 << " "
            << C[i](0) << " " << C[i](1)  << " " << C[i](2) << " " << std::endl;
        }
        output_file.close();
        printf("Full tsdf has %lu verts\n", V.size());

        while(1)
        {

        }
    }



} // namespace chisel 

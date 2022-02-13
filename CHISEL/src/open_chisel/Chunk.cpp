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

#include <open_chisel/Chunk.h>

namespace chisel
{

    Chunk::Chunk() :
            voxelResolutionMeters(0)
    {
        // TODO Auto-generated constructor stub

        frameIndex = -1;
        p_normal == NULL;
        p_color == NULL;

    }

    Chunk::Chunk(const ChunkID id, const Eigen::Vector3i& nv, float r, bool useColor) :
            ID(id), numVoxels(nv), voxelResolutionMeters(r)
    {
        AllocateDistVoxels();

        if(useColor)
        {
            AllocateColorVoxels();
        }

        frameIndex = -1;
        p_normal == NULL;
        p_color == NULL;

        origin = Vec3(numVoxels(0) * ID(0) * voxelResolutionMeters, numVoxels(1) * ID(1) * voxelResolutionMeters, numVoxels(2) * ID(2) * voxelResolutionMeters);
    }

    Chunk::~Chunk()
    {

    }

    void Chunk::AllocateDistVoxels()
    {
        int totalNum = GetTotalNumVoxels();
        voxels.clear();
        voxels.sdf.assign(totalNum,999.0f);
        voxels.weight.assign(totalNum,0.0f);

//        voxels.resize(totalNum, DistVoxel());
    }

    void Chunk::AllocateColorVoxels()
    {
        int totalNum = GetTotalNumVoxels();

        memset(colors.colorData,0,totalNum * 4);
//        colors.colorData.assign(totalNum * 4, 0);
   //     colors.resize(totalNum, ColorVoxel());
    }

    AABB Chunk::ComputeBoundingBox()
    {
        Vec3 pos = origin;
        Vec3 size = numVoxels.cast<float>() * voxelResolutionMeters;
        return AABB(pos, pos + size);
    }

    Point3 Chunk::GetVoxelCoords(const Vec3& worldCoords) const
    {
        const float roundingFactorX = 1.0f / (voxelResolutionMeters);
        const float roundingFactorY = 1.0f / (voxelResolutionMeters);
        const float roundingFactorZ = 1.0f / (voxelResolutionMeters);

        return Point3( static_cast<int>(std::floor(worldCoords(0) * roundingFactorX)),
                       static_cast<int>(std::floor(worldCoords(1) * roundingFactorY)),
                       static_cast<int>(std::floor(worldCoords(2) * roundingFactorZ)));
    }

    VoxelID Chunk::GetVoxelID(const Vec3& relativePos) const
    {
        return GetVoxelID(GetVoxelCoords(relativePos));
    }

    VoxelID Chunk::GetLocalVoxelIDFromGlobal(const Point3& worldPoint) const
    {
        return GetVoxelID(GetLocalCoordsFromGlobal(worldPoint));
    }

    Point3 Chunk::GetLocalCoordsFromGlobal(const Point3& worldPoint) const
    {
        return (worldPoint - Point3(ID.x() * numVoxels.x(), ID.y() * numVoxels.y(), ID.z() * numVoxels.z()));
    }


    void Chunk::ComputeStatistics(ChunkStatistics* stats)
    {
        assert(stats != nullptr);
        int voxelsNum = voxels.sdf.size();
        for(int i = 0; i < voxelsNum; i++)
        {
            float weight = voxels.GetWeight(i);
            if (weight > 0)
            {
                float sdf = voxels.GetSDF(i);
                if (sdf < 0)
                {
                    stats->numKnownInside++;
                }
                else
                {
                    stats->numKnownOutside++;
                }
            }
            else
            {
                stats->numUnknown++;
            }

            stats->totalWeight += weight;

        }
    }

    Vec3 Chunk::GetColorAt(const Vec3& pos)
    {
        if(ComputeBoundingBox().Contains(pos))
        {
            Vec3 chunkPos = (pos - origin) / voxelResolutionMeters;
            int chunkX = static_cast<int>(chunkPos(0));
            int chunkY = static_cast<int>(chunkPos(1));
            int chunkZ = static_cast<int>(chunkPos(2));

            if(IsCoordValid(chunkX, chunkY, chunkZ))
            {
                VoxelID voxelIndex = GetVoxelID(chunkX,chunkY,chunkZ);
                return Vec3((colors.GetRed(voxelIndex)),
                            (colors.GetGreen(voxelIndex)),
                            (colors.GetBlue(voxelIndex)));
            }
        }

        return Vec3::Zero();
    }


} // namespace chisel 

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

#ifndef CHUNK_H_
#define CHUNK_H_

#include <memory>
#include <vector>
#include <Eigen/Core>

#include <open_chisel/geometry/AABB.h>
#include "DistVoxel.h"
#include "ColorVoxel.h"

namespace chisel
{

    typedef Eigen::Vector3i ChunkID;
    typedef int VoxelID;

    struct ChunkStatistics
    {
            size_t numKnownInside;
            size_t numKnownOutside;
            size_t numUnknown;
            float totalWeight;
    };

    class Chunk
    {
        public:

            Chunk();
            Chunk(const ChunkID id, const Eigen::Vector3i& numVoxels, float resolution, bool useColor);
            virtual ~Chunk();


            void resetChunk()
            {
                AllocateColorVoxels();
                AllocateDistVoxels();
            }

            void AllocateDistVoxels();
            void AllocateColorVoxels();

            inline const ChunkID& GetID() const { return ID; }
            inline ChunkID& GetIDMutable() { return ID; }
            inline void SetID(const ChunkID& id) { ID = id; }

            inline bool HasColors() const { return !(colors.colorData == NULL); }
            inline bool HasVoxels() const { return !voxels.sdf.empty(); }
            inline const DistVoxel & GetVoxels() const { return voxels; }

            inline const Eigen::Vector3i& GetNumVoxels() const { return numVoxels; }
            inline float GetVoxelResolutionMeters() const { return voxelResolutionMeters; }

//            inline const DistVoxel& GetDistVoxel(const VoxelID& voxelID) const { return voxels.at(voxelID); }
//            inline DistVoxel& GetDistVoxelMutable(const VoxelID& voxelID) { return voxels.at(voxelID); }
//            inline const ColorVoxel& GetColorVoxel(const VoxelID& voxelID) const { return colors.at(voxelID); }
//            inline ColorVoxel& GetColorVoxelMutable(const VoxelID& voxelID) { return colors.at(voxelID); }

            Point3 GetVoxelCoords(const Vec3& worldCoords) const;

            inline VoxelID GetVoxelID(const Point3& coords) const
            {
                return GetVoxelID(coords.x(), coords.y(), coords.z());
            }

            inline VoxelID GetVoxelID(int x, int y, int z) const
            {
                return (z * numVoxels(1) + y) * numVoxels(0) + x;
            }

//            inline const DistVoxel& GetDistVoxel(int x, int y, int z) const
//            {
//                return GetDistVoxel(GetVoxelID(x, y, z));
//            }

//            inline DistVoxel& GetDistVoxelMutable(int x, int y, int z)
//            {
//                return GetDistVoxelMutable(GetVoxelID(x, y, z));
//            }

//            inline const ColorVoxel& GetColorVoxel(int x, int y, int z) const
//            {
//                return GetColorVoxel(GetVoxelID(x, y, z));
//            }

//            inline ColorVoxel& GetColorVoxelMutable(int x, int y, int z)
//            {
//                return GetColorVoxelMutable(GetVoxelID(x, y, z));
//            }

            inline bool IsCoordValid(VoxelID idx) const
            {
                return idx >= 0 && idx < voxels.sdf.size();
            }

            inline bool IsCoordValid(int x, int y, int z) const
            {
                return (x >= 0 && x < numVoxels(0) && y >= 0 && y < numVoxels(1) && z >= 0 && z < numVoxels(2));
            }


            inline size_t GetTotalNumVoxels() const
            {
                return numVoxels(0) * numVoxels(1) * numVoxels(2);
            }


            void ComputeStatistics(ChunkStatistics* stats);

            AABB ComputeBoundingBox();

            inline const Vec3& GetOrigin() { return origin; }

            Vec3 GetColorAt(const Vec3& pos);

            VoxelID GetVoxelID(const Vec3& relativePos) const;
            VoxelID GetLocalVoxelIDFromGlobal(const Point3& worldPoint) const;
            Point3 GetLocalCoordsFromGlobal(const Point3& worldPoint) const;


            EIGEN_MAKE_ALIGNED_OPERATOR_NEW




            void UpdateReferenceFrame(const int keyframeIndex,
                                   const Mat4x4 inputPos,
                                   const float *normalPointer,
                                   const unsigned char * colorPointer,
                                   const float *depthPointer)
            {
                frameIndex = keyframeIndex;
                pos = inputPos;
                p_normal = normalPointer;
                p_color = colorPointer;
                p_depth = depthPointer;
            }


            Mat4x4 GetReferenceFramePose()
            {
                return pos;
            }

            void GetReferenceFrame(int &keyframeIndex,
                                   Mat4x4 &inputPos,
                                   const float * &normalPointer,
                                   const unsigned char * &colorPointer,
                                   const float * &depthPointer)
            {
                keyframeIndex = frameIndex;
                inputPos = pos;
                normalPointer = p_normal;
                colorPointer = p_color;
                depthPointer = p_depth;
            }

            int GetReferenceFrameIndex() {return frameIndex;}

            DistVoxel voxels;
            ColorVoxel colors;


        protected:
            ChunkID ID;
            Eigen::Vector3i numVoxels;
            float voxelResolutionMeters;
            Vec3 origin;


            int frameIndex;
            Eigen::Matrix4f pos;
            const float * p_normal;
            const unsigned char * p_color;
            const float * p_depth;

    };

    typedef std::shared_ptr<Chunk> ChunkPtr;
    typedef std::shared_ptr<const Chunk> ChunkConstPtr;
    typedef std::vector<ChunkID, Eigen::aligned_allocator<ChunkID> > ChunkIDList;

} // namespace chisel 

#endif // CHUNK_H_ 

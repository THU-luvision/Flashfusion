#ifndef MOBILEFUSION_H
#define MOBILEFUSION_H



#include "../GCSLAM/frame.h"
#include "../GCSLAM/MultiViewGeometry.h"
#include "../BasicAPI.h"
#include "../GCSLAM/GCSLAM.h"
#include "MapMaintain.hpp"
#include <Eigen/Core>
#include <Eigen/Dense>
#include <sys/stat.h>

#include "../Shaders/Shaders.h"

#include <pangolin/pangolin.h>
#include <pangolin/gl/gl.h>
#include <pangolin/gl/gldraw.h>
#include <GL/freeglut.h>
#include <GL/glut.h>
#include <GL/glew.h>

#include <boost/thread/thread.hpp>
#include <boost/thread/mutex.hpp>
#include <boost/thread/condition.hpp>
#include <boost/filesystem.hpp>
#include <open_chisel/Chisel.h>
#include <open_chisel/ProjectionIntegrator.h>
#include <open_chisel/camera/PinholeCamera.h>
#include <open_chisel/camera/DepthImage.h>
#include <open_chisel/camera/ColorImage.h>
#include <open_chisel/truncation/QuadraticTruncator.h>
#include <open_chisel/weighting/ConstantWeighter.h>

#define MAX_MOBILEFUSION_IMAGE_WIDTH  640
#define MAX_MOBILEFUSION_IMAGE_HEIGHT 480
// 30 M * 12 * 4 = 1.5G
#define GLOBLA_MODLE_VERTEX_NUM (1024 * 1024 * 30)
#define VERTEX_WEIGHT_THRESHOLD 3

#define VOXEL_SIZE 10

typedef float DepthData;
typedef uint8_t ColorData;


typedef std::vector<Eigen::Matrix4f, Eigen::aligned_allocator<Eigen::Matrix4f> > Mat4List;
typedef std::vector<Eigen::Matrix3f, Eigen::aligned_allocator<Eigen::Matrix3f> > Mat3List;
typedef std::vector<Eigen::Vector3f, Eigen::aligned_allocator<Eigen::Vector3f> > Vec3List;

#define INTEGRATE_ALL 1
#define FUSE_COLOR  0

struct VertexElement
{
    float loc[4];
    float color[4];
    float normal[4];
};

class SLAMSystem
{

};

class MobileFusion
{

public:

    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    GLuint vbo;
    GLuint vbo_point_cloud;
    GLuint vbo_data_transfer;
    GLuint feedback_vbo;
    GLuint unstable_vbo;
    std::shared_ptr<Shader> drawProgram;
    std::shared_ptr<Shader> drawPhongLighting;
    std::shared_ptr<Shader> drawVoxelHashingStyle;

    VertexElement *currentObservationBuffer;          //As a simple version, global model vertices are maintained at CPU for simplicity



    float * tsdf_visualization_buffer;
    int tsdf_vertice_num;
    int global_model_vertex_cnt;
    int visualize_vertex_num;
    int valid_vertex_cnt;
    int vertex_data_updated;                        // update vertex data;
    std::vector<int> verticesHashTable;

    Mat4List PrePoseList;

    chisel::ChiselPtr chiselMap;
    std::shared_ptr<chisel::DepthImage<DepthData> > lastDepthImage;
    std::shared_ptr<chisel::ColorImage<ColorData> > lastColorImage;
    chisel::ProjectionIntegrator projectionIntegrator;
    chisel::PinholeCamera cameraModel;

    Mat4List IntegratePoseList;
    //for tsdf fusion
    ChunkIDList chunksIntersecting;
    std::vector<bool> needsUpdateFlag;
    std::vector<bool> newChunkFlag;
    Frame * lastKeyframe;


    boost::mutex update_globalMap_mutex;
    boost::condition_variable update_globalMap_cv;
    int validFrameNum;
    int fuseKeyframeId;


    GCSLAM gcSLAM;

    // clear redudent memory stored in frames, including depth, normal, color, features.
    void clearRedudentFrameMemory(int integrateLocalFrameNum)
    {

        int keyFrameNum = gcSLAM.GetKeyframeDataList().size();
        if(keyFrameNum > 1)
        {
            MultiViewGeometry::KeyFrameDatabase kd = gcSLAM.GetKeyframeDataList()[keyFrameNum - 2];

            float inc = 0;
            for(int k = 0; k < kd.corresponding_frames.size(); k++)
            {
                if(k < inc - 1e-4)
                {
                    gcSLAM.globalFrameList[kd.corresponding_frames[k]].clear_memory();
                    continue;
                }
                inc += kd.corresponding_frames.size() / integrateLocalFrameNum;
                gcSLAM.globalFrameList[kd.corresponding_frames[k]].clear_redudent_memoery();
            }

            gcSLAM.globalFrameList[kd.keyFrameIndex].clear_keyframe_memory();
        }
    }

    void updateGlobalMap(int inputValidFrameNum, int inputFuseKeyframeId)
    {
        validFrameNum = inputValidFrameNum;
        fuseKeyframeId = inputFuseKeyframeId;
//        boost::unique_lock<boost::mutex> lock(update_globalMap_mutex);
        update_globalMap_cv.notify_one();
    }

    inline void MapManagement()
    {
        while(1)
        {
            boost::unique_lock<boost::mutex> lock(update_globalMap_mutex);
            update_globalMap_cv.wait(lock);
            TICK("MobileFusion::TSDFFusion");
            tsdfFusion(gcSLAM.globalFrameList,
                       fuseKeyframeId,
                       gcSLAM.GetKeyframeDataList(),
                       gcSLAM.GetKeyframeDataList().size() - 2);
            TOCK("MobileFusion::TSDFFusion");
            Stopwatch::getInstance().printAll();
        }

    }


    void ReIntegrateKeyframe(std::vector<Frame> &frame_list,const MultiViewGeometry::KeyFrameDatabase &kfDatabase, const int integrateFlag)
    {
        Frame &kf = frame_list[kfDatabase.keyFrameIndex];

        int totalPixelNum =  cameraModel.GetWidth() * cameraModel.GetHeight();
        float cx = cameraModel.GetCx();
        float cy = cameraModel.GetCy();
        float fx = cameraModel.GetFx();
        float fy = cameraModel.GetFy();
        int width = cameraModel.GetWidth();
        int height = cameraModel.GetHeight();
        chisel::Transform lastPose;
        ChunkIDList localChunksIntersecting;
        std::vector<void  *> localChunksPtr;
        std::vector<bool> localNeedsUpdateFlag;
        std::vector<bool> localNewChunkFlag;
        if(integrateFlag == 1)
        {
            lastPose = kf.pose_sophus[0].matrix().cast<float>();
            kf.pose_sophus[1] = kf.pose_sophus[0];
        }
        else if(integrateFlag == 0)
        {

            lastPose = kf.pose_sophus[1].matrix().cast<float>();
            localChunksIntersecting = kf.validChunks;
            localChunksPtr = kf.validChunksPtr;
            for(int i = 0; i < localChunksIntersecting.size(); i++)
            {
                localNeedsUpdateFlag.push_back(true);
                localNewChunkFlag.push_back(false);
            }
        }
        float *depthImageData;
        static unsigned char *colorImageData = new unsigned char[totalPixelNum * 4];

        unsigned char *colorValid = (unsigned char *)kf.colorValidFlag.data;
        depthImageData = (float *) kf.refined_depth.data;
#if 1
        for(int i = 0; i < height; i++)
        {
            for(int j = 0; j < width ; j++)
            {
                int pos = i * width + j;
                colorImageData[pos*4 + 0] = kf.rgb.at<unsigned char>(pos*3+0);
                colorImageData[pos*4 + 1] = kf.rgb.at<unsigned char>(pos*3+1);
                colorImageData[pos*4 + 2] = kf.rgb.at<unsigned char>(pos*3+2);
                colorImageData[pos*4 + 3] = 1;
                if(!colorValid[pos])
                {
                    colorImageData[pos*4 + 0] = 0;
                    colorImageData[pos*4 + 1] = 0;
                    colorImageData[pos*4 + 2] = 0;
                    colorImageData[pos*4 + 3] = 0;
                }
            }

        }
#endif


        if(integrateFlag == 1)
        {
//            std::cout << "PrepareIntersectChunks" << std::endl;
            TICK("CHISEL::Reintegration::1::prepareIntersectChunks");
            chiselMap->PrepareIntersectChunks(projectionIntegrator,
                                              depthImageData,
                                              lastPose,
                                              cameraModel,
                                              localChunksIntersecting,
                                              localNeedsUpdateFlag,
                                              localNewChunkFlag);
    //        chiselMap->GetSearchRegion(searchArea,cameraModel,lastPose);
            TOCK("CHISEL::Reintegration::1::prepareIntersectChunks");
        }


//        std::cout << "IntegrateDepthScanColor" << std::endl;
        TICK("CHISEL::Reintegration::2::IntegrateKeyDepthAndColor");

        chiselMap->IntegrateDepthScanColor(projectionIntegrator,
                                           depthImageData,
                                           colorImageData,
                                           lastPose,
                                           cameraModel,
                                           localChunksIntersecting,
                                           localNeedsUpdateFlag,
                                           integrateFlag);
        TOCK("CHISEL::Reintegration::2::IntegrateKeyDepthAndColor");

#if INTEGRATE_ALL
        TICK("CHISEL::Reintegration::3::IntegrateLocalDepth");

        // only integrate ten frames evenly distributed in this keyframe

        for(int i = 0; i < kfDatabase.corresponding_frames.size(); i++)
        {
            Frame & local_frame = frame_list[kfDatabase.corresponding_frames[i]];
            if(local_frame.refined_depth.empty())
            {
                continue;
            }
//            printf("integrating frame: %d %d\r\n",local_frame.frame_index, integrateFlag);
            if(integrateFlag == 1)
            {
                lastPose = local_frame.pose_sophus[0].matrix().cast<float>();
                local_frame.pose_sophus[1] = local_frame.pose_sophus[0];
            }
            else if(integrateFlag == 0)
            {

                lastPose = local_frame.pose_sophus[1].matrix().cast<float>();

            }
            depthImageData = (float *) local_frame.refined_depth.data;


            chiselMap->IntegrateDepthScanColor(projectionIntegrator,
                                               depthImageData,
                                               NULL,
                                               lastPose,
                                               cameraModel,
                                               localChunksIntersecting,
                                               localNeedsUpdateFlag,
                                               integrateFlag);

        }
        TOCK("CHISEL::Reintegration::3::IntegrateLocalDepth");
#endif



        TICK("CHISEL::Reintegration::4::FinalizeIntegrateChunks");
        if(integrateFlag == 1)
        {
            chiselMap->FinalizeIntegrateChunks(localChunksIntersecting,localNeedsUpdateFlag,localNewChunkFlag,kf.validChunks);
        }
        else if(integrateFlag == 0)
        {

            std::vector<void *> localChunksPtrValid;
            ChunkIDList localValidChunks;
            chiselMap->FinalizeIntegrateChunks(localChunksIntersecting,localNeedsUpdateFlag,localNewChunkFlag,localValidChunks);
            kf.validChunks.clear();
        }
        TOCK("CHISEL::Reintegration::4::FinalizeIntegrateChunks");

    }

    MobileFusion()
    {
        int argc = 1;
        char ProjectName[256] = "MobileFusion";
        char *argv = ProjectName;
        glutInit(&argc, &argv);
        glutInitDisplayMode(GLUT_SINGLE);
        GLenum err=glewInit();
        if(err!=GLEW_OK) {
          // Problem: glewInit failed, something is seriously wrong.
          std::cout << "glewInit failed: " << glewGetErrorString(err) << std::endl;
          exit(1);
        }

        tsdf_vertice_num = 0;
        validFrameNum = 0;
        fuseKeyframeId = 0;
        tsdf_visualization_buffer = new float[GLOBLA_MODLE_VERTEX_NUM*12];
        memset(tsdf_visualization_buffer,0,GLOBLA_MODLE_VERTEX_NUM*12 * sizeof(float));
        global_model_vertex_cnt = 0;
        vertex_data_updated = 0;
        visualize_vertex_num = 0;

        glGenBuffers(1, &vbo);
        glBindBuffer(GL_ARRAY_BUFFER,vbo);
        glBufferData(GL_ARRAY_BUFFER, GLOBLA_MODLE_VERTEX_NUM * sizeof(VertexElement), &tsdf_visualization_buffer[0], GL_DYNAMIC_DRAW);
        glBindBuffer(GL_ARRAY_BUFFER, 0);

        glGenBuffers(1, &vbo_data_transfer);
        glGenBuffers(1, &unstable_vbo);
        glGenTransformFeedbacks(1, &feedback_vbo);
        glGenBuffers(1, &vbo_point_cloud);


        currentObservationBuffer = new VertexElement[MAX_MOBILEFUSION_IMAGE_WIDTH * MAX_MOBILEFUSION_IMAGE_HEIGHT];

        drawVoxelHashingStyle = loadProgramFromFile("draw_feedback_VoxelHashing.vert","draw_feedback_VoxelHashing.frag");

        PrePoseList.clear();


        lastKeyframe = NULL;

        chunksIntersecting.clear();
        needsUpdateFlag.clear();
        newChunkFlag.clear();
        IntegratePoseList.clear();
    }

    void IntegrateFrame(const Frame &frame_ref);

    int tsdfFusion(std::vector<Frame> &frame_list, int CorrKeyframeIndex, const std::vector<MultiViewGeometry::KeyFrameDatabase> &kflist, int integrateKeyframeID);

    void initGCSLAM(const int maximum_frame_num,
                    const MultiViewGeometry::GlobalParameters para,
                    const MultiViewGeometry::CameraPara &camera)
    {
        gcSLAM.init(maximum_frame_num, camera);
        gcSLAM.SetMinimumDisparity(para.minimum_disparity);
        gcSLAM.SetSalientScoreThreshold(para.salient_score_threshold);
        gcSLAM.SetMaxCandidateNum(para.maximum_keyframe_match_num);
    }

    void initChiselMap(const MultiViewGeometry::CameraPara &camera, float ipnutVoxelResolution, float farPlaneDist = 3)
    {
        float fx = camera.c_fx;
        float fy = camera.c_fy;
        float cx = camera.c_cx;
        float cy = camera.c_cy;
        int width = camera.width;
        int height = camera.height;

#if 1
        float truncationDistConst=  0.001504;
        float truncationDistLinear=  0.00152;
        float truncationDistQuad=  0.0019;
        float truncationDistScale=  6.0;
#else
        float truncationDistConst=  0.01;
        float truncationDistLinear=  0.01;
        float truncationDistQuad=  0.01;
        float truncationDistScale=  1.0;
#endif
        float weight=  1;
        bool useCarving=  true;
        float carvingDist=  0.05;
        float nearPlaneDist=  0.01;

        std::cout << "far plane dist: " << farPlaneDist << std::endl;
        chunkSizeX=  8;
        chunkSizeY=  8;
        chunkSizeZ=  8;
        voxelResolution=  ipnutVoxelResolution;
        useColor=  true;

         chisel::Vec4 truncation(truncationDistQuad, truncationDistLinear, truncationDistConst, truncationDistScale);
         chiselMap = chisel::ChiselPtr(new chisel::Chisel(Eigen::Vector3i(chunkSizeX, chunkSizeY, chunkSizeZ), voxelResolution, useColor));

         projectionIntegrator.SetCentroids(chiselMap->GetChunkManager().GetCentroids());
         projectionIntegrator.SetTruncator(chisel::TruncatorPtr(new chisel::QuadraticTruncator(truncation(0), truncation(1), truncation(2), truncation(3))));
         projectionIntegrator.SetWeighter(chisel::WeighterPtr(new chisel::ConstantWeighter(weight)));
         projectionIntegrator.SetCarvingDist(carvingDist);
         projectionIntegrator.SetCarvingEnabled(useCarving);


         cameraModel.SetIntrinsics(fx,fy,cx,cy);
         cameraModel.SetNearPlane(nearPlaneDist);
         cameraModel.SetFarPlane(farPlaneDist);
         cameraModel.SetWidth(width);
         cameraModel.SetHeight(height);

    }


    ~MobileFusion()
    {
        delete tsdf_visualization_buffer;
        delete currentObservationBuffer;
        glDeleteBuffers(1, &vbo);
        glDeleteBuffers(1, &vbo_data_transfer);
        glDeleteBuffers(1, &unstable_vbo);
        glDeleteBuffers(1, &feedback_vbo);
        glDeleteBuffers(1, &vbo_point_cloud);
    }

    void DrawCube(float *vertex_list, GLint *index_list)
    {
        int i,j;

        glBegin(GL_LINES);
        for(i=0; i<12; ++i) // 12 条线段

        {
            for(j=0; j<2; ++j) // 每条线段 2个顶点

            {
//                Eigen::Vector4f vertex(vertex_list[index_list[i*2 + j] * 3],
//                        vertex_list[index_list[i*2 + j] * 3 + 1],
//                        vertex_list[index_list[i*2 + j] * 3 + 2],
//                        1);
//                vertex = t * vertex;
                glVertex3fv(&vertex_list[index_list[i*2 + j] * 3]);
            }
        }
        glEnd();
    }



    void RenderPointCloud(Frame &f,
                          pangolin::OpenGlMatrix mvp,
                          const bool drawNormals,
                          const bool drawColors)
    {
        drawProgram->Bind();

        const Eigen::Matrix4f pose = f.pose_sophus[0].matrix().cast<float>();

        drawProgram->setUniform(Uniform("MVP", mvp));
        drawProgram->setUniform(Uniform("threshold", 5.0f));
        drawProgram->setUniform(Uniform("pose", pose));
        drawProgram->setUniform(Uniform("colorType", (drawNormals ? 1 : drawColors ? 2 : 0)));

        printf("begin draw colors\r\n");

        int width = cameraModel.GetWidth();
        int height = cameraModel.GetHeight();
        float fx = cameraModel.GetFx();
        float fy = cameraModel.GetFy();
        float cx = cameraModel.GetCx();
        float cy = cameraModel.GetCy();

        for(int i = 0; i < height; i++)
        {
            for(int j = 0; j < width; j++)
            {
                int pos = i * width + j;
                float depth = f.refined_depth.at<float>(i,j);
                currentObservationBuffer[pos].loc[0] = (j - cx)/fx * depth;
                currentObservationBuffer[pos].loc[1] = (j - cy)/fy * depth;
                currentObservationBuffer[pos].loc[2] = depth;
                currentObservationBuffer[pos].loc[3] = 50;
                int rgb_value = (f.rgb.at<unsigned char>(pos * 3+ 0) * 255);
                rgb_value = (rgb_value << 8) + int(f.rgb.at<unsigned char>(pos * 3+ 1) * 255);
                rgb_value = (rgb_value << 8) + int(f.rgb.at<unsigned char>(pos * 3+ 2) * 255);
                currentObservationBuffer[pos].color[0] = rgb_value;
                currentObservationBuffer[pos].color[1] = f.frame_index;
                currentObservationBuffer[pos].color[2] = f.frame_index;
                currentObservationBuffer[pos].color[3] = f.frame_index;

                currentObservationBuffer[pos].normal[0] = f.normal_map.at<float>(pos * 3+0);
                currentObservationBuffer[pos].normal[1] = f.normal_map.at<float>(pos * 3+1);
                currentObservationBuffer[pos].normal[2] = f.normal_map.at<float>(pos * 3+2);
                currentObservationBuffer[pos].normal[3] = 5;

            }
        }


        //buffer data;
        glBindBuffer(GL_ARRAY_BUFFER, vbo_point_cloud);
        glBufferData(GL_ARRAY_BUFFER, (width*height) * sizeof(VertexElement), &currentObservationBuffer[0], GL_DYNAMIC_DRAW);
        glBindBuffer(GL_ARRAY_BUFFER, 0);

        glBindBuffer(GL_ARRAY_BUFFER, vbo_point_cloud);

        glEnableVertexAttribArray(0);
        glVertexAttribPointer(0, 4, GL_FLOAT, GL_FALSE, 12 * sizeof(float), 0);

        glEnableVertexAttribArray(1);
        glVertexAttribPointer(1, 4, GL_FLOAT, GL_FALSE, 12 * sizeof(float), reinterpret_cast<GLvoid*>(sizeof(Eigen::Vector4f) * 1));

        glEnableVertexAttribArray(2);
        glVertexAttribPointer(2, 4, GL_FLOAT, GL_FALSE, 12 * sizeof(float), reinterpret_cast<GLvoid*>(sizeof(Eigen::Vector4f) * 2));

        glDrawArrays(GL_POINTS, 0, width*height);

        glDisableVertexAttribArray(0);
        glDisableVertexAttribArray(1);
        glDisableVertexAttribArray(2);
        glBindBuffer(GL_ARRAY_BUFFER, 0);

        drawProgram->Unbind();
    }


    void GetColor(double v, double vmin, double vmax, int &r, int &g, int &b)
    {
        double dv;

        if (v < vmin)
           v = vmin;
        if (v > vmax)
           v = vmax;
        dv = vmax - vmin;

        r = 0;
        g = 0;
        b = 0;
        if (v < (vmin + 0.25 * dv)) {
           r = 0;
           g = (4 * (v - vmin) / dv) * 255;
        } else if (v < (vmin + 0.5 * dv)) {
           r = 0;
           b = (1 + 4 * (vmin + 0.25 * dv - v) / dv) * 255;
        } else if (v < (vmin + 0.75 * dv)) {
           r = (4 * (v - vmin - 0.5 * dv) / dv) * 255;
           b = 0;
        } else {
           g = (1 + 4 * (vmin + 0.75 * dv - v) / dv) * 255;
           b = 0;
        }

    }

    inline int MobileShow(pangolin::OpenGlMatrix mvp,
                   const float threshold,
                   const bool drawUnstable,
                   const bool drawNormals,
                   const bool drawColors,
                   const bool drawPoints,
                   const bool drawWindow,
                   const bool drawTimes,
                   const int time,
                   const int timeDelta,
                   std::vector<Frame> &frame_list)
    {
        std::shared_ptr<Shader> program = drawVoxelHashingStyle;
        program->Bind();                    // set this program as current program
        Eigen::Matrix4f pose = Eigen::Matrix4f::Identity();
        program->setUniform(Uniform("MVP", mvp));
        program->setUniform(Uniform("pose", pose));
        program->setUniform(Uniform("threshold", threshold));
        program->setUniform(Uniform("colorType", (drawNormals ? 1 : drawColors ? 2 : drawTimes ? 3 : 0)));
        program->setUniform(Uniform("unstable", drawUnstable));
        program->setUniform(Uniform("drawWindow", drawWindow));
        program->setUniform(Uniform("time", time));
        program->setUniform(Uniform("timeDelta", timeDelta));


        if(program == drawPhongLighting)
        {
            program->setUniform(Uniform("view_matrix", pose));
            program->setUniform(Uniform("proj_matrix", mvp));
            Eigen::Vector3f Lightla(0.2f, 0.2f, 0.2f);
            Eigen::Vector3f Lightld(1.0f, 1.0f, 1.0f);
            Eigen::Vector3f Lightls(1.0f, 1.0f, 1.0f);
            Eigen::Vector3f Lightldir(0.0, 0.0, 1.0f);
            Eigen::Vector3f fma(0.26f, 0.26f, 0.26f);
            Eigen::Vector3f fmd(0.35f, 0.35f, 0.35f);
            Eigen::Vector3f fms(0.30f, 0.30f, 0.30f);
            float fss = 16.0f;
            Eigen::Vector3f bma(0.85f, 0.85f, 0.85f);
            Eigen::Vector3f bmd(0.85f, 0.85f, 0.85f);
            Eigen::Vector3f bms(0.60f, 0.60f, 0.60f);
            float bss = 16.0f;

            Eigen::Matrix4f user_view_matrix = Eigen::Matrix4f::Identity();
            Eigen::Matrix4f user_light_matrix = Eigen::Matrix4f::Identity();
            Eigen::Vector4f user_rot_center = Eigen::Vector4f(0, 0, 0, 1);
            program->setUniform(Uniform("Lightla", Lightla));
            program->setUniform(Uniform("Lightld", Lightld));
            program->setUniform(Uniform("Lightls", Lightls));
            program->setUniform(Uniform("Lightldir", Lightldir));
            program->setUniform(Uniform("fma", fma));
            program->setUniform(Uniform("fmd", fmd));
            program->setUniform(Uniform("fms", fms));
            program->setUniform(Uniform("bma", bma));
            program->setUniform(Uniform("bmd", bmd));
            program->setUniform(Uniform("bms", bms));
            program->setUniform(Uniform("bss", bss));
            program->setUniform(Uniform("fss", fss));
            program->setUniform(Uniform("user_view_matrix", user_view_matrix));
            program->setUniform(Uniform("user_light_matrix", user_light_matrix));
            program->setUniform(Uniform("user_rot_center", user_rot_center));
        }

        if(program == drawVoxelHashingStyle)
        {
            float s_materialShininess = 16.0f;
            Eigen::Vector4f s_materialAmbient   = Eigen::Vector4f(0.75f, 0.65f, 0.5f, 1.0f);
            Eigen::Vector4f s_materialDiffuse   = Eigen::Vector4f(1.0f, 0.9f, 0.7f, 1.0f);
            Eigen::Vector4f s_materialSpecular  = Eigen::Vector4f(1.0f, 1.0f, 1.0f, 1.0f);
            Eigen::Vector4f s_lightAmbient 	    = Eigen::Vector4f(0.4f, 0.4f, 0.4f, 1.0f);
            Eigen::Vector4f s_lightDiffuse 		= Eigen::Vector4f(0.6f, 0.52944f, 0.4566f, 0.6f);
            Eigen::Vector4f s_lightSpecular 	= Eigen::Vector4f(0.3f, 0.3f, 0.3f, 1.0f);
            Eigen::Vector3f lightDir 	= Eigen::Vector3f(0.0f, -1.0f, 2.0f);

            program->setUniform(Uniform("materialShininess", s_materialShininess));
            program->setUniform(Uniform("materialAmbient", s_materialAmbient));
            program->setUniform(Uniform("materialDiffuse", s_materialDiffuse));
            program->setUniform(Uniform("materialSpecular", s_materialSpecular));
            program->setUniform(Uniform("lightAmbient", s_lightAmbient));
            program->setUniform(Uniform("lightDiffuse", s_lightDiffuse));
            program->setUniform(Uniform("lightSpecular", s_lightSpecular));
            program->setUniform(Uniform("lightDir", lightDir));
        }
        //This is for the point shader
        //setup a uniform:

    //    GLuint loc = glGetUniformLocation(program->programId(), "camera_array");
    //    glUniformMatrix4fv(loc, 20, false, camera_array_matrices, 0);

        static float * points;


        if(vertex_data_updated)
        {


            glBindBuffer(GL_ARRAY_BUFFER, vbo);
//            glBufferData(GL_ARRAY_BUFFER, (tsdf_vertice_num + 3) * sizeof(VertexElement), &tsdf_visualization_buffer[0], GL_STREAM_DRAW);

            glBufferSubData(GL_ARRAY_BUFFER,0,(tsdf_vertice_num) * sizeof(VertexElement),&tsdf_visualization_buffer[0 * 12]);
            glBindBuffer(GL_ARRAY_BUFFER, 0);
            vertex_data_updated = 0;
            // begin draw signed distance fields

#if 0
            const chisel::ChunkMap &chunks = chiselMap->GetChunkManager().GetChunks();
            chisel::Vec3List centroids = chiselMap->GetChunkManager().GetCentroids();


            int frameNum = frame_list.size();
            char fileName[255];
            memset(fileName,0,255);
            sprintf(fileName,"output/ply/%d.ply",frameNum);
            Point3dList p;
            std::vector<Eigen::Vector3i>colorList;
            int count = 0;
            for (const std::pair<ChunkID, chisel::ChunkPtr>& chunk : chunks)
            {
                const chisel::ChunkPtr &c = chunk.second;
                chisel::Vec3 origin = c->GetOrigin();
                for(int i = 0; i < c->voxels.sdf.size();i++)
                {
                    chisel::Vec3 pos = origin + centroids[i];
                    float sdf = c->voxels.sdf[i];
                    sdf += 0.1;
                    sdf = sdf < 0 ? 0 : sdf;
                    sdf = sdf > 0.25 ? 0.25: sdf;
                    float color = sdf / 0.25;

                    int R,G,B;
                    R = G = B = 0;
                    GetColor(color,0,1,R,G,B);
                    if(!(R == 0 && G == 0 && B == 0))
                    {
                        p.push_back(Eigen::Vector3d(pos(0),pos(1),pos(2)));
                        colorList.push_back(Eigen::Vector3i(B,G,R));
                        count ++;
                    }
                }
            }
            std::cout << "draw points: " << count << std::endl;
            MultiViewGeometry::savePLYFiles(fileName,p,colorList);
#endif
        }


        glBindBuffer(GL_ARRAY_BUFFER, vbo);

        glEnableVertexAttribArray(0);
        glVertexAttribPointer(0, 4, GL_FLOAT, GL_FALSE, 12 * sizeof(float), 0);

        glEnableVertexAttribArray(1);
        glVertexAttribPointer(1, 4, GL_FLOAT, GL_FALSE, 12 * sizeof(float), reinterpret_cast<GLvoid*>(sizeof(Eigen::Vector4f) * 1));

        glEnableVertexAttribArray(2);
        glVertexAttribPointer(2, 4, GL_FLOAT, GL_FALSE, 12 * sizeof(float), reinterpret_cast<GLvoid*>(sizeof(Eigen::Vector4f) * 2));

        glDrawArrays(GL_TRIANGLES,0,tsdf_vertice_num );
        glDisableVertexAttribArray(0);
        glDisableVertexAttribArray(1);
        glDisableVertexAttribArray(2);
        glBindBuffer(GL_ARRAY_BUFFER, 0);
        program->Unbind();


        float vertex_list[24] =
        {
            -0.5f, -0.5f, -0.5f,
            0.5f, -0.5f, -0.5f,
            -0.5f, 0.5f, -0.5f,
            0.5f, 0.5f, -0.5f,
            -0.5f, -0.5f, 0.5f,
            0.5f, -0.5f, 0.5f,
            -0.5f, 0.5f, 0.5f,
            0.5f, 0.5f, 0.5f,
        };
         GLint index_list[24] =
        {
            0, 1,
            2, 3,
            4, 5,
            6, 7,
            0, 2,
            1, 3,
            4, 6,
            5, 7,
            0, 4,
            1, 5,
            7, 3,
            2, 6
        };
#if 0
        DrawCube(searchArea,index_list);
        std::vector<float> corners = chiselMap->candidateCubes;
        int cornersNum = corners.size() / 24;
        float *corner_pointer = corners.data();
        for(int i = 0; i < cornersNum; i++)
        {
            DrawCube(&corner_pointer[i*24],index_list);
        }
#endif

        return 0;
    }

    float GetVoxelResolution(){return voxelResolution;}


    float searchArea[24];
private:


    int chunkSizeX;
    int chunkSizeY;
    int chunkSizeZ;
    float voxelResolution;
    bool useColor;
};


#endif // MOBILEFUSION_H

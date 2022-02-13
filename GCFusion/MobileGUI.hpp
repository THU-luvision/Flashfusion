#ifndef MOBILEGUI_HPP
#define MOBILEGUI_HPP

#include <pangolin/pangolin.h>
#include <pangolin/gl/gl.h>
#include <pangolin/gl/gldraw.h>
#include <GL/glut.h>
#include <GL/glew.h>
#include <Eigen/Eigen>
#include <Eigen/StdVector>

#define GL_GPU_MEM_INFO_CURRENT_AVAILABLE_MEM_NVX 0x9049
class MobileGUI
{
  public:
    MobileGUI(int showCaseMode)
    {

        showcaseMode =  showCaseMode;
        width = 1280;
        height = 980;
        panel = 205;

        width += panel;

        RGB = "RGB";
        DepthNorm = "DepthNorm";
        ModelImg = "ModelImg";
        ModelNorm = "ModelNorm";


        int imageWidth = 640;
        int imageHeight = 480;



        RGBTexture = pangolin::GlTexture(imageWidth,imageHeight,GL_RGBA,true,0,GL_RGB,GL_UNSIGNED_BYTE);
        DepthTexture = pangolin::GlTexture(imageWidth,imageHeight,GL_RGBA,true,0,GL_RGB,GL_UNSIGNED_BYTE);
        DepthNormTexture = pangolin::GlTexture(imageWidth,imageHeight,GL_RGBA,true,0,GL_RGB,GL_UNSIGNED_BYTE);
        ModelImgTexture = pangolin::GlTexture(imageWidth,imageHeight,GL_RGBA,true,0,GL_RGB,GL_UNSIGNED_BYTE);
        ModelNormTexture = pangolin::GlTexture(imageWidth,imageHeight,GL_RGBA,true,0,GL_RGB,GL_UNSIGNED_BYTE);

        pangolin::Params windowParams;
        windowParams.Set("SAMPLE_BUFFERS", 0);
        windowParams.Set("SAMPLES", 0);

        pangolin::CreateWindowAndBind("testPangolin", width, height, windowParams);

        glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
        glPixelStorei(GL_PACK_ALIGNMENT, 1);


        pangolin::SetFullscreen(showcaseMode);

//        gpuMem = new pangolin::Var<int>("ui.GPU memory free", 0);
        glEnable(GL_DEPTH_TEST);
        glDepthMask(GL_TRUE);
        glDepthFunc(GL_LESS);

        s_cam = pangolin::OpenGlRenderState(pangolin::ProjectionMatrix(640, 480, 420, 420, 320, 240, 0.1, 1000),
                                            pangolin::ModelViewLookAt(0, 0, -1, 0, 0, 1, pangolin::AxisNegY));
        pangolin::Display("cam").SetBounds(0, 1.0f, 0, 1.0f, -640 / 480.0)
                                .SetHandler(new pangolin::Handler3D(s_cam));

        pangolin::Display(RGB).SetAspect(640.0f / 480.0f);
        pangolin::Display(DepthNorm).SetAspect(640.0f / 480.0f);
        pangolin::Display(ModelImg).SetAspect(640.0f / 480.0f);
        pangolin::Display(ModelNorm).SetAspect(640.0f / 480.0f);

        pangolin::CreatePanel("ui").SetBounds(0.0, 1.0, 0.0, pangolin::Attach::Pix(panel));
        pangolin::Display("multi").SetBounds(pangolin::Attach::Pix(0), 1 / 4.0f, showcaseMode ? 0 : pangolin::Attach::Pix(180), 1.0)
                                  .SetLayout(pangolin::LayoutEqualHorizontal)
                                  .AddDisplay(pangolin::Display(RGB))
                                  .AddDisplay(pangolin::Display(DepthNorm))
                                  .AddDisplay(pangolin::Display(ModelImg))
                                  .AddDisplay(pangolin::Display(ModelNorm));

        pause = new pangolin::Var<bool>("ui.Pause", false, true);
        step = new pangolin::Var<bool>("ui.Step", false, false);
//        save = new pangolin::Var<bool>("ui.Save", false, false);
        reset = new pangolin::Var<bool>("ui.Reset", false, false);
//        flipColors = new pangolin::Var<bool>("ui.Flip RGB", false, true);

//        pyramid = new pangolin::Var<bool>("ui.Pyramid", true, true);
//        so3 = new pangolin::Var<bool>("ui.SO(3)", true, true);
//        frameToFrameRGB = new pangolin::Var<bool>("ui.Frame to frame RGB", false, true);
//        fastOdom = new pangolin::Var<bool>("ui.Fast Odometry", false, true);
//        rgbOnly = new pangolin::Var<bool>("ui.RGB only tracking", false, true);
//        confidenceThreshold = new pangolin::Var<float>("ui.Confidence threshold", 10.0, 0.0, 24.0);
//        depthCutoff = new pangolin::Var<float>("ui.Depth cutoff", 10.0, 0.0, 12.0);
//        icpWeight = new pangolin::Var<float>("ui.ICP weight", 10.0, 0.0, 100.0);

        followPose = new pangolin::Var<bool>("ui.Follow pose", true, true);
        drawRawCloud = new pangolin::Var<bool>("ui.Draw raw", false, true);
        drawFilteredCloud = new pangolin::Var<bool>("ui.Draw filtered", false, true);
        drawGlobalModel = new pangolin::Var<bool>("ui.Draw global model", true, true);
        drawUnstable = new pangolin::Var<bool>("ui.Draw unstable points", false, true);
        drawPoints = new pangolin::Var<bool>("ui.Draw points", false, true);
        drawColors = new pangolin::Var<bool>("ui.Draw colors", false, true);
//        drawFxaa = new pangolin::Var<bool>("ui.Draw FXAA", showcaseMode, true);
        drawWindow = new pangolin::Var<bool>("ui.Draw time window", false, true);
        drawNormals = new pangolin::Var<bool>("ui.Draw normals", false, true);
        drawTimes = new pangolin::Var<bool>("ui.Draw times", false, true);
//        drawDefGraph = new pangolin::Var<bool>("ui.Draw deformation graph", false, true);
//        drawFerns = new pangolin::Var<bool>("ui.Draw ferns", false, true);
//        drawDeforms = new pangolin::Var<bool>("ui.Draw deformations", true, true);


//        gpuMem = new pangolin::Var<int>("ui.GPU memory free", 0);

//        totalPoints = new pangolin::Var<std::string>("ui.Total points", "0");
//        totalNodes = new pangolin::Var<std::string>("ui.Total nodes", "0");
//        totalFerns = new pangolin::Var<std::string>("ui.Total ferns", "0");
//        totalDefs = new pangolin::Var<std::string>("ui.Total deforms", "0");
//        totalFernDefs = new pangolin::Var<std::string>("ui.Total fern deforms", "0");

//        trackInliers = new pangolin::Var<std::string>("ui.Inliers", "0");
//        trackRes = new pangolin::Var<std::string>("ui.Residual", "0");
//        logProgress = new pangolin::Var<std::string>("ui.Log", "0");
    }
    virtual ~MobileGUI()
    {
 //       delete gpuMem;
        delete renderBuffer;
        delete colorFrameBuffer;
    }

    void PreCall()
    {
        glClearColor(1.0f,1.0f, 1.0f,1.0f);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        width = pangolin::DisplayBase().v.w;
        height = pangolin::DisplayBase().v.h;

        pangolin::Display("cam").Activate(s_cam);
    }
    void PostCall()
    {
//        GLint cur_avail_mem_kb = 0;
//        glGetIntegerv(GL_GPU_MEM_INFO_CURRENT_AVAILABLE_MEM_NVX, &cur_avail_mem_kb);

//        int memFree = cur_avail_mem_kb / 1024;

//        gpuMem->operator=(memFree);

        pangolin::FinishFrame();

        glFinish();
    }

    void loadImageToTexture(pangolin::GlTexture *texture, unsigned char*imageArray)
    {
        texture->Upload(imageArray,GL_RGB,GL_UNSIGNED_BYTE);
    }

    void DisplayImg(const std::string & id, pangolin::GlTexture *img)
    {

        glDisable(GL_DEPTH_TEST);

        pangolin::Display(id).Activate();
        img->RenderToViewport(true);
        glEnable(GL_DEPTH_TEST);
    }

    void setModelView(Eigen::Matrix4f &currPose, int iclnuim)
    {

        pangolin::OpenGlMatrix mv;
        Eigen::Matrix3f currRot = currPose.topLeftCorner(3, 3);

        Eigen::Matrix3f rot = Eigen::Matrix3f::Identity();
        Eigen::Quaternionf currQuat(currRot);
        Eigen::Vector3f forwardVector(0, 0, 1);
        Eigen::Vector3f upVector(0, iclnuim ? 1 : -1, 0);

        Eigen::Vector3f forward = (currQuat * forwardVector).normalized();
        Eigen::Vector3f up = (currQuat * upVector).normalized();

        Eigen::Vector3f eye(currPose(0, 3), currPose(1, 3), currPose(2, 3));

        eye -= forward;

        Eigen::Vector3f at = eye + forward;

        Eigen::Vector3f z = (eye - at).normalized();  // Forward
        Eigen::Vector3f x = up.cross(z).normalized(); // Right
        Eigen::Vector3f y = z.cross(x);

        Eigen::Matrix4d m;
        m << x(0),  x(1),  x(2),  -(x.dot(eye)),
             y(0),  y(1),  y(2),  -(y.dot(eye)),
             z(0),  z(1),  z(2),  -(z.dot(eye)),
                0,     0,     0,              1;

        memcpy(&mv.m[0], m.data(), sizeof(Eigen::Matrix4d));

        s_cam.SetModelViewMatrix(mv);
    }

    bool showcaseMode;
    int width;
    int height;
    pangolin::GlRenderBuffer * renderBuffer;
    pangolin::GlFramebuffer * colorFrameBuffer;

    pangolin::OpenGlRenderState s_cam;
    pangolin::Var<int> * gpuMem;

    std::string RGB;
    std::string DepthNorm;
    std::string ModelImg;
    std::string ModelNorm;
    int panel;

    pangolin::GlTexture RGBTexture;
    pangolin::GlTexture DepthTexture;
    pangolin::GlTexture DepthNormTexture;
    pangolin::GlTexture ModelImgTexture;
    pangolin::GlTexture ModelNormTexture;


    pangolin::Var<bool> * pause,
                        * step,
                        * save,
                        * reset,
                        * flipColors,
                        * rgbOnly,
                        * pyramid,
                        * so3,
                        * frameToFrameRGB,
                        * fastOdom,
                        * followPose,
                        * drawRawCloud,
                        * drawFilteredCloud,
                        * drawNormals,
                        * autoSettings,
                        * drawDefGraph,
                        * drawColors,
                        * drawFxaa,
                        * drawGlobalModel,
                        * drawUnstable,
                        * drawPoints,
                        * drawTimes,
                        * drawFerns,
                        * drawDeforms,
                        * drawWindow;

    pangolin::Var<std::string> * totalPoints,
                               * totalNodes,
                               * totalFerns,
                               * totalDefs,
                               * totalFernDefs,
                               * trackInliers,
                               * trackRes,
                               * logProgress;
    pangolin::Var<float> * confidenceThreshold,
                         * depthCutoff,
                         * icpWeight;
};


#endif // MOBILEGUI_HPP

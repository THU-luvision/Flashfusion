#ifndef GCSLAM_H
#define GCSLAM_H
#include "MultiViewGeometry.h"
#include <iostream>



class GCSLAM
{

public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    GCSLAM()
    {
        GCSLAM_INIT_FLAG = 0;
        minimumDisparity = 0.1;
        salientScoreThreshold = 1.0;
        maxCandidateNum = 5;
    }
    void init(const int maximum_frame_num, const MultiViewGeometry::CameraPara &camera_para_t)
    {
        GCSLAM_INIT_FLAG = 1;
        mild.init(FEATURE_TYPE_ORB, 16, 0);
        keyFrameCorrList.reserve(maximum_frame_num * 5);
        KeyframeDataList.reserve(maximum_frame_num);
        globalFrameList.reserve(maximum_frame_num);
        camera_para = camera_para_t;
    }



    void finalBA(std::vector<Frame> &frame_list)
    {

        std::vector<MultiViewGeometry::FrameCorrespondence> &fCorrList_keyframes = keyFrameCorrList;
        std::vector<MultiViewGeometry::KeyFrameDatabase> &kflist = KeyframeDataList;
        initGraphHuberNorm(fCorrList_keyframes,frame_list);
        MultiViewGeometry::optimizeKeyFrameMap(fCorrList_keyframes, frame_list, kflist,0);
    }



    void select_closure_candidates(Frame &f, std::vector<int> &candidate_frame_index);

    void update_keyframe(int newKeyFrameIndex,
                         MultiViewGeometry::FrameCorrespondence &key_frame_corr,
                         float average_disparity,
                         PoseSE3d relative_pose_from_key_to_new,
                         int registration_success);

    /*
    frame_input will be added into frame database, and matched with previous keyframes
    if the disparity between frame_input and previous keyframe is larger than a threshold, frame_input is recogonized as a new keyframe
    FastBA is activated if new keyframe is inserted, and all previous frames' poses are updated
    */
    void update_frame(Frame &frame_input);


    void updateMapOrigin(std::vector<MultiViewGeometry::FrameCorrespondence> &fCorrCandidate,
                         std::vector<int> &registration_success_list, int newKeyFrameIndex);
    inline const std::vector<MultiViewGeometry::KeyFrameDatabase> & GetKeyframeDataList() {return KeyframeDataList;}




    void SetMinimumDisparity(float inputMinDisp) { minimumDisparity = inputMinDisp; }
    void SetMaxCandidateNum(int inputMaxCandidateNum) {maxCandidateNum = inputMaxCandidateNum; }
    void SetSalientScoreThreshold(float salientThreshold) {salientScoreThreshold = salientThreshold; }

    std::vector<Frame> globalFrameList;                 // database for all frames
    std::vector<MultiViewGeometry::FrameCorrespondence> keyFrameCorrList; // database for matched keyframe pairs
    std::vector<MultiViewGeometry::KeyFrameDatabase>  KeyframeDataList;   // database for all keyframes

private:

    int GCSLAM_INIT_FLAG;
    MILD::LoopClosureDetector mild;                                       // interface for loop closure detector
    MultiViewGeometry::CameraPara camera_para;                            // camera parameters

    float minimumDisparity;        // minimum disparity to update keyframes
    float salientScoreThreshold;   // minimum salient score for loop closures
    int maxCandidateNum;         // maximum closure candidate num
};

#endif


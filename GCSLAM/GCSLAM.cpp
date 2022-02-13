#include "GCSLAM.h"
#include "../CHISEL/src/open_chisel/Stopwatch.h"



using namespace std;



void GCSLAM::select_closure_candidates(Frame &f, std::vector<int> &candidate_frame_index)

{

    std::vector<MultiViewGeometry::KeyFrameDatabase> &kflist = KeyframeDataList;
    MILD::LoopClosureDetector & lcd = mild;
    std::vector<Frame> & frame_list = globalFrameList;

  MILD::BayesianFilter spatial_filter;
  std::vector<float > similarity_score;
  lcd.query_database(f.descriptor, similarity_score);
  // select candidates
  candidate_frame_index.clear();
  std::vector<float> salient_score;
  std::vector<MILD::LCDCandidate> candidates;
  spatial_filter.calculateSalientScore(similarity_score, salient_score);

  TICK("GCSLAM::GlobalRegistration");

  //only select top 5, disgard the last frame
  for (int k = 0; k < kflist.size() - 1; k++)
  {
    if (salient_score[k] > salientScoreThreshold &&
        frame_list[kflist[k].keyFrameIndex].is_keyframe)
    {
      MILD::LCDCandidate candidate(salient_score[k],k);
      candidates.push_back(candidate);
    }
  }

  std::sort(candidates.begin(), candidates.end(),greater<MILD::LCDCandidate>());
  for (int k = 0; k < fmin(candidates.size(), maxCandidateNum); k++)
  {
//    cout << kflist[candidates[k].index].keyFrameIndex << " " << candidates[k].salient_score << endl;
    candidate_frame_index.push_back(candidates[k].index);
  }


  string candidates_str = "candidates: ";
  string candidates_score;
  for (int k = 0; k < candidate_frame_index.size(); k++)
  {
    candidates_str += std::to_string(kflist[candidate_frame_index[k]].keyFrameIndex) + " ";
  }
//  cout << "running frame : " << f.frame_index << " " << candidates_str << endl;
}

void GCSLAM::update_keyframe(int newKeyFrameIndex,
                          MultiViewGeometry::FrameCorrespondence &key_frame_corr,
                          float average_disparity,
                          PoseSE3d relative_pose_from_key_to_new,
                          int registration_success)
{
  float scale_change_ratio;

  bool update_keyframe_from_dense_matching =0;
  int global_tracking_success = 0;


  std::vector<MultiViewGeometry::FrameCorrespondence> &fCorrList_keyframes = keyFrameCorrList;
  std::vector<MultiViewGeometry::KeyFrameDatabase> &kflist = KeyframeDataList;
  MILD::LoopClosureDetector & lcd = mild;
  std::vector<Frame> & frame_list = globalFrameList;

  MILD::SparseMatcher sparseMatcher(FEATURE_TYPE_ORB, 32, 0, 50);
  sparseMatcher.train(frame_list[newKeyFrameIndex].descriptor);
  // loop closure detection
  std::vector<int> candidate_keyframes;

  TICK("GCSLAM::MILD");
  select_closure_candidates(frame_list[newKeyFrameIndex],candidate_keyframes);

  TOCK("GCSLAM::MILD");

  TICK("GCSLAM::GlobalRegistration");
  //*********************** add current keyframe to keyframe list
  MultiViewGeometry::KeyFrameDatabase kfd(newKeyFrameIndex);
  kflist.push_back(kfd);

  //*********************** select candidates

  std::vector<MultiViewGeometry::FrameCorrespondence> fCorrCandidate;

  for (size_t k = 0; k < candidate_keyframes.size(); k++)
  {
    int candidate_frame_index = kflist[candidate_keyframes[k]].keyFrameIndex;
    MultiViewGeometry::FrameCorrespondence global_frame_corr(frame_list[candidate_frame_index], frame_list[newKeyFrameIndex]);
    fCorrCandidate.push_back(global_frame_corr);
//    std::cout << "candidate key frame: " << kflist[candidate_keyframes[k]].keyFrameIndex << std::endl;
  }

  std::vector<float> average_disparity_list(candidate_keyframes.size());
  std::vector<int> registration_success_list(candidate_keyframes.size());
  PoseSE3dList relative_pose_from_ref_to_new_list(candidate_keyframes.size());
  for (size_t k = 0; k < candidate_keyframes.size(); k++)
  {
    int candidate_frame_index = kflist[candidate_keyframes[k]].keyFrameIndex;

    registration_success_list[k] = 1e8;
    average_disparity_list[k] = 1e8;
    registration_success_list[k] = MultiViewGeometry::FrameMatchingTwoViewRGB(fCorrCandidate[k],
                                                                              camera_para,
                                                                              sparseMatcher,
                                                                              relative_pose_from_ref_to_new_list[k],
                                                                              average_disparity_list[k],
                                                                              scale_change_ratio,
                                                                              update_keyframe_from_dense_matching);
    relative_pose_from_ref_to_new_list[k] = relative_pose_from_ref_to_new_list[k].inverse();

  }
  relative_pose_from_ref_to_new_list.push_back(relative_pose_from_key_to_new);
  registration_success_list.push_back(registration_success);
  average_disparity_list.push_back(average_disparity);
  fCorrCandidate.push_back(key_frame_corr);


  for(size_t k = 0; k < registration_success_list.size(); k++)
  {

      if(registration_success_list[k] )
      {
          kflist.back().corresponding_keyframes.push_back(fCorrCandidate[k].frame_ref.frame_index);
 //         kflist.back().relative_pose_from_key_to_current.push_back(relative_pose_from_ref_to_new_list[k]);
      }
  }
  //update camera pose based on previous results
  float min_average_disparity = 1e9;
  int min_index = 0;
//  std::cout << "average disparity / reprojection error: ";
  for (size_t k = 0; k < fCorrCandidate.size(); k++)
  {
//    std::cout << fCorrCandidate[k].frame_ref.frame_index << "	/"
//      << average_disparity_list[k] << "	/"
//      << registration_success_list[k] << std::endl;
    if (min_average_disparity > average_disparity_list[k] && registration_success_list[k])
    {
      min_average_disparity = average_disparity_list[k];
      min_index = k;
      global_tracking_success = 1;
    }
  }
//  std::cout << std::endl;

  int current_map_origin = 0;
  if (global_tracking_success == 1)
  {
    frame_list[newKeyFrameIndex].tracking_success = 1;
    frame_list[newKeyFrameIndex].pose_sophus[0] = frame_list[fCorrCandidate[min_index].frame_ref.frame_index].pose_sophus[0]
            * relative_pose_from_ref_to_new_list[min_index] ;
  }


  if(!global_tracking_success)
  {
    current_map_origin = newKeyFrameIndex;
//      std::cout << "update anchor keyframe index! " << std::endl;
  }
  else
  {

    std::vector<int> matched_frames;
    for (size_t k = 0; k < fCorrCandidate.size(); k++)
    {
      if (registration_success_list[k] )
      {
        matched_frames.push_back(fCorrCandidate[k].frame_ref.origin_index);
      }
    }
    current_map_origin = *max_element(matched_frames.begin(), matched_frames.end());
  }
//  std::cout << "add new keyframe!" << std::endl;
  frame_list[newKeyFrameIndex].is_keyframe = 1;
  frame_list[newKeyFrameIndex].origin_index = current_map_origin;
  int reg_success_cnt = 0;
  for (size_t k = 0; k < fCorrCandidate.size(); k++)
  {
    if (registration_success_list[k] )
    {
        reg_success_cnt ++;
    }
  }
  if(reg_success_cnt < 4)
  {
      lcd.construct_database(frame_list[newKeyFrameIndex].descriptor);
  }
  else
  {
      cv::Mat descriptor;
      descriptor.release();
      lcd.construct_database(descriptor);
  }
  for (size_t k = 0; k < fCorrCandidate.size(); k++)
  {
    if (registration_success_list[k] )
    {
      fCorrList_keyframes.push_back(fCorrCandidate[k]);
    }
  }
  updateMapOrigin(fCorrCandidate, registration_success_list,newKeyFrameIndex);
  TOCK("GCSLAM::GlobalRegistration");
 }

void GCSLAM::updateMapOrigin(std::vector<MultiViewGeometry::FrameCorrespondence> &fCorrCandidate,
                             std::vector<int> &registration_success_list,
                             int newKeyFrameIndex)
{


    std::vector<MultiViewGeometry::KeyFrameDatabase> &kflist = KeyframeDataList;
    std::vector<Frame> &frame_list = globalFrameList;

    std::vector<int> keyFrameIndex(frame_list.size());
    for (size_t k = 0; k < frame_list.size(); k++)
    {
      keyFrameIndex[k] = -1;
    }
    for (size_t k = 0; k < kflist.size(); k++)
    {
      keyFrameIndex[kflist[k].keyFrameIndex] = k;
    }
    std::vector<int >tracked_frame_index;
    for (size_t k = 0; k < fCorrCandidate.size(); k++)
    {
      if (registration_success_list[k])
      {
        int ref_frame_index = fCorrCandidate[k].frame_ref.frame_index;
        if (keyFrameIndex[ref_frame_index] < 0)
        {
          std::cout << "warning! ref frame is not keyframe" << std::endl;
        }
        int ref_origin = frame_list[ref_frame_index].origin_index;
        int current_origin = frame_list[newKeyFrameIndex].origin_index;
        frame_list[newKeyFrameIndex].origin_index = std::min(ref_origin,current_origin);

        if(0)
        {
          if (ref_origin < current_origin)
          {
            for (int keyframeCnt = 0; keyframeCnt < kflist.size(); keyframeCnt++)
            {
              if (frame_list[kflist[keyframeCnt].keyFrameIndex].origin_index == current_origin)
              {
                frame_list[kflist[keyframeCnt].keyFrameIndex].origin_index = ref_origin;
                frame_list[kflist[keyframeCnt].keyFrameIndex].is_keyframe = 0;
                for (int localframeCnt = 0; localframeCnt < kflist[keyframeCnt].corresponding_frames.size(); localframeCnt++)
                {
                  frame_list[kflist[keyframeCnt].corresponding_frames[localframeCnt]].origin_index = ref_origin;
                }
              }
            }
          }
          if (current_origin < ref_origin)
          {
            for (int keyframeCnt = 0; keyframeCnt < kflist.size() - 1; keyframeCnt++)
            {
              if (frame_list[kflist[keyframeCnt].keyFrameIndex].origin_index == ref_origin)
              {
                frame_list[kflist[keyframeCnt].keyFrameIndex].origin_index = current_origin;
                frame_list[kflist[keyframeCnt].keyFrameIndex].is_keyframe = 0;
                for (int localframeCnt = 0; localframeCnt < kflist[keyframeCnt].corresponding_frames.size(); localframeCnt++)
                {
                  frame_list[kflist[keyframeCnt].corresponding_frames[localframeCnt]].origin_index = current_origin;
                }
              }
            }
          }
        }

      }
    }

}


void GCSLAM::update_frame(Frame &frame_input)
{
    if(GCSLAM_INIT_FLAG == 0)
    {
        std::cout << "error ! gcSLAM not initialized! " << std::endl;
        return;
    }
    std::vector<MultiViewGeometry::FrameCorrespondence> &fCorrList_keyframes = keyFrameCorrList;
    std::vector<MultiViewGeometry::KeyFrameDatabase> &kflist = KeyframeDataList;
    std::vector<Frame> &frame_list = globalFrameList;
    MILD::LoopClosureDetector & lcd = mild;


    //*********************** add current frame to database
    frame_list.push_back(frame_input);
    Frame &f = frame_list.back();

    //*********************** init keyframe database
    if (kflist.size() == 0)
    {
      MultiViewGeometry::KeyFrameDatabase kfd(f.frame_index);
      kflist.push_back(kfd);
      lcd.construct_database(f.descriptor);
      f.origin_index = f.frame_index;
      f.tracking_success = 1;
      f.is_fixed_frame = 1;
      f.is_keyframe = 1;
      return;
    }

    int add_new_key_frame_flag = 0;
    bool registration_success = 0;
    float average_disparity = 1e8;
    float scale_change_ratio = 0;


    //*********************** SparseMatcher is used for efficient binary feature matching
    MILD::SparseMatcher sparseMatcher(FEATURE_TYPE_ORB, 32, 0, 50);
    sparseMatcher.train(f.descriptor);

    int last_keyframe_index = kflist.back().keyFrameIndex;
    int anchor_frame = f.frame_index - 1;

    static int local_tracking_cnt = 0;
    PoseSE3d relative_transform_from_key_to_new;
    if(anchor_frame >= 0 && frame_list[anchor_frame].tracking_success)
    {
      relative_transform_from_key_to_new = frame_list[anchor_frame].pose_sophus[0].inverse() *
              frame_list[last_keyframe_index].pose_sophus[0];
    }

    MultiViewGeometry::FrameCorrespondence key_frame_corr(frame_list[last_keyframe_index], f);
    bool update_keyframe_from_dense_matching = 0;

    //*********************** Match two frames based on RGBD features
    registration_success = MultiViewGeometry::FrameMatchingTwoViewRGB(key_frame_corr,
                                                                      camera_para,
                                                                      sparseMatcher,
                                                                      relative_transform_from_key_to_new,
                                                                      average_disparity,
                                                                      scale_change_ratio,
                                                                      update_keyframe_from_dense_matching,
                                                                      1);
    int update_keyframe_flag = 0;


    if((average_disparity > minimumDisparity  || (scale_change_ratio > 0.4)) && registration_success)
    {
            update_keyframe_flag = 1;
    }

    if(!registration_success)
    {
        local_tracking_cnt++;
    }

    if(local_tracking_cnt > 3)
    {
            update_keyframe_flag = 1;
    }
    PoseSE3d relative_pose_from_key_to_new = relative_transform_from_key_to_new;
    relative_pose_from_key_to_new = relative_pose_from_key_to_new.inverse();
    if (registration_success && !update_keyframe_flag)
    {
        local_tracking_cnt = 0;
        f.tracking_success = 1;
        f.pose_sophus[0] = frame_list[last_keyframe_index].pose_sophus[0] * relative_pose_from_key_to_new;
        f.origin_index = frame_list[last_keyframe_index].origin_index;

        kflist.back().corresponding_frames.push_back(f.frame_index);

        kflist.back().localFrameCorrList.push_back(key_frame_corr);
        kflist.back().relative_pose_from_key_to_current.push_back(relative_pose_from_key_to_new);
    }

    //*********************** update keyframe
    if (update_keyframe_flag)
    {
      local_tracking_cnt = 0;
      update_keyframe(f.frame_index,
                      key_frame_corr,
                      average_disparity,
                      relative_pose_from_key_to_new,
                      registration_success);
      f.is_keyframe = 1;
      //*********************** fastBA for globally consistent pose estimation
      TICK("GCSLAM::FastBA");
      MultiViewGeometry::optimizeKeyFrameMap(fCorrList_keyframes, frame_list, kflist,0);
      TOCK("GCSLAM::FastBA");
    }
}

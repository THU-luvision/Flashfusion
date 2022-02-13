#ifndef REALSENSEINTERFACE_H
#define REALSENSEINTERFACE_H



#include <librealsense2/rs.hpp>



float get_depth_scale(rs2::device dev);
rs2_stream find_stream_to_align(const std::vector<rs2::stream_profile>& streams);
bool profile_changed(const std::vector<rs2::stream_profile>& current, const std::vector<rs2::stream_profile>& prev);


#endif // REALSENSEINTERFACE_H

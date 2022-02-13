/*
 * This file is part of ElasticFusion.
 *
 * Copyright (C) 2015 Imperial College London
 * 
 * The use of the code within this file and all code within files that 
 * make up the software that is ElasticFusion is permitted for 
 * non-commercial purposes only.  The full terms and conditions that 
 * apply to the code within this file are detailed within the LICENSE.txt 
 * file and at <http://www.imperial.ac.uk/dyson-robotics-lab/downloads/elastic-fusion/elastic-fusion-license/> 
 * unless explicitly stated.  By downloading this file you agree to 
 * comply with these terms.
 *
 * If you wish to use any of this code for commercial purposes then 
 * please email researchcontracts.engineering@imperial.ac.uk.
 *
 */

#version 450 core

layout (location = 0) in vec4 position;
layout (location = 1) in vec4 color;
layout (location = 2) in vec4 normal;

uniform mat4 proj_matrix;
uniform mat4 view_matrix;
uniform float threshold;
uniform int colorType;
uniform int time;
uniform int timeDelta;


uniform mat4 user_view_matrix;
uniform mat4 user_light_matrix;
uniform vec4 user_rot_center;

out VS_OUT
{
    vec3 v;
    vec3 fn;
    vec3 bn;
    vec4 vColor;
} vs_out;



#include "color.glsl"

void main()
{
    if(position.w > threshold)
    {

//        gl_Position = proj_matrix * view_matrix * vec4(position.xyz, 1.0);

        mat4 mv_mat = view_matrix;
        mat3 normal_mat = mat3(user_view_matrix) * mat3(mv_mat);

        /*calculate vertex coordinates in camera frame*/
        vec4 v_cam = mv_mat * vec4(position.xyz, 1.0);

        /*calculate vertex coordinates in user view*/
        vec4 v_user = user_view_matrix * vec4((v_cam-user_rot_center).xyz, 1.0) + user_rot_center;

        vs_out.v = v_user.xyz;

        /*calculating front and back normal directions*/
        vec3 front_normal = normalize(normal_mat * normal.xyz);
        vec3 back_normal = -front_normal;
        vs_out.fn = front_normal;
        vs_out.bn = back_normal;

//	    gl_Position = MVP * pose * vec4(position.xyz, 1.0);
//        gl_Position = proj_matrix * v_user;
	gl_Position = proj_matrix * view_matrix * vec4(position.xyz, 1.0);
        if(colorType == 1)
        {
            vs_out.vColor = vec4(-normal.xyz, 1.0);
        }
        else if(colorType == 2)
        {
            vs_out.vColor = vec4(decodeColor(color.x), 1.0);
        }
        else if(colorType == 3)
        {
            vs_out.vColor = vec4(decodeColor(color.x), 1.0);
            float minimum = 1.0f;
            float maximum = 300;
            float ctime = color.z - (int(color.z /300)) * 300;
            ctime = min(ctime, maximum);
            ctime = max(ctime, 0);   
            vs_out.vColor.x = min(1,  3 * ctime / maximum );
            vs_out.vColor.y = min(1,  max(0, (3 * ctime - maximum) / maximum ));
            vs_out.vColor.z = min(1,  max(0, (3 * ctime - 2 * maximum) / maximum ));
        }
        else
        {

//            shading
//            vColor = vec4((vec3(.5f, .5f, .5f) * abs(dot(normal.xyz, vec3(1.0, 1.0, 1.0)))) + vec3(0.1f, 0.1f, 0.1f), 1.0f);


           
        }
    }
    else
    {
        gl_Position = vec4(-10, -10, 0, 1);
    }
}

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


uniform vec3 Lightla;
uniform vec3 Lightld;
uniform vec3 Lightls;
uniform vec3 Lightldir;
uniform vec3 fma;
uniform vec3 fmd;
uniform vec3 fms;
uniform float fss;
uniform vec3 bma;
uniform vec3 bmd;
uniform vec3 bms;
uniform float bss;

uniform int colorType;
in VS_OUT
{
	vec3 v;
	vec3 fn;
	vec3 bn;
    vec4 vColor;
} fs_in;


out vec4 FragColor;

void main()
{
	vec4 PhongColor;
    vec3 ldir = normalize(-Lightldir);
    vec3 fn = normalize(fs_in.fn);
    vec3 bn = normalize(fs_in.bn);
    vec3 vdir = normalize(-fs_in.v);
    vec3 frdir = reflect(-ldir, fn);
    vec3 brdir = reflect(-ldir, bn);
    if (gl_FrontFacing) {
        vec3 ka = Lightla * fma;
        vec3 kd = Lightld * fmd;
        vec3 ks = Lightls * fms;
        /*calculate Phong lighting of front-facing fragment*/
        vec3 fca = ka;
        vec3 fcd = kd * max(dot(fn, ldir), 0.0);
        vec3 fcs = ks * pow(max(dot(vdir, frdir), 0.0), fss);
        vec3 fc = clamp(fca + fcd + fcs, 0.0, 1.0);
        PhongColor = vec4(fc, 1.0);
    }
    else{
        vec3 ka = Lightla * bma;
        vec3 kd = Lightld * bmd;
        vec3 ks = Lightls * bms;
        /*calculate Phong lighting of back-facing fragment*/
        vec3 bca = ka;
        vec3 bcd = kd * max(dot(bn, ldir), 0.0);
        vec3 bcs = ks * pow(max(dot(vdir, brdir), 0.0), bss);
        vec3 bc = clamp(bca + bcd + bcs, 0.0, 1.0);
        PhongColor = vec4(bc, 1.0);
    }

    if(colorType == 1 || colorType == 2 || colorType == 3)
    {
	    FragColor = fs_in.vColor;
	  //  FragColor = vec4(fss/2,fss/2,fss/2,1.0);
	    // if(colorType == 1)
	    // {
	    // 	FragColor = vec4(fma,1);
	    // }
	    // if(colorType == 2)
	    // {
	    // 	FragColor = vec4(fms, 1);
	    // }
	    // if(colorType == 3)
	    // {
	    // 	FragColor = vec4(fmd, 1);
	    //}
    }
    else
    {
    	FragColor = PhongColor;
    }
}

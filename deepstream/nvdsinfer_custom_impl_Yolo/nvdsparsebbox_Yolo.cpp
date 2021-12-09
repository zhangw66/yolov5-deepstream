/*
 * Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a
 * copy of this software and associated documentation files (the "Software"),
 * to deal in the Software without restriction, including without limitation
 * the rights to use, copy, modify, merge, publish, distribute, sublicense,
 * and/or sell copies of the Software, and to permit persons to whom the
 * Software is furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL
 * THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
 * DEALINGS IN THE SOFTWARE.
 */

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstring>
#include <fstream>
#include <iostream>
#include <unordered_map>
#include <map>
#include "nvdsinfer_custom_impl.h"
#include "../../common.hpp"
#define NMS_THRESH 0.5
#define CONF_THRESH 0.7

extern "C" bool NvDsInferParseCustomYoloV5(
    std::vector<NvDsInferLayerInfo> const &outputLayersInfo,
    NvDsInferNetworkInfo const &networkInfo,
    NvDsInferParseDetectionParams const &detectionParams,
    std::vector<NvDsInferParseObjectInfo> &objectList);


/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

/* This is a sample bounding box parsing function for the sample YoloV5m detector model */
static bool NvDsInferParseYoloV5(
    std::vector<NvDsInferLayerInfo> const& outputLayersInfo,
    NvDsInferNetworkInfo const& networkInfo,
    NvDsInferParseDetectionParams const& detectionParams,
    std::vector<NvDsInferParseObjectInfo>& objectList)
{
    if (Yolo::CLASS_NUM != detectionParams.numClassesConfigured)
    {
        std::cerr << "WARNING: Num classes mismatch. Configured:"
                  << detectionParams.numClassesConfigured
                  << ", detected by network: " << Yolo::CLASS_NUM << std::endl;
    }

    std::vector<Yolo::Detection> res;

    nms(res, (float*)(outputLayersInfo[0].buffer), CONF_THRESH, NMS_THRESH);
    //std::cout<<"Nms done sucessfully----"<<std::endl;
    
    for(auto& r : res) {
	    NvDsInferParseObjectInfo oinfo;        
        
	    oinfo.classId = r.class_id;
	    oinfo.left    = static_cast<unsigned int>(r.bbox[0]-r.bbox[2]*0.5f);
	    oinfo.top     = static_cast<unsigned int>(r.bbox[1]-r.bbox[3]*0.5f);
	    oinfo.width   = static_cast<unsigned int>(r.bbox[2]);
	    oinfo.height  = static_cast<unsigned int>(r.bbox[3]);
	    oinfo.detectionConfidence = r.conf;
        //std::cout << static_cast<unsigned int>(r.bbox[0]) << "," << static_cast<unsigned int>(r.bbox[1]) << "," << static_cast<unsigned int>(r.bbox[2]) << "," 
        //          << static_cast<unsigned int>(r.bbox[3]) << "," << static_cast<unsigned int>(r.class_id) << "," << static_cast<unsigned int>(r.conf) << std::endl;
	    objectList.push_back(oinfo);        
    }
    
    return true;
}
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

/* This is a sample bounding box parsing function for the sample YoloV4 detector model */
/* C-linkage to prevent name-mangling */
extern "C" bool NvDsInferParseCustomYoloV5(
    std::vector<NvDsInferLayerInfo> const &outputLayersInfo,
    NvDsInferNetworkInfo const &networkInfo,
    NvDsInferParseDetectionParams const &detectionParams,
    std::vector<NvDsInferParseObjectInfo> &objectList)
{
    return NvDsInferParseYoloV5(
        outputLayersInfo, networkInfo, detectionParams, objectList);
}

/* Check that the custom function has been defined correctly */
CHECK_CUSTOM_PARSE_FUNC_PROTOTYPE(NvDsInferParseCustomYoloV5);

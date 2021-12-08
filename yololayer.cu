#include <assert.h>
#include "yololayer.h"
#include "utils.h"

using namespace Yolo;

namespace nvinfer1
{
    YoloLayerPlugin::YoloLayerPlugin(int classCount, int netWidth, int netHeight, int maxOut, const std::vector<Yolo::YoloKernel>& vYoloKernel)
    {
	    printf("%s:%d\n", __func__, __LINE__);
        mClassCount = classCount;
        mYoloV5NetWidth = netWidth;
        mYoloV5NetHeight = netHeight;
        mMaxOutObject = maxOut;
        mYoloKernel = vYoloKernel;
        mKernelCount = vYoloKernel.size();
	std::cout << "class count: " << mClassCount
		  << " NET WIDTH: " << mYoloV5NetWidth
		  << " NET HEIGHT: " << mYoloV5NetHeight
		  << " max out: " << mMaxOutObject
		  << " thread count: " << mThreadCount
		  << " kernel count: " << mKernelCount << std::endl;
        CUDA_CHECK(cudaMallocHost(&mAnchor, mKernelCount * sizeof(void*)));
        size_t AnchorLen = sizeof(float)* CHECK_COUNT * 2;
        for (int ii = 0; ii < mKernelCount; ii++)
        {
            CUDA_CHECK(cudaMalloc(&mAnchor[ii], AnchorLen));
            const auto& yolo = mYoloKernel[ii];
	    std::cout << "width: " << yolo.width;
	    std::cout << " height: " << yolo.height;
	    for (int iii = 0; iii < CHECK_COUNT * 2;iii++)
	    std::cout << " anchors: " << yolo.anchors[iii] << std::endl;
	    
	    CUDA_CHECK(cudaMemcpy(mAnchor[ii], yolo.anchors, AnchorLen, cudaMemcpyHostToDevice));
        }
    }
    YoloLayerPlugin::~YoloLayerPlugin()
    {
	    printf("%s:%d\n", __func__, __LINE__);
        for (int ii = 0; ii < mKernelCount; ii++)
        {
            CUDA_CHECK(cudaFree(mAnchor[ii]));
        }
        CUDA_CHECK(cudaFreeHost(mAnchor));
    }

    // create the plugin at runtime from a byte stream
    YoloLayerPlugin::YoloLayerPlugin(const void* data, size_t length)
    {
	    printf("%s:%d\n", __func__, __LINE__);
        using namespace Tn;
        const char *d = reinterpret_cast<const char *>(data), *a = d;
        read(d, mClassCount);
        read(d, mThreadCount);
        read(d, mKernelCount);
        read(d, mYoloV5NetWidth);
        read(d, mYoloV5NetHeight);
        read(d, mMaxOutObject);
        mYoloKernel.resize(mKernelCount);
        auto kernelSize = mKernelCount * sizeof(YoloKernel);
        memcpy(mYoloKernel.data(), d, kernelSize);
        d += kernelSize;
        CUDA_CHECK(cudaMallocHost(&mAnchor, mKernelCount * sizeof(void*)));
        size_t AnchorLen = sizeof(float)* CHECK_COUNT * 2;
        for (int ii = 0; ii < mKernelCount; ii++)
        {
            CUDA_CHECK(cudaMalloc(&mAnchor[ii], AnchorLen));
            const auto& yolo = mYoloKernel[ii];
            CUDA_CHECK(cudaMemcpy(mAnchor[ii], yolo.anchors, AnchorLen, cudaMemcpyHostToDevice));
        }
        assert(d == a + length);
    }

    void YoloLayerPlugin::serialize(void* buffer) const
    {
	    printf("%s:%d\n", __func__, __LINE__);
        using namespace Tn;
        char* d = static_cast<char*>(buffer), *a = d;
        write(d, mClassCount);
        write(d, mThreadCount);
        write(d, mKernelCount);
        write(d, mYoloV5NetWidth);
        write(d, mYoloV5NetHeight);
        write(d, mMaxOutObject);
        auto kernelSize = mKernelCount * sizeof(YoloKernel);
        memcpy(d, mYoloKernel.data(), kernelSize);
        d += kernelSize;

        assert(d == a + getSerializationSize());
    }

    size_t YoloLayerPlugin::getSerializationSize() const
    {
	    printf("%s:%d\n", __func__, __LINE__);
        return sizeof(mClassCount) + sizeof(mThreadCount) + sizeof(mKernelCount) + sizeof(Yolo::YoloKernel) * mYoloKernel.size() + sizeof(mYoloV5NetWidth) + sizeof(mYoloV5NetHeight) + sizeof(mMaxOutObject);
    }

    int YoloLayerPlugin::initialize()
    {
	    printf("%s:%d\n", __func__, __LINE__);
        return 0;
    }

    Dims YoloLayerPlugin::getOutputDimensions(int index, const Dims* inputs, int nbInputDims)
    {
	    printf("%s:%d\n", __func__, __LINE__);
        //output the result to channel
        int totalsize = mMaxOutObject * sizeof(Detection) / sizeof(float);

        return Dims3(totalsize + 1, 1, 1);
    }

    // Set plugin namespace
    void YoloLayerPlugin::setPluginNamespace(const char* pluginNamespace)
    {
	    printf("%s:%d\n", __func__, __LINE__);
        mPluginNamespace = pluginNamespace;
    }

    const char* YoloLayerPlugin::getPluginNamespace() const
    {
	    printf("%s:%d\n", __func__, __LINE__);
        return mPluginNamespace;
    }

    // Return the DataType of the plugin output at the requested index
    DataType YoloLayerPlugin::getOutputDataType(int index, const nvinfer1::DataType* inputTypes, int nbInputs) const
    {
	    printf("%s:%d\n", __func__, __LINE__);
        return DataType::kFLOAT;
    }

    // Return true if output tensor is broadcast across a batch.
    bool YoloLayerPlugin::isOutputBroadcastAcrossBatch(int outputIndex, const bool* inputIsBroadcasted, int nbInputs) const
    {
	    printf("%s:%d\n", __func__, __LINE__);
        return false;
    }

    // Return true if plugin can use input that is broadcast across batch without replication.
    bool YoloLayerPlugin::canBroadcastInputAcrossBatch(int inputIndex) const
    {
	    printf("%s:%d\n", __func__, __LINE__);
        return false;
    }

    void YoloLayerPlugin::configurePlugin(const PluginTensorDesc* in, int nbInput, const PluginTensorDesc* out, int nbOutput)
    {
	    printf("%s:%d\n", __func__, __LINE__);
    }

    // Attach the plugin object to an execution context and grant the plugin the access to some context resource.
    void YoloLayerPlugin::attachToContext(cudnnContext* cudnnContext, cublasContext* cublasContext, IGpuAllocator* gpuAllocator)
    {
	    printf("%s:%d\n", __func__, __LINE__);
    }

    // Detach the plugin object from its execution context.
    void YoloLayerPlugin::detachFromContext() {}

    const char* YoloLayerPlugin::getPluginType() const
    {
	    printf("%s:%d\n", __func__, __LINE__);
        return "YoloLayer_TRT";
    }

    const char* YoloLayerPlugin::getPluginVersion() const
    {
	    printf("%s:%d\n", __func__, __LINE__);
        return "1";
    }

    void YoloLayerPlugin::destroy()
    {
	    printf("%s:%d\n", __func__, __LINE__);
        delete this;
    }

    // Clone the plugin
    IPluginV2IOExt* YoloLayerPlugin::clone() const
    {
	    printf("%s:%d\n", __func__, __LINE__);
        YoloLayerPlugin* p = new YoloLayerPlugin(mClassCount, mYoloV5NetWidth, mYoloV5NetHeight, mMaxOutObject, mYoloKernel);
        p->setPluginNamespace(mPluginNamespace);
        return p;
    }
#if 1
    __device__ float Logist(float data) { return 1.0f / (1.0f + expf(-data)); };

    __global__ void CalDetection(const float *input, float *output, int noElements,
        const int netwidth, const int netheight, int maxoutobject, int yoloWidth, int yoloHeight, const float anchors[CHECK_COUNT * 2], int classes, int outputElem)
    {

        int idx = threadIdx.x + blockDim.x * blockIdx.x;
        if (idx >= noElements) return;

        int total_grid = yoloWidth * yoloHeight;
        int bnIdx = idx / total_grid;
        idx = idx - total_grid * bnIdx;
        int info_len_i = 5 + classes;
        const float* curInput = input + bnIdx * (info_len_i * total_grid * CHECK_COUNT);

        for (int k = 0; k < 3; ++k) {
            float box_prob = Logist(curInput[idx + k * info_len_i * total_grid + 4 * total_grid]);
            if (box_prob < IGNORE_THRESH) continue;
            int class_id = 0;
            float max_cls_prob = 0.0;
            for (int i = 5; i < info_len_i; ++i) {
                float p = Logist(curInput[idx + k * info_len_i * total_grid + i * total_grid]);
                if (p > max_cls_prob) {
                    max_cls_prob = p;
                    class_id = i - 5;
                }
            }
            float *res_count = output + bnIdx * outputElem;
            int count = (int)atomicAdd(res_count, 1);
            if (count >= maxoutobject) return;
            char* data = (char *)res_count + sizeof(float) + count * sizeof(Detection);
            Detection* det = (Detection*)(data);

            int row = idx / yoloWidth;
            int col = idx % yoloWidth;

            //Location
            // pytorch:
            //  y = x[i].sigmoid()
            //  y[..., 0:2] = (y[..., 0:2] * 2. - 0.5 + self.grid[i].to(x[i].device)) * self.stride[i]  # xy
            //  y[..., 2:4] = (y[..., 2:4] * 2) ** 2 * self.anchor_grid[i]  # wh 
            //  X: (sigmoid(tx) + cx)/FeaturemapW *  netwidth 
            det->bbox[0] = (col - 0.5f + 2.0f * Logist(curInput[idx + k * info_len_i * total_grid + 0 * total_grid])) * netwidth / yoloWidth;
            det->bbox[1] = (row - 0.5f + 2.0f * Logist(curInput[idx + k * info_len_i * total_grid + 1 * total_grid])) * netheight / yoloHeight;

            // W: (Pw * e^tw) / FeaturemapW * netwidth  
            // v5: https://github.com/ultralytics/yolov5/issues/471
            det->bbox[2] = 2.0f * Logist(curInput[idx + k * info_len_i * total_grid + 2 * total_grid]);
            det->bbox[2] = det->bbox[2] * det->bbox[2] * anchors[2 * k];
            det->bbox[3] = 2.0f * Logist(curInput[idx + k * info_len_i * total_grid + 3 * total_grid]);
            det->bbox[3] = det->bbox[3] * det->bbox[3] * anchors[2 * k + 1];
            det->conf = box_prob * max_cls_prob;
            det->class_id = class_id;
        }
    }
#endif

#if 0
    __device__ float Logist(float data){ return 1.0f / (1.0f + expf(-data)); };

    __global__ void CalDetection(const float *input, float *output,int noElements, 
            int yoloWidth,int yoloHeight,const float anchors[CHECK_COUNT*2],int classes,int outputElem) {
 
        int idx = threadIdx.x + blockDim.x * blockIdx.x;
        if (idx >= noElements) return;

        int total_grid = yoloWidth * yoloHeight;
        int bnIdx = idx / total_grid;
        idx = idx - total_grid*bnIdx;
        int info_len_i = 5 + classes;
        const float* curInput = input + bnIdx * (info_len_i * total_grid * CHECK_COUNT);

        for (int k = 0; k < 3; ++k) {
            float box_prob = Logist(curInput[idx + k * info_len_i * total_grid + 4 * total_grid]);
            if (box_prob < IGNORE_THRESH) continue;
            int class_id = 0;
            float max_cls_prob = 0.0;
            for (int i = 5; i < info_len_i; ++i) {
                float p = Logist(curInput[idx + k * info_len_i * total_grid + i * total_grid]);
                if (p > max_cls_prob) {
                    max_cls_prob = p;
                    class_id = i - 5;
                }
            }
            float *res_count = output + bnIdx*outputElem;
            int count = (int)atomicAdd(res_count, 1);
            if (count >= MAX_OUTPUT_BBOX_COUNT) return;
            char* data = (char *)res_count + sizeof(float) + count * sizeof(Detection);
            Detection* det =  (Detection*)(data);

            int row = idx / yoloWidth;
            int col = idx % yoloWidth;

            //Location
            det->bbox[0] = (col - 0.5f + 2.0f * Logist(curInput[idx + k * info_len_i * total_grid + 0 * total_grid])) * INPUT_W / yoloWidth;
            det->bbox[1] = (row - 0.5f + 2.0f * Logist(curInput[idx + k * info_len_i * total_grid + 1 * total_grid])) * INPUT_H / yoloHeight;
            det->bbox[2] = 2.0f * Logist(curInput[idx + k * info_len_i * total_grid + 2 * total_grid]);
            det->bbox[2] = det->bbox[2] * det->bbox[2] * anchors[2*k];
            det->bbox[3] = 2.0f * Logist(curInput[idx + k * info_len_i * total_grid + 3 * total_grid]);
            det->bbox[3] = det->bbox[3] * det->bbox[3] * anchors[2*k + 1];
            det->conf = box_prob * max_cls_prob;
            det->class_id = class_id;
        }
    }
#endif
    
    void YoloLayerPlugin::forwardGpu(const float *const * inputs, float* output, cudaStream_t stream, int batchSize)
    {
	    printf("%s:%d\n", __func__, __LINE__);
        int outputElem = 1 + mMaxOutObject * sizeof(Detection) / sizeof(float);
        for (int idx = 0; idx < batchSize; ++idx) {
            CUDA_CHECK(cudaMemset(output + idx * outputElem, 0, sizeof(float)));
        }
        int numElem = 0;
        for (unsigned int i = 0; i < mYoloKernel.size(); ++i)
        {
            const auto& yolo = mYoloKernel[i];
            numElem = yolo.width*yolo.height*batchSize; 
            if (numElem < mThreadCount)
                mThreadCount = numElem;

            //printf("Net: %d  %d \n", mYoloV5NetWidth, mYoloV5NetHeight);
#if 1
	    //CalDetection << < (yolo.width*yolo.height*batchSize + mThreadCount - 1) / mThreadCount, mThreadCount >> >
     	    CalDetection<<< (yolo.width*yolo.height*batchSize + mThreadCount - 1) / mThreadCount, mThreadCount, 0, stream>>>
                (inputs[i], output, numElem, mYoloV5NetWidth, mYoloV5NetHeight, mMaxOutObject, yolo.width, yolo.height, (float *)mAnchor[i], mClassCount, outputElem);
#else
     	    CalDetection<<< (yolo.width*yolo.height*batchSize + mThreadCount - 1) / mThreadCount, mThreadCount, 0, stream>>>
                (inputs[i], output, numElem, yolo.width, yolo.height, (float *)mAnchor[i], mClassCount, outputElem);
#endif
        }
    }
#if 0
    void YoloLayerPlugin::forwardGpu(const float *const * inputs, float* output, cudaStream_t stream, int batchSize) {

        int outputElem = 1 + MAX_OUTPUT_BBOX_COUNT * sizeof(Detection) / sizeof(float);

        for(int idx = 0 ; idx < batchSize; ++idx) {
            CUDA_CHECK(cudaMemset(output + idx*outputElem, 0, sizeof(float)));
        }
        int numElem = 0;
        for (unsigned int i = 0; i < mYoloKernel.size(); ++i)
        {
            const auto& yolo = mYoloKernel[i];
            numElem = yolo.width*yolo.height*batchSize;
            if (numElem < mThreadCount)
                mThreadCount = numElem;
            CalDetection<<< (yolo.width*yolo.height*batchSize + mThreadCount - 1) / mThreadCount, mThreadCount, 0, stream>>>
                (inputs[i], output, numElem, yolo.width, yolo.height, (float *)mAnchor[i], mClassCount, outputElem);
        }

    }
    
#endif

    int YoloLayerPlugin::enqueue(int batchSize, const void*const * inputs, void** outputs, void* workspace, cudaStream_t stream)
    {
	    printf("%s:%d\n", __func__, __LINE__);
        forwardGpu((const float *const *)inputs, (float*)outputs[0], stream, batchSize);
        return 0;
    }

    PluginFieldCollection YoloPluginCreator::mFC{};
    std::vector<PluginField> YoloPluginCreator::mPluginAttributes;

    YoloPluginCreator::YoloPluginCreator()
    {
	    printf("%s:%d\n", __func__, __LINE__);
        mPluginAttributes.clear();

        mFC.nbFields = mPluginAttributes.size();
        mFC.fields = mPluginAttributes.data();
    }

    const char* YoloPluginCreator::getPluginName() const
    {
	    printf("%s:%d\n", __func__, __LINE__);
        return "YoloLayer_TRT";
    }

    const char* YoloPluginCreator::getPluginVersion() const
    {
	    printf("%s:%d\n", __func__, __LINE__);
        return "1";
    }

    const PluginFieldCollection* YoloPluginCreator::getFieldNames()
    {
	    printf("%s:%d\n", __func__, __LINE__);
        return &mFC;
    }

    IPluginV2IOExt* YoloPluginCreator::createPlugin(const char* name, const PluginFieldCollection* fc)
    {
	    printf("%s:%d\n", __func__, __LINE__);
        int class_count = 80;
        int input_w = 416;
        int input_h = 416;
        int max_output_object_count = 1000;
        std::vector<Yolo::YoloKernel> yolo_kernels(3);

        const PluginField* fields = fc->fields;
        for (int i = 0; i < fc->nbFields; i++) {
            if (strcmp(fields[i].name, "netdata") == 0) {
                assert(fields[i].type == PluginFieldType::kFLOAT32);
                int *tmp = (int*)(fields[i].data);
                class_count = tmp[0];
                input_w = tmp[1];
                input_h = tmp[2];
                max_output_object_count = tmp[3];
            } else if (strstr(fields[i].name, "yolodata") != NULL) {
                assert(fields[i].type == PluginFieldType::kFLOAT32);
                int *tmp = (int*)(fields[i].data);
                YoloKernel kernel;
                kernel.width = tmp[0];
                kernel.height = tmp[1];
                for (int j = 0; j < fields[i].length - 2; j++) {
                    kernel.anchors[j] = tmp[j + 2];
                }
                yolo_kernels[2 - (fields[i].name[8] - '1')] = kernel;
            }
        }
        YoloLayerPlugin* obj = new YoloLayerPlugin(class_count, input_w, input_h, max_output_object_count, yolo_kernels);
        obj->setPluginNamespace(mNamespace.c_str());
        return obj;
    }

    IPluginV2IOExt* YoloPluginCreator::deserializePlugin(const char* name, const void* serialData, size_t serialLength)
    {
	    printf("%s:%d\n", __func__, __LINE__);
        // This object will be deleted when the network is destroyed, which will
        // call YoloLayerPlugin::destroy()
        YoloLayerPlugin* obj = new YoloLayerPlugin(serialData, serialLength);
        obj->setPluginNamespace(mNamespace.c_str());
        return obj;
    }
}

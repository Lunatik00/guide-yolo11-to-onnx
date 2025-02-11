// https://github.com/microsoft/onnxruntime/blob/v1.8.2/csharp/test/Microsoft.ML.OnnxRuntime.EndToEndTests.Capi/CXX_Api_Sample.cpp
// https://github.com/microsoft/onnxruntime/blob/v1.8.2/include/onnxruntime/core/session/onnxruntime_cxx_api.h
#include <onnxruntime_cxx_api.h>

#include <opencv2/dnn/dnn.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include <chrono>
#include <cmath>
#include <exception>
#include <fstream>
#include <iostream>
#include <limits>
#include <numeric>
#include <string>

#include <opencv2/core/core.hpp>

#include <fstream>
#include <iterator>
#include <vector>
#include <memory>
#include <opencv2/opencv.hpp>


#include <argparse/argparse.hpp>

/**
 * @brief Define names based depends on Unicode path support
 */
#define tcout                  std::cout
#define file_name_t            std::string
#define imread_t               cv::imread
#define NMS_THRESH 0.5
#define BBOX_CONF_THRESH 0.1

static const int INPUT_W = 640;
static const int INPUT_H = 640;
static const int NUM_CLASSES = 1;

template <typename T>
T vectorProduct(const std::vector<T>& v)
{
    return accumulate(v.begin(), v.end(), 1, std::multiplies<T>());
}

/**
 * @brief Operator overloading for printing vectors
 * @tparam T
 * @param os
 * @param v
 * @return std::ostream&
 */
template <typename T>
std::ostream& operator<<(std::ostream& os, const std::vector<T>& v)
{
    os << "[";
    for (int i = 0; i < v.size(); ++i)
    {
        os << v[i];
        if (i != v.size() - 1)
        {
            os << ", ";
        }
    }
    os << "]";
    return os;
}

/**
 * @brief Print ONNX tensor data type
 * https://github.com/microsoft/onnxruntime/blob/rel-1.6.0/include/onnxruntime/core/session/onnxruntime_c_api.h#L93
 * @param os
 * @param type
 * @return std::ostream&
 */
std::ostream& operator<<(std::ostream& os,
                         const ONNXTensorElementDataType& type)
{
    switch (type)
    {
        case ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED:
            os << "undefined";
            break;
        case ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT:
            os << "float";
            break;
        case ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8:
            os << "uint8_t";
            break;
        case ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8:
            os << "int8_t";
            break;
        case ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT16:
            os << "uint16_t";
            break;
        case ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_INT16:
            os << "int16_t";
            break;
        case ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32:
            os << "int32_t";
            break;
        case ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64:
            os << "int64_t";
            break;
        case ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_STRING:
            os << "std::string";
            break;
        case ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL:
            os << "bool";
            break;
        case ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16:
            os << "float16";
            break;
        case ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE:
            os << "double";
            break;
        case ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT32:
            os << "uint32_t";
            break;
        case ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT64:
            os << "uint64_t";
            break;
        case ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_COMPLEX64:
            os << "float real + float imaginary";
            break;
        case ONNXTensorElementDataType::
            ONNX_TENSOR_ELEMENT_DATA_TYPE_COMPLEX128:
            os << "double real + float imaginary";
            break;
        case ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_BFLOAT16:
            os << "bfloat16";
            break;
        default:
            break;
    }

    return os;
}


cv::Mat static_resize(cv::Mat& img) {
    // YOLO11 needs the image to be RGB with float values between 0 and 1
    cv::Mat img_rgb;
    cv::cvtColor(img,img_rgb,cv::COLOR_BGR2RGB );
    cv::Mat img_rezised;
    cv::resize(img_rgb,img_rezised,cv::Size(INPUT_H,INPUT_W));
    cv::Mat out;
    img_rezised.convertTo(out, CV_32F);
    return out/255;
}

struct Results
{
    float x1;
    float y1;
    float x2;
    float y2;
    int label;
    float prob;
};

std::vector<Results> sort_onnx_nms_output(std::vector<float> onnx_output_values,float prob_threshold)
{
    std::vector<Results> out;
    const int n_outputs = onnx_output_values.size()/6;
    for (int i = 0; i < n_outputs; i++)
    {
        int ix1 = i*6;
        int iy1 = i*6 +1;
        int ix2 = i*6 +2;
        int iy2 = i*6 +3;
        int iprob = i*6+4;
        int ilabel = i*6+5;
        if (onnx_output_values[iprob]>=prob_threshold && isfinite(onnx_output_values[iprob]))
        {
            Results element;
            element.x1 = onnx_output_values[ix1]/INPUT_W;
            element.x2 = onnx_output_values[ix2]/INPUT_W;
            element.y1 = onnx_output_values[iy1]/INPUT_H;
            element.y2 = onnx_output_values[iy2]/INPUT_H;
            element.prob = onnx_output_values[iprob];
            element.label = onnx_output_values[ilabel];
            out.push_back(element);
        }
    }
    return out;
}





int main(int argc, char* argv[])
{   
    argparse::ArgumentParser program("Yolo11n-onnx");
    program.add_argument("onnx_file")
        .help("The onnx file to use");
    program.add_argument("image")
        .help("The image file to use");
    program.add_argument("--cuda")
        .help("To use cuda as the provider for onnx")
        .flag();

    try {
        program.parse_args(argc, argv);
    }
    catch (const std::exception& err) {
        std::cerr << err.what() << std::endl;
        std::cerr << program;
        return 1;
    }

    if (program["--cuda"] == true)
    {
        std::cout << "Inference Execution Provider: CUDA" << std::endl;
    }
    else
    {
        std::cout << "Inference Execution Provider: CPU" << std::endl;
    }


    std::string instanceName{"onnx-inference"};

    std::string modelFilepath = program.get<std::string>("onnx_file");
    std::string imageFilepath = program.get<std::string>("image");

    // Set some onnx environment values
    Ort::Env env(OrtLoggingLevel::ORT_LOGGING_LEVEL_WARNING,
                 instanceName.c_str());
    // Prepare session options for inference
    Ort::SessionOptions sessionOptions;
    sessionOptions.SetIntraOpNumThreads(1);
    // Add some optimizations
    sessionOptions.SetGraphOptimizationLevel(
        GraphOptimizationLevel::ORT_ENABLE_EXTENDED);
    // Start onnx session
    Ort::Session session(env, modelFilepath.c_str(), sessionOptions);

    // create allocator to retrive data
    Ort::AllocatorWithDefaultOptions allocator;

    size_t numInputNodes = session.GetInputCount();
    size_t numOutputNodes = session.GetOutputCount();

    // Get input name, it is retrieved as a string but changed to char* because the char* is required for inference
    const std::string _inputName = session.GetInputNameAllocated(0, allocator).get();
    const char* inputName= _inputName.c_str();
    // std::cout << "Input Name: " << inputName << std::endl;

    Ort::TypeInfo inputTypeInfo = session.GetInputTypeInfo(0);
    auto inputTensorInfo = inputTypeInfo.GetTensorTypeAndShapeInfo();

    ONNXTensorElementDataType inputType = inputTensorInfo.GetElementType();
    //std::cout << "Input Type: " << inputType << std::endl;

    // Get inputs dimensions IMPORTANT: if the model has variable inputs the use of 
    // this data should be examinated because the default values will give an error
    std::vector<int64_t> inputDims = inputTensorInfo.GetShape();
    std::cout << "Input Dimensions: " << inputDims << std::endl;

    // Repeat the process for the output
    const std::string _outputName = session.GetOutputNameAllocated(0, allocator).get();
    const char* outputName= _outputName.c_str();
    std::cout << "Output Name: " << outputName << std::endl;

    Ort::TypeInfo outputTypeInfo = session.GetOutputTypeInfo(0);
    auto outputTensorInfo = outputTypeInfo.GetTensorTypeAndShapeInfo();

    ONNXTensorElementDataType outputType = outputTensorInfo.GetElementType();
    //std::cout << "Output Type: " << outputType << std::endl;

    // Get output dimensions IMPORTANT: if the model has variable input the use of 
    // this data should be examinated because the default values will give an error.
    // Even when the output should be fixed even for variable inputs
    std::vector<int64_t> outputDims = outputTensorInfo.GetShape();
    std::cout << "Output Dimensions: " << outputDims << std::endl;


    // open image and prepare it for inference
    cv::Mat imageBGR = cv::imread(imageFilepath, cv::ImreadModes::IMREAD_COLOR);
    cv::Mat preprocessedImage;
    cv::Mat resizedImage = static_resize(imageBGR);
    cv::dnn::blobFromImage(resizedImage, preprocessedImage);

    size_t inputTensorSize = vectorProduct(inputDims);
    // std::cout << "inputTensorSize: " << inputTensorSize << std::endl;
    // IMPORTANT: This is where it fails if the input values are variable, 
    // seek another way to create the vector that will not have this problem
    std::vector<float> inputTensorValues(inputTensorSize);
    inputTensorValues.assign(preprocessedImage.begin<float>(),
                             preprocessedImage.end<float>());

    size_t outputTensorSize = vectorProduct(outputDims);
    std::vector<float> outputTensorValues(outputTensorSize);

    // Prepare variables for inference
    std::vector<const char*> inputNames{inputName};
    std::vector<const char*> outputNames{outputName};
    std::vector<Ort::Value> inputTensors;
    std::vector<Ort::Value> outputTensors;

    Ort::MemoryInfo memoryInfo = Ort::MemoryInfo::CreateCpu(
        OrtAllocatorType::OrtArenaAllocator, OrtMemType::OrtMemTypeDefault);
    // Assign data for input
    inputTensors.push_back(Ort::Value::CreateTensor<float>(
        memoryInfo, inputTensorValues.data(), inputTensorSize, inputDims.data(),
        inputDims.size()));
    // Finish preparations for output
    outputTensors.push_back(Ort::Value::CreateTensor<float>(
        memoryInfo, outputTensorValues.data(), outputTensorSize,
        outputDims.data(), outputDims.size()));

    // Run inference
    session.Run(Ort::RunOptions{nullptr}, inputNames.data(),
                inputTensors.data(), 1, outputNames.data(),
                outputTensors.data(), 1);

    // Sort results
    std::vector<Results> output_onnx = sort_onnx_nms_output(outputTensorValues,BBOX_CONF_THRESH);
    // std::cout << "response: " << output_onnx.size()<< std::endl;

    // Open image to show inference
    cv::Mat image = imread_t(imageFilepath);
    // Select a color to use for bounding box in BGR UINT8
    const cv::Scalar color = cv::Scalar(255, 0, 0);
    const int thickness = 5; 
    // Iterate over the result vector
    for (int i = 0; i <  output_onnx.size(); i++)
    {
        // Scale xy values and cast them to uint8
        int x1 = static_cast<int>(output_onnx[i].x1 * (image.cols*1.0)) ;
        int y1 = static_cast<int>(output_onnx[i].y1 * (image.rows*1.0)) ;
        int x2 = static_cast<int>(output_onnx[i].x2 * (image.cols*1.0)) ;
        int y2 = static_cast<int>(output_onnx[i].y2 * (image.rows*1.0)) ;
        // Create points
        cv::Point p1(x1,y1);
        cv::Point p2(x2,y2);
        // Draw bounding box
        cv::rectangle(image, p1, p2, 
              color, 
              thickness);
    }
    // Save image
    cv::imwrite("pills-out.jpg",image);
}
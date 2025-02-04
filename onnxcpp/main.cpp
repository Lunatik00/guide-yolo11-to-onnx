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

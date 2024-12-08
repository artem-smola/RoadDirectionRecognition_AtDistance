#pragma once
#include <opencv2/dnn_superres.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include "reader.hpp"

const double kInitIncorrectMetricValue = -1.0;

class MetricsEvaluator {
public:
    double GetIoU(const cv::Mat &marking_res, const cv::Mat &ground_truth);
    double GetAccuracy(const cv::Mat &marking_res, const cv::Mat &ground_truth);
};
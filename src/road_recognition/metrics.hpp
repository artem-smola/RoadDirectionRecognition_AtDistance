#pragma once
#include <opencv2/dnn_superres.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

double CompareByIoU(const cv::Mat &marking_res, const cv::Mat &ground_truth);

double CompareByAccuracy(const cv::Mat &marking_res,
                         const cv::Mat &ground_truth);
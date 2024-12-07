#pragma once
#include <opencv2/dnn_superres.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

double EvaluationByIoU(const cv::Mat &marking_res, const cv::Mat &ground_truth);

double EvaluationByAccuracy(const cv::Mat &marking_res,
                            const cv::Mat &ground_truth);
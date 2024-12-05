#pragma once
#include "../../TwinLiteNet-onnxruntime/include/twinlitenet_onnxruntime.hpp"
#include <iostream>
#include <opencv2/dnn_superres.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <string>
#include <vector>

const int kRoiWidth = 640;
const int kRoiHeight = 360;

cv::Rect GetRoiRect(const cv::Mat &img);

void MarkLaneAtDistance(cv::Mat &img, cv::Rect);
#pragma once
#include "twinlitenet_onnxruntime.hpp"
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

std::vector<cv::Point> GetLanePixels(const cv::Mat &img);

bool IsBlackPixel(cv::Vec3b pixel);
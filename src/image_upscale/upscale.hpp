#pragma once
#include "enums.hpp"
#include <iostream>
#include <opencv2/dnn_superres.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <string>
#include <vector>

class Upscale {
public:
  virtual void Execute(cv::Mat &img) = 0;
};

class UpscaleESPCN : public Upscale {
public:
  void Execute(cv::Mat &img);
};
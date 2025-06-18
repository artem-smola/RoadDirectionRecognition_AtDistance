#pragma once
#include "enums.hpp"
#include <opencv2/dnn_superres.hpp>

class Upscale {
public:
  virtual void Execute(cv::Mat &img) = 0;
};

class UpscaleESPCN : public Upscale {
public:
  void Execute(cv::Mat &img);
};
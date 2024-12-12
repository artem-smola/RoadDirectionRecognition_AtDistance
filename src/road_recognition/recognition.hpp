#pragma once
#include "enums.hpp"
#include "twinlitenet_onnxruntime.hpp"
#include <iostream>
#include <opencv2/dnn_superres.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <string>
#include <vector>

const int kRoiWidth = 640;
const int kRoiHeight = 360;

using Points = std::vector<cv::Point>;

class DistantRoadRecognition {
public:
  virtual Points MarkLaneAtDistance(cv::Mat &img) = 0;
  virtual ~DistantRoadRecognition() {}
  virtual void SetRoi(const cv::Mat &img) = 0;

protected:
  DistantRoadRecognition();
  virtual void MarkLane(cv::Mat &img) = 0;
  virtual Points GetLanePixels(const cv::Mat &img) = 0;
  cv::Rect roi_;
};

class DistantRoadRecognitionTwinLiteNet : public DistantRoadRecognition {
public:
  DistantRoadRecognitionTwinLiteNet();
  virtual ~DistantRoadRecognitionTwinLiteNet();
  Points MarkLaneAtDistance(cv::Mat &img) override;
  void SetRoi(const cv::Mat &img) override;

protected:
  TwinLiteNet model_;
  void MarkLane(cv::Mat &img) override;
  Points GetLanePixels(const cv::Mat &img) override;
};

class DistantRoadRecognitionTwinLiteNetUpscale
    : public DistantRoadRecognitionTwinLiteNet {
public:
  DistantRoadRecognitionTwinLiteNetUpscale();
  Points MarkLaneAtDistance(cv::Mat &img) override;

private:
  void Upscale(cv::Mat &img);
};

bool IsBlackPixel(cv::Vec3b pixel);
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
  virtual void SetRoi(cv::Mat img) = 0;

protected:
  DistantRoadRecognition();
  cv::Rect roi_;

private:
  virtual void MarkLane(cv::Mat &img) = 0;

  virtual Points GetLanePixels(const cv::Mat &img) = 0;
};

class DistantRoadRecognitionTwinLiteNet : public DistantRoadRecognition {
public:
  DistantRoadRecognitionTwinLiteNet();
  ~DistantRoadRecognitionTwinLiteNet();

  Points MarkLaneAtDistance(cv::Mat &img);
  void SetRoi(cv::Mat img);

private:
  void MarkLane(cv::Mat &img);
  Points GetLanePixels(const cv::Mat &img);
};

bool IsBlackPixel(cv::Vec3b pixel);
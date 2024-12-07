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

class DistantRoadDirectionRecognition {
public:
  virtual std::vector<Points> MarkLaneAtDistance(std::string out_path = "") = 0;

protected:
  DistantRoadDirectionRecognition(std::string path_to_photos,
                                  PhotoExtension photos_extension);
  virtual ~DistantRoadDirectionRecognition();
  std::string path_to_photos_;
  PhotoExtension photos_extension_;
  std::vector<cv::String> photos_;

private:
  virtual cv::Rect GetRoiRect() = 0;

  virtual void MarkLane(cv::Mat &img) = 0;

  virtual Points GetLanePixels(const cv::Mat &img) = 0;
};

class DistantRoadDirectionRecognitionTwinLiteNet
    : public DistantRoadDirectionRecognition {
public:
  DistantRoadDirectionRecognitionTwinLiteNet(std::string path_to_photos,
                                             PhotoExtension photos_extension);
  ~DistantRoadDirectionRecognitionTwinLiteNet();

  std::vector<Points> MarkLaneAtDistance(std::string out_path = "");

private:
  cv::Rect GetRoiRect();
  void MarkLane(cv::Mat &img);
  Points GetLanePixels(const cv::Mat &img);
};

bool IsBlackPixel(cv::Vec3b pixel);
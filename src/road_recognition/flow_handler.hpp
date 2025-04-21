#pragma once
#include "reader.hpp"
#include <opencv2/opencv.hpp>

class FlowHandler {
public:
  FlowHandler(Reader &reader, const size_t &min_count,
              const size_t &min_variance);
  void SetRoi(const cv::Rect &ROI);
  cv::Rect GetRoi();
  bool Next();
  cv::Mat GetSampleFrame();
  cv::Mat GetCurrentFrame();
  size_t GetSize();

private:
  Reader &reader_;
  cv::Scalar ROI_color_;
  cv::Rect ROI_;
  cv::Mat frame_;
  cv::Mat frame_gray_;
  bool is_just_updated_ = false;
  size_t min_count_;
  size_t min_variance_;

  size_t num_without_variance_ = 0;

  std::vector<cv::Point2f> points_;
};

cv::Rect SetRoiRect(cv::Mat img);
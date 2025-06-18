#pragma once
#include "reader.hpp"
#include <opencv2/opencv.hpp>

using RectSize = std::pair<size_t, size_t>;

class FlowHandler {
public:
  FlowHandler(Reader &reader, size_t min_count, size_t min_variance);
  void SetRoi(const cv::Rect &ROI);
  void SetStandardRoiPosition(const cv::Point &default_ROI_position);
  void SetRoiLowerBound(size_t lower_bound);
  cv::Rect GetRoi();
  cv::Point GetDefaultRoiPosition();
  bool Next();
  cv::Mat GetSampleFrame();
  cv::Mat GetCurrentFrame();
  size_t GetSize();

private:
  Reader &reader_;
  cv::Rect ROI_;
  cv::Mat frame_;
  cv::Mat frame_gray_;
  bool is_just_updated_ = false;
  size_t min_count_;
  size_t min_variance_;
  cv::Point default_ROI_position_;
  size_t lower_bound_;

  size_t num_without_variance_ = 0;

  std::vector<cv::Point2f> points_;
  double correction_ = 1;
};
double GetCompressionToFitRectOnScreen(cv::Size rect_size,
                                       cv::Size screen_size);

cv::Rect SetRoiSize(cv::Rect ROI, const cv::Mat &img);
cv::Rect SetInitialRoiPosition(cv::Rect ROI, const cv::Mat &img);
cv::Point SetDefaultRoiPosition(cv::Rect ROI, const cv::Mat &img);
size_t SetRoiLowerBound(const cv::Mat &img);
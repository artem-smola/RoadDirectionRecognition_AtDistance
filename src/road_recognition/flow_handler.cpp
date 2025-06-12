#include "flow_handler.hpp"
#include "constant.hpp"
#include <algorithm>
#include <cmath>
#include <iostream>
#include <vector>

FlowHandler::FlowHandler(Reader &reader, size_t min_count, size_t min_variance)
    : reader_(reader), min_count_(min_count), min_variance_(min_variance),
      lower_bound_(Constant::ROI_border) {
  frame_ = reader_.Read();

  double ROI_compression = GetCompressionToFitRectOnScreen(
      cv::Size(Constant::default_ROI_width, Constant::default_ROI_height),
      cv::Size(frame_.cols, frame_.rows));
  ROI_.width = static_cast<int>(
      static_cast<double>(Constant::default_ROI_width) / ROI_compression);
  ROI_.height = static_cast<int>(
      static_cast<double>(Constant::default_ROI_height) / ROI_compression);
  ROI_.x =
      (Constant::default_ROI_x < frame_.cols) ? Constant::default_ROI_x : 0;
  ROI_.y =
      (Constant::default_ROI_y < frame_.rows) ? Constant::default_ROI_y : 0;

  default_ROI_position_ = cv::Point(ROI_.x, ROI_.y);
  cv::cvtColor(frame_, frame_gray_, cv::COLOR_BGR2GRAY);
  cv::goodFeaturesToTrack(frame_gray_, points_, 250, 0.01, 20, cv::Mat(), 3,
                          false, 0.04);
}

void FlowHandler::SetRoi(const cv::Rect &ROI) { ROI_ = ROI; }
void FlowHandler::SetStandardRoiPosition(
    const cv::Point &default_ROI_position) {
  default_ROI_position_ = default_ROI_position;
}
void FlowHandler::SetRoiLowerBound(size_t lower_bound) {
  lower_bound_ = lower_bound;
}

cv::Rect FlowHandler::GetRoi() { return ROI_; }
cv::Point FlowHandler::GetDefaultRoiPosition() { return default_ROI_position_; }

cv::Mat FlowHandler::GetSampleFrame() { return reader_.GetSample(); }

cv::Mat FlowHandler::GetCurrentFrame() { return frame_; }

bool FlowHandler::Next() {
  if (reader_.GetCurrentIndex() >= reader_.GetSize()) {
    return false;
  }
  cv::Mat next_frame = reader_.Read();
  cv::Mat next_frame_gray;
  cv::cvtColor(next_frame, next_frame_gray, cv::COLOR_BGR2GRAY);

  std::vector<uchar> status;
  std::vector<float> err;
  cv::TermCriteria criteria = cv::TermCriteria(
      cv::TermCriteria::COUNT + cv::TermCriteria::EPS, 30, 0.03);

  std::vector<cv::Point2f> next_points;

  cv::calcOpticalFlowPyrLK(frame_gray_, next_frame_gray, points_, next_points,
                           status, err, cv::Size(21, 21), 3, criteria);

  frame_ = next_frame;
  frame_gray_ = next_frame_gray;

  size_t count = std::count(status.begin(), status.end(), 1);
  if (count < min_count_) {
    cv::goodFeaturesToTrack(frame_gray_, points_, 250, 0.01, 20, cv::Mat(), 3,
                            false, 0.04);
    std::cout << "Tracking points refreshed!" << std::endl;
    return true;
  }

  std::vector<cv::Point2f> good_points;
  std::vector<int> diffs;
  good_points.reserve(count);
  diffs.reserve(count);
  for (size_t i = 0; i < points_.size(); i++) {
    if (status[i] == 1) {
      good_points.push_back(next_points[i]);
      int diff = next_points[i].x - points_[i].x;
      if (std::abs(diff) >= min_variance_) {
        diffs.push_back(diff);
      }
    }
  }
  size_t turn_indicators = diffs.size();
  int variance = 0;

  size_t min_num_of_indicators =
      static_cast<size_t>(static_cast<double>(good_points.size()) *
                          Constant::min_share_of_indicators);
  if (turn_indicators >= min_num_of_indicators) {
    std::sort(diffs.begin(), diffs.end());
    variance = static_cast<int>(static_cast<double>(diffs[diffs.size() / 2]) *
                                Constant::variance_smoothing_factor);
    ROI_.x = std::clamp(ROI_.x - variance, 0,
                        static_cast<int>(frame_.cols - ROI_.width));

    if (ROI_.y + Constant::default_ROI_height < lower_bound_) {
      ROI_.y = (ROI_.y + std::abs(variance) <= lower_bound_)
                   ? (ROI_.y + std::abs(variance))
                   : lower_bound_;
    }
    num_without_variance_ = 0;
    return true;
  }
  if (++num_without_variance_ >= Constant::num_frame_to_reset_ROI) {
    ROI_.x = default_ROI_position_.x;
    ROI_.y = default_ROI_position_.y;
  }
  return true;
}

size_t FlowHandler::GetSize() { return reader_.GetSize(); }

cv::Rect SetRoiSize(cv::Rect ROI, const cv::Mat &img) {
  double compression = GetCompressionToFitRectOnScreen(
      cv::Size(img.cols, img.rows), cv::Size(Constant::default_screen_width,
                                             Constant::default_screen_height));
  while (true) {
    cv::Mat marked_img = img.clone();
    cv::rectangle(marked_img, ROI, cv::Scalar(0, 244, 0), 4);

    cv::Mat resized_img;
    cv::resize(marked_img, resized_img,
               cv::Size(static_cast<int>(static_cast<double>(marked_img.cols) /
                                         compression),
                        static_cast<int>(static_cast<double>(marked_img.rows) /
                                         compression)));
    cv::imshow("Roi size setting", resized_img);

    int key = cv::waitKey(25);

    switch (key) {
    case '-':
      ROI.width = (ROI.width * 10) / 11;
      ROI.height = (ROI.height * 10) / 11;
      break;
    case '+':
      ROI.width = (ROI.width * 11) / 10;
      ROI.height = (ROI.height * 11) / 10;
      break;
    case static_cast<char>(13):
      return ROI;
    }
  }
}

cv::Rect SetInitialRoiPosition(cv::Rect ROI, const cv::Mat &img) {
  double compression = GetCompressionToFitRectOnScreen(
      cv::Size(img.cols, img.rows), cv::Size(Constant::default_screen_width,
                                             Constant::default_screen_height));
  while (true) {
    cv::Mat marked_img = img.clone();
    cv::rectangle(marked_img, ROI, cv::Scalar(0, 244, 0), 4);

    cv::Mat resized_img;
    cv::resize(marked_img, resized_img,
               cv::Size(static_cast<int>(static_cast<double>(marked_img.cols) /
                                         compression),
                        static_cast<int>(static_cast<double>(marked_img.rows) /
                                         compression)));
    cv::imshow("Roi position setting", resized_img);

    int key = cv::waitKey(25);

    switch (key) {
    case 'w':
      if (ROI.y > 10)
        ROI.y -= 10;
      break;
    case 'a':
      if (ROI.x > 10)
        ROI.x -= 10;
      break;
    case 's':
      if (ROI.y < img.rows - 10)
        ROI.y += 10;
      break;
    case 'd':
      if (ROI.x < img.cols - 10)
        ROI.x += 10;
      break;
    case static_cast<char>(13):
      return ROI;
    }
  }
}

cv::Point SetDefaultRoiPosition(cv::Rect ROI, const cv::Mat &img) {
  double compression = GetCompressionToFitRectOnScreen(
      cv::Size(img.cols, img.rows), cv::Size(Constant::default_screen_width,
                                             Constant::default_screen_height));
  while (true) {
    cv::Mat marked_img = img.clone();
    cv::rectangle(marked_img, ROI, cv::Scalar(0, 244, 0), 4);

    cv::Mat resized_img;
    cv::resize(marked_img, resized_img,
               cv::Size(static_cast<int>(static_cast<double>(marked_img.cols) /
                                         compression),
                        static_cast<int>(static_cast<double>(marked_img.rows) /
                                         compression)));
    cv::imshow("Roi default position setting", resized_img);

    int key = cv::waitKey(25);

    switch (key) {
    case 'w':
      if (ROI.y > 10)
        ROI.y -= 10;
      break;
    case 'a':
      if (ROI.x > 10)
        ROI.x -= 10;
      break;
    case 's':
      if (ROI.y < img.rows - 10)
        ROI.y += 10;
      break;
    case 'd':
      if (ROI.x < img.cols - 10)
        ROI.x += 10;
      break;
    case static_cast<char>(13):
      return cv::Point(ROI.x, ROI.y);
    }
  }
}

size_t SetRoiLowerBound(const cv::Mat &img) {
  double compression = GetCompressionToFitRectOnScreen(
      cv::Size(img.cols, img.rows), cv::Size(Constant::default_screen_width,
                                             Constant::default_screen_height));
  size_t bound = (Constant::ROI_border <= img.rows) ? Constant::ROI_border
                                                    : (img.rows / 2);
  while (true) {
    cv::Mat marked_img = img.clone();
    cv::line(marked_img, cv::Point(0, bound), cv::Point(img.cols, bound),
             cv::Scalar(0, 244, 0), 4);
    cv::Mat resized_img;
    cv::resize(marked_img, resized_img,
               cv::Size(static_cast<int>(static_cast<double>(marked_img.cols) /
                                         compression),
                        static_cast<int>(static_cast<double>(marked_img.rows) /
                                         compression)));
    cv::imshow("Roi default position setting", resized_img);

    int key = cv::waitKey(25);
    switch (key) {
    case 'w':
      if (bound > 10)
        bound -= 10;
      break;
    case 's':
      if (bound < img.rows - 10)
        bound += 10;
      break;
    case static_cast<int>(Keys::enter):
      return bound;
    }
  }
}

double GetCompressionToFitRectOnScreen(cv::Size rect_size,
                                       cv::Size screen_size) {
  double compression_x =
      static_cast<double>(rect_size.width) / screen_size.width;
  double compression_y =
      static_cast<double>(rect_size.height) / screen_size.height;
  return std::max(compression_x, compression_y);
}
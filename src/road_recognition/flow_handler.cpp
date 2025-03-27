#include "flow_handler.hpp"
#include "constant.hpp"
#include <cmath>
#include <iostream>
#include <vector>

FlowHandler::FlowHandler(const std::string &dir_path, PhotoExtension extension,
                         size_t min_count, size_t min_variance)
    : reader_(dir_path, extension), ROI_color_(0, 0, 244),
      min_count_(min_count), min_variance_(min_variance) {
  ROI_.x = Constant::default_ROI_x;
  ROI_.y = Constant::default_ROI_y;
  ROI_.width = Constant::default_ROI_width;
  ROI_.height = Constant::default_ROI_height;

  frame_ = reader_.Read();
  cv::cvtColor(frame_, frame_gray_, cv::COLOR_BGR2GRAY);
  cv::goodFeaturesToTrack(frame_gray_, points_, 100, 0.3, 7, cv::Mat(), 7,
                          false, 0.04);
}

void FlowHandler::SetRoi(const cv::Rect &ROI) { ROI_ = ROI; }

cv::Rect FlowHandler::GetRoi() { return ROI_; }

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
      cv::TermCriteria::COUNT + cv::TermCriteria::EPS, 10, 0.03);

  std::vector<cv::Point2f> next_points;

  cv::calcOpticalFlowPyrLK(frame_gray_, next_frame_gray, points_, next_points,
                           status, err, cv::Size(15, 15), 2, criteria);

  frame_ = next_frame;
  frame_gray_ = next_frame_gray;

  size_t count = std::count(status.begin(), status.end(), 1);
  if (count < min_count_) {
    cv::goodFeaturesToTrack(frame_gray_, points_, 100, 0.3, 7, cv::Mat(), 7,
                            false, 0.04);
    std::cout << "Tracking points refreshed!" << std::endl;
    return true;
  }

  std::vector<cv::Point2f> good_points;
  size_t turn_indicators = 0;
  int sum_variance = 0;
  int variance = 0;
  for (size_t i = 0; i < points_.size(); i++) {
    if (status[i] == 1) {
      good_points.push_back(next_points[i]);
      int diff = next_points[i].x - points_[i].x;
      if (std::abs(diff) >= min_variance_) {
        sum_variance += diff;
        turn_indicators++;
      }
    }
  }

  size_t min_num_of_indicators =
      static_cast<size_t>(static_cast<double>(good_points.size()) *
                          Constant::min_share_of_indicators);
  if (turn_indicators >= min_num_of_indicators) {
    variance = sum_variance / static_cast<int>(turn_indicators);
    ROI_.x -= variance;
    if (ROI_.y + Constant::default_ROI_height < Constant::ROI_border) {
      ROI_.y = (ROI_.y + std::abs(variance) <= Constant::ROI_border)
                   ? (ROI_.y + std::abs(variance))
                   : Constant::ROI_border;
    }
    num_without_variance_ = 0;
    return true;
  }
  if (++num_without_variance_ >= Constant::num_frame_to_reset_ROI) {
    cv::Point point =
        cv::Point(Constant::default_ROI_x, Constant::default_ROI_y);
    ROI_.x = point.x;
    ROI_.y = point.y;
  }
  return true;
}

size_t FlowHandler::GetSize() { return reader_.GetSize(); }

cv::Rect SetRoiRect(cv::Mat img) {
  cv::Rect ROI;
  ROI.x = Constant::default_ROI_x;
  ROI.y = Constant::default_ROI_y;
  ROI.width = Constant::default_ROI_width;
  ROI.height = Constant::default_ROI_height;

  while (true) {
    cv::Mat marked_img = img.clone();
    cv::rectangle(marked_img, ROI, cv::Scalar(0, 244, 0), 4);

    cv::Mat resized_img;
    cv::resize(marked_img, resized_img,
               cv::Size(marked_img.cols / Constant::compression,
                        marked_img.rows / Constant::compression));
    cv::imshow("Roi setter", resized_img);

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
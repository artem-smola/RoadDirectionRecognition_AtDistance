#include "metrics.hpp"
#include "recognition.hpp"

double MetricsEvaluator::GetIoU(const cv::Mat &ground_truth,
                                const cv::Mat &marking_res) {
  if (marking_res.rows != kRoiHeight || ground_truth.rows != kRoiHeight ||
      marking_res.cols != kRoiWidth || ground_truth.cols != kRoiWidth) {
    throw std::invalid_argument("Error: Incorrect image size");
  }
  int num_of_intersection = 0;
  int num_of_union = 0;
  for (int y = 0; y < marking_res.rows; y++) {
    for (int x = 0; x < marking_res.cols; x++) {
      bool is_marking_res_pixel_black =
          IsBlackPixel(marking_res.at<cv::Vec3b>(y, x));
      bool is_ground_truth_pixel_black =
          IsBlackPixel(ground_truth.at<cv::Vec3b>(y, x));
      if (!is_marking_res_pixel_black || !is_ground_truth_pixel_black) {
        num_of_union++;
      }
      if (!is_marking_res_pixel_black && !is_ground_truth_pixel_black) {
        num_of_intersection++;
      }
    }
  }
  if (num_of_union == 0) {
    return 1;
  }
  return static_cast<double>(num_of_intersection) /
         static_cast<double>(num_of_union);
}

double MetricsEvaluator::GetAccuracy(const cv::Mat &marking_res,
                                     const cv::Mat &ground_truth) {
  if (marking_res.rows != kRoiHeight || ground_truth.rows != kRoiHeight ||
      marking_res.cols != kRoiWidth || ground_truth.cols != kRoiWidth) {
    throw std::invalid_argument("Error: Incorrect image size");
  }
  int num_true_positive = 0;
  int num_true_negative = 0;
  int num_false_positive = 0;
  int num_false_negative = 0;
  for (int y = 0; y < marking_res.rows; y++) {
    for (int x = 0; x < marking_res.cols; x++) {
      bool is_marking_res_pixel_black =
          IsBlackPixel(marking_res.at<cv::Vec3b>(y, x));
      bool is_ground_truth_pixel_black =
          IsBlackPixel(ground_truth.at<cv::Vec3b>(y, x));
      if (is_marking_res_pixel_black) {
        if (is_ground_truth_pixel_black) {
          num_true_negative++;
        } else {
          num_false_negative++;
        }
      } else {
        if (!is_ground_truth_pixel_black) {
          num_true_positive++;
        } else {
          num_false_positive++;
        }
      }
    }
  }
  return static_cast<double>(num_true_positive + num_true_negative) /
         static_cast<double>(num_true_positive + num_true_negative +
                             num_false_positive + num_false_negative);
}
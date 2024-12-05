#include "recognition.hpp"

cv::Rect GetRoiRect(const cv::Mat &img) {
  cv::Rect roi;
  roi.x = (img.size().width * 7) / 18;
  roi.y = (img.size().height * 5) / 9;
  int remain_width = img.size().width - roi.x;
  int remain_height = img.size().height - roi.y;
  int correction = 1;
  if (remain_width < kRoiWidth || remain_height < kRoiHeight) {
    int correction_width = (kRoiWidth + remain_width - 1) / remain_width;
    int correction_height = (kRoiHeight + remain_height - 1) / remain_height;
    correction = (correction_width > correction_height) ? correction_width
                                                        : correction_height;
  }
  roi.width = kRoiWidth / correction;
  roi.height = kRoiHeight / correction;
  return roi;
}

void MarkLaneAtDistance(cv::Mat &img, cv::Rect roi) {
  img = img(roi);
  if (img.size().width != kRoiWidth || img.size().height != kRoiHeight) {
    resize(img, img, cv::Size(kRoiWidth, kRoiHeight));
  }
  TwinLiteNet twin_lite_net("../../TwinLiteNet-onnxruntime/models/best.onnx");
  cv::Mat da_out, ll_out;
  twin_lite_net.Infer(img, da_out, ll_out);
  img.setTo(cv::Scalar(0, 0, 0));
  img.setTo(cv::Scalar(0, 255, 255), ll_out);
}

std::vector<cv::Point> GetLanePixels(const cv::Mat &img) {
  std::vector<cv::Point> lane_pixels;

  for (int y = 0; y < img.rows; y++) {
    for (int x = 0; x < img.cols; x++) {
      cv::Vec3b pixel = img.at<cv::Vec3b>(y, x);
      if (pixel[0] != 0 || pixel[1] != 0 || pixel[2] != 0) {
        lane_pixels.emplace_back(x, y);
      }
    }
  }
  return lane_pixels;
}

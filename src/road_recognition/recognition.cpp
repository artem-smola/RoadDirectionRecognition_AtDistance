#include "recognition.hpp"

DistantRoadRecognition::DistantRoadRecognition() : roi_() {}

DistantRoadRecognitionTwinLiteNet::DistantRoadRecognitionTwinLiteNet()
    : DistantRoadRecognition(), model_("../../TwinLiteNet-onnxruntime/models/best.onnx") {}

DistantRoadRecognitionTwinLiteNet::~DistantRoadRecognitionTwinLiteNet() {}

void DistantRoadRecognitionTwinLiteNet::SetRoi(const cv::Mat &img) {
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
  roi_ = roi;
}

void DistantRoadRecognitionTwinLiteNet::MarkLane(cv::Mat &img) {
  if (img.size().width != kRoiWidth || img.size().height != kRoiHeight) {
    cv::resize(img, img, cv::Size(kRoiWidth, kRoiHeight));
  }
  cv::Mat da_out, ll_out;
  model_.Infer(img, da_out, ll_out);
  img.setTo(cv::Scalar(0, 0, 0));
  img.setTo(cv::Scalar(180, 130, 70), ll_out);
}

Points DistantRoadRecognitionTwinLiteNet::GetLanePixels(const cv::Mat &img) {
  Points lane_pixels;
  for (int y = 0; y < img.rows; y++) {
    for (int x = 0; x < img.cols; x++) {
      cv::Vec3b pixel = img.at<cv::Vec3b>(y, x);
      if (!IsBlackPixel(pixel)) {
        lane_pixels.emplace_back(x, y);
      }
    }
  }
  return lane_pixels;
}

Points DistantRoadRecognitionTwinLiteNet::MarkLaneAtDistance(cv::Mat &img) {
  cv::Mat origin_img = img.clone();
  img = img(roi_);
  MarkLane(img);
  cv::imshow("Origin road image", origin_img);
  cv::imshow("Marked distant part of the road", img);
  cv::waitKey(30);
  Points lane_pixels = GetLanePixels(img);
  return lane_pixels;
}

bool IsBlackPixel(cv::Vec3b pixel) {
  return (pixel[0] == 0 && pixel[1] == 0 && pixel[2] == 0);
}

DistantRoadRecognitionTwinLiteNetUpscale::
    DistantRoadRecognitionTwinLiteNetUpscale()
    : DistantRoadRecognitionTwinLiteNet() {}

void DistantRoadRecognitionTwinLiteNetUpscale::Upscale(cv::Mat &img) {
  std::string path = "../models/ESPCN_x4.pb";
  std::string model_name = "espcn";
  int scale = 4;
  cv::dnn_superres::DnnSuperResImpl sr;
  sr.readModel(path);
  sr.setModel(model_name, scale);
  sr.upsample(img, img);
}

Points
DistantRoadRecognitionTwinLiteNetUpscale::MarkLaneAtDistance(cv::Mat &img) {
  cv::Mat origin_img = img.clone();
  img = img(roi_);
  Upscale(img);
  MarkLane(img);
  cv::imshow("Origin road image", origin_img);
  cv::imshow("Marked distant part of the road", img);
  cv::waitKey(30);
  Points lane_pixels = GetLanePixels(img);
  return lane_pixels;
}
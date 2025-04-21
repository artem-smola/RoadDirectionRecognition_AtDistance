#include "recognition.hpp"
#include "constant.hpp"

DistantRoadRecognition::DistantRoadRecognition() : roi_() {}

DistantRoadRecognitionTwinLiteNet::DistantRoadRecognitionTwinLiteNet()
    : DistantRoadRecognition(),
      model_("/home/artem/practice_4_sem/RoadDirectionRecognition_AtDistance/"
             "TwinLiteNet-onnxruntime/models/best.onnx") {}

DistantRoadRecognitionTwinLiteNet::~DistantRoadRecognitionTwinLiteNet() {}

void DistantRoadRecognitionTwinLiteNet::SetRoi(const cv::Mat &img) {
  cv::Rect roi;
  roi.x = (img.size().width * 7) / 18;
  roi.y = (img.size().height * 5) / 9;
  int remain_width = img.size().width - roi.x;
  int remain_height = img.size().height - roi.y;
  int correction = 1;
  if (remain_width < Constant::default_ROI_width ||
      remain_height < Constant::default_ROI_height) {
    int correction_width =
        (Constant::default_ROI_width + remain_width - 1) / remain_width;
    int correction_height =
        (Constant::default_ROI_height + remain_height - 1) / remain_height;
    correction = (correction_width > correction_height) ? correction_width
                                                        : correction_height;
  }
  roi.width = Constant::default_ROI_width / correction;
  roi.height = Constant::default_ROI_height / correction;
  roi_ = roi;
}

void DistantRoadRecognitionTwinLiteNet::MarkLane(cv::Mat &img) {
  if (img.size().width != Constant::default_ROI_width ||
      img.size().height != Constant::default_ROI_height) {
    cv::resize(
        img, img,
        cv::Size(Constant::default_ROI_width, Constant::default_ROI_height));
  }
  cv::Mat da_out, ll_out;
  model_.Infer(img, da_out, ll_out);
  img.setTo(cv::Scalar(0, 0, 0));
  img.setTo(cv::Scalar(180, 130, 70), ll_out);
}

PtrPoints DistantRoadRecognitionTwinLiteNet::GetLanePixels(const cv::Mat &img) {
  auto lane_pixels = std::make_unique<std::vector<cv::Point>>();
  lane_pixels->reserve(img.rows * img.cols);

  for (int y = 0; y < img.rows; y++) {
    for (int x = 0; x < img.cols; x++) {
      cv::Vec3b pixel = img.at<cv::Vec3b>(y, x);
      if (!IsBlackPixel(pixel)) {
        lane_pixels->emplace_back(x, y);
      }
    }
  }

  return lane_pixels;
}

PtrPoints DistantRoadRecognitionTwinLiteNet::MarkLaneAtDistance(cv::Mat &img) {
  cv::Mat origin_img = img.clone();
  MarkLane(img);
  return GetLanePixels(img);
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

PtrPoints
DistantRoadRecognitionTwinLiteNetUpscale::MarkLaneAtDistance(cv::Mat &img) {
  cv::Mat origin_img = img.clone();
  Upscale(img);
  MarkLane(img);
  // cv::imshow("Origin road image", origin_img);
  // cv::imshow("Marked distant part of the road", img);
  // cv::waitKey(1);
  return GetLanePixels(img);
}
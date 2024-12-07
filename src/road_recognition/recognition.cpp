#include "recognition.hpp"

DistantRoadDirectionRecognition::DistantRoadDirectionRecognition(
    std::string path_to_photos, PhotoExtension photos_extension)
    : path_to_photos_(path_to_photos), photos_extension_(photos_extension) {
  if (path_to_photos_.empty()) {
    throw std::invalid_argument("Path to photos cannot be empty.");
  }
  cv::glob(path_to_photos + "*" +
               kExtensionNames[static_cast<int>(photos_extension_)],
           photos_);
}
DistantRoadDirectionRecognition::~DistantRoadDirectionRecognition() {
  photos_.clear();
}

DistantRoadDirectionRecognitionTwinLiteNet::
    DistantRoadDirectionRecognitionTwinLiteNet(std::string path_to_photos,
                                               PhotoExtension photos_extension)
    : DistantRoadDirectionRecognition(path_to_photos, photos_extension) {}

DistantRoadDirectionRecognitionTwinLiteNet::
    ~DistantRoadDirectionRecognitionTwinLiteNet() {}

cv::Rect DistantRoadDirectionRecognitionTwinLiteNet::GetRoiRect() {
  cv::Rect roi;
  cv::Mat img = cv::imread(photos_[0]);
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

void DistantRoadDirectionRecognitionTwinLiteNet::MarkLane(cv::Mat &img) {
  if (img.size().width != kRoiWidth || img.size().height != kRoiHeight) {
    resize(img, img, cv::Size(kRoiWidth, kRoiHeight));
  }
  TwinLiteNet twin_lite_net("../../TwinLiteNet-onnxruntime/models/best.onnx");
  cv::Mat da_out, ll_out;
  twin_lite_net.Infer(img, da_out, ll_out);
  img.setTo(cv::Scalar(0, 0, 0));
  img.setTo(cv::Scalar(0, 255, 255), ll_out);
}

Points
DistantRoadDirectionRecognitionTwinLiteNet::GetLanePixels(const cv::Mat &img) {
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

std::vector<Points>
DistantRoadDirectionRecognitionTwinLiteNet::MarkLaneAtDistance(
    std::string out_path) {
  cv::Rect roi = GetRoiRect();
  std::vector<Points> result;
  for (int i = 0; i < static_cast<int>(photos_.size()); i++) {
    cv::Mat origin_img = cv::imread(photos_[i]);
    cv::Mat marked_img = origin_img.clone();
    marked_img = marked_img(roi);
    MarkLane(marked_img);
    if (out_path != "") {
      cv::imwrite(out_path + "photo" + std::to_string(i) +
                      kExtensionNames[static_cast<int>(photos_extension_)],
                  marked_img);
    }
    cv::imshow("Origin road image", origin_img);
    cv::imshow("Marked distant part of the road", marked_img);
    cv::waitKey(300);
    Points lane_pixels = GetLanePixels(marked_img);
    result.push_back(lane_pixels);
  }
  return result;
}

bool IsBlackPixel(cv::Vec3b pixel) {
  return (pixel[0] == 0 && pixel[1] == 0 && pixel[2] == 0);
}

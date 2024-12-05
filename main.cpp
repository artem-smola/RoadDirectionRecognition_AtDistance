#include "src/road_recognition/recognition.hpp"
#include <filesystem>

int main() {
  std::filesystem::path cwd = std::filesystem::current_path();
  std::string dir_path = cwd.parent_path().string() + "/images/";

  std::cout << dir_path << std::endl;

  std::vector<cv::String> image_files;
  cv::glob(dir_path + "*.jpg", image_files);
  cv::Mat one_frame = cv::imread(image_files[0]);
  cv::Rect roi = GetRoiRect(one_frame);
  for (cv::String str : image_files) {
    cv::Mat img = cv::imread(str);
    if (img.empty()) {
      std::cerr << "Error: Could not read the image " << str << std::endl;
      continue;
    }
    cv::Mat roi_img = img.clone();
    roi_img = roi_img(roi);
    cv::Mat lane_img = img.clone();
    MarkLaneAtDistance(lane_img, roi);
    cv::imshow("Original road image", img);
    cv::imshow("Area at the distance", roi_img);
    cv::imshow("Detected road lane at the distance", lane_img);
    cv::waitKey(2000);
  }

  return 0;
}
#include "writer.hpp"

FolderWriter::FolderWriter(const std::string &folder_path,
                           PhotoExtension photo_extension)
    : folder_path_(folder_path), photo_extension_(photo_extension),
      photo_index_(0) {}

void FolderWriter::Write(const cv::Mat &img) {
  std::string photo_path = folder_path_ + "Photo" +
                           std::to_string(photo_index_) +
                           kExtensionNames[static_cast<int>(photo_extension_)];
  cv::imwrite(photo_path, img);
  photo_index_++;
}

VideoWriter::VideoWriter(const std::string &video_path, const cv::Mat &sample,
                         double fps)
    : video_path_(video_path), fps_(fps) {
  int fourcc = cv::VideoWriter::fourcc('M', 'J', 'P', 'G');
  cv::Size frameSize = sample.size();
  bool isColor = (sample.channels() > 1);
  writer_.open(video_path_, fourcc, fps_, frameSize, isColor);
}

VideoWriter::~VideoWriter() { writer_.release(); }

void VideoWriter::Write(const cv::Mat &img) {
  if (!writer_.isOpened()) {
    throw std::runtime_error("Video writer is not opened");
  }
  writer_.write(img);
}

#include "reader.hpp"

size_t Reader::GetSize() { return size_; }

FolderReader::FolderReader(std::string folder_path,
                           PhotoExtension photo_extension)
    : photo_extension_(photo_extension), current_index_(0) {
  cv::glob(folder_path + "*" +
               kExtensionNames[static_cast<int>(photo_extension_)],
           photos_paths_);
  size_ = photos_paths_.size();
  sample_ = cv::imread(photos_paths_[0]);
}

FolderReader::~FolderReader() { photos_paths_.clear(); }

cv::Mat FolderReader::Read() {
  if (current_index_ < size_) {
    current_index_++;
    return cv::imread(photos_paths_[current_index_]);
  }
  return cv::Mat();
}

cv::Mat FolderReader::GetSample() { return sample_; }

VideoReader::VideoReader(std::string video_path) {
  video_capture_.open(video_path);
  size_ = video_capture_.get(cv::CAP_PROP_FRAME_COUNT);
  video_capture_ >> sample_;
  video_capture_.set(cv::CAP_PROP_POS_FRAMES, 0);
}

VideoReader::~VideoReader() {}

cv::Mat VideoReader::Read() {
  cv::Mat img;
  if (video_capture_.isOpened()) {
    video_capture_ >> img;
  }
  return img;
}

cv::Mat VideoReader::GetSample() { return sample_; }
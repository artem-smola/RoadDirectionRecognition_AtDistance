#pragma once
#include "enums.hpp"
#include <iostream>
#include <opencv2/dnn_superres.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <string>
#include <vector>

class Reader {
public:
  virtual ~Reader() {}
  virtual cv::Mat Read() = 0;
  size_t GetSize();
  virtual cv::Mat GetSample() = 0;

protected:
  size_t size_;
  cv::Mat sample_;
};

class FolderReader : public Reader {
public:
  FolderReader(std::string folder_path, PhotoExtension photo_extension);
  ~FolderReader();
  cv::Mat Read();
  cv::Mat GetSample();

private:
  std::vector<cv::String> photos_paths_;
  size_t current_index_;
  PhotoExtension photo_extension_;
};

class VideoReader : public Reader {
public:
  VideoReader(std::string video_path);
  ~VideoReader();
  cv::Mat Read();
  cv::Mat GetSample();

private:
  cv::VideoCapture video_capture_;
};
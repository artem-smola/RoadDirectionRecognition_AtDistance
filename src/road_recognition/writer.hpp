#pragma once
#include "enums.hpp"
#include <iostream>
#include <opencv2/dnn_superres.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <string>
#include <vector>

class Writer {
public:
  virtual ~Writer() {}
  virtual void Write(const cv::Mat &img) = 0;
};

class FolderWriter : public Writer {
public:
  FolderWriter(const std::string &folder_path, PhotoExtension photo_extension);
  void Write(const cv::Mat &img) override;

private:
  std::string folder_path_;
  PhotoExtension photo_extension_;
  size_t photo_index_;
};

class VideoWriter : public Writer {
public:
  VideoWriter(const std::string &video_path, const cv::Mat &sample, double fps);
  ~VideoWriter();
  void Write(const cv::Mat &img) override;

private:
  cv::VideoWriter writer_;
  std::string video_path_;
  double fps_;
};
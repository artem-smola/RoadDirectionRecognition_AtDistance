#include "manager.hpp"

Manager::Manager(Reader &reader) : reader_(reader) {}

DistantRoadRecognitionManager::DistantRoadRecognitionManager(
    Reader &reader, DistantRoadRecognition &marker)
    : Manager(reader), reader_(reader), marker_(marker) {
  cv::Mat sample = reader_.GetSample();
  marker_.SetRoi(sample);
}

DistantRoadRecognitionManager::~DistantRoadRecognitionManager() {
  pixels_vectors_.clear();
}

void DistantRoadRecognitionManager::Process() {
  for (size_t i = 0; i < reader_.GetSize(); i++) {
    cv::Mat img = reader_.Read();
    Points lane_pixels = marker_.MarkLaneAtDistance(img);
    pixels_vectors_.push_back(lane_pixels);
  }
}
std::vector<Points> DistantRoadRecognitionManager::GetPixelsVectors() {
  return pixels_vectors_;
}

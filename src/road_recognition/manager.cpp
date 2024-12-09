#include "manager.hpp"

DistantRoadRecognitionManager::DistantRoadRecognitionManager(
    Reader &reader, DistantRoadRecognition &marker, Writer &writer)
    : Manager(), reader_(reader), marker_(marker), writer_(writer) {
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
    writer_.Write(img);
    pixels_vectors_.push_back(lane_pixels);
  }
}
std::vector<Points> DistantRoadRecognitionManager::GetPixelsVectors() {
  return pixels_vectors_;
}

MetricsManager::MetricsManager(Reader &marking_res_reader,
                               Reader &ground_truth_reader,
                               MetricsEvaluator &evaluator)
    : marking_res_reader_(marking_res_reader),
      ground_truth_reader_(ground_truth_reader), evaluator_(evaluator),
      res_IoU_(kInitIncorrectMetricValue),
      res_accuracy_(kInitIncorrectMetricValue) {}

void MetricsManager::Process() {
  double sum_IoU = 0;
  double sum_accuracy = 0;
  for (size_t i = 0; i < marking_res_reader_.GetSize(); i++) {
    sum_IoU += evaluator_.GetIoU(marking_res_reader_.Read(),
                                 ground_truth_reader_.Read());
    sum_accuracy += evaluator_.GetAccuracy(marking_res_reader_.Read(),
                                           ground_truth_reader_.Read());
  }
  res_IoU_ = sum_IoU / static_cast<double>(marking_res_reader_.GetSize());
  res_accuracy_ =
      sum_accuracy / static_cast<double>(marking_res_reader_.GetSize());
}

double MetricsManager::GetIoU() {
  if (res_IoU_ == kInitIncorrectMetricValue) {
    Process();
  }
  return res_IoU_;
}

double MetricsManager::GetAccuracy() {
  if (res_accuracy_ == kInitIncorrectMetricValue) {
    Process();
  }
  return res_accuracy_;
}

UpscaleManager::UpscaleManager(Reader &reader, Upscale &improver,
                               Writer &writer)
    : reader_(reader), improver_(improver), writer_(writer) {}

void UpscaleManager::Process() {
  for (size_t i = 0; i < reader_.GetSize(); i++) {
    cv::Mat img = reader_.Read();
    improver_.Execute(img);
    writer_.Write(img);
  }
}
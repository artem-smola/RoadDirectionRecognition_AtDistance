#include "manager.hpp"

Manager::Manager(Reader &reader, Writer &writer)
    : reader_(reader), writer_(writer) {}

DistantRoadRecognitionManager::DistantRoadRecognitionManager(
    Reader &reader, Writer &writer, DistantRoadRecognition &marker)
    : Manager(reader, writer), marker_(marker) {
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

UpscaleManager::UpscaleManager(Reader &reader, Writer &writer,
                               Upscale &improver)
    : Manager(reader, writer), improver_(improver) {}

void UpscaleManager::Process() {
  for (size_t i = 0; i < reader_.GetSize(); i++) {
    cv::Mat img = reader_.Read();
    improver_.Execute(img);
    writer_.Write(img);
  }
}

RoiManager::RoiManager(Reader &reader, Writer &writer)
    : Manager(reader, writer), roi_() {}

void RoiManager::SetRoi(const cv::Mat &img) {
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

void RoiManager::Process() {
  if (roi_.empty()) {
    SetRoi(reader_.GetSample());
  }
  for (size_t i = 0; i < reader_.GetSize(); i++) {
    cv::Mat img = reader_.Read();
    img = img(roi_);
    writer_.Write(img);
  }
}
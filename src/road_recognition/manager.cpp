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
                               Reader &ground_truth_reader)
    : marking_res_reader_(marking_res_reader),
      ground_truth_reader_(ground_truth_reader),
      res_IoU_(kInitIncorrectMetricValue),
      res_accuracy_(kInitIncorrectMetricValue) {}

double MetricsManager::EvaluateIoU(const cv::Mat &marking_res,
                                   const cv::Mat &ground_truth) {
  if (marking_res.rows != kRoiHeight || ground_truth.rows != kRoiHeight ||
      marking_res.cols != kRoiWidth || ground_truth.cols != kRoiWidth) {
    throw std::invalid_argument("Error: Incorrect image size");
  }
  int num_of_intersection = 0;
  int num_of_union = 0;
  for (int y = 0; y < marking_res.rows; y++) {
    for (int x = 0; x < marking_res.cols; x++) {
      bool is_marking_res_pixel_black =
          IsBlackPixel(marking_res.at<cv::Vec3b>(y, x));
      bool is_ground_truth_pixel_black =
          IsBlackPixel(ground_truth.at<cv::Vec3b>(y, x));
      if (!is_marking_res_pixel_black || !is_ground_truth_pixel_black) {
        num_of_union++;
      }
      if (!is_marking_res_pixel_black && !is_ground_truth_pixel_black) {
        num_of_intersection++;
      }
    }
  }
  if (num_of_union == 0) {
    return 1;
  }
  return static_cast<double>(num_of_intersection) /
         static_cast<double>(num_of_union);
}

double MetricsManager::EvaluateAccuracy(const cv::Mat &marking_res,
                                        const cv::Mat &ground_truth) {
  if (marking_res.rows != kRoiHeight || ground_truth.rows != kRoiHeight ||
      marking_res.cols != kRoiWidth || ground_truth.cols != kRoiWidth) {
    throw std::invalid_argument("Error: Incorrect image size");
  }
  int num_true_positive = 0;
  int num_true_negative = 0;
  int num_false_positive = 0;
  int num_false_negative = 0;
  for (int y = 0; y < marking_res.rows; y++) {
    for (int x = 0; x < marking_res.cols; x++) {
      bool is_marking_res_pixel_black =
          IsBlackPixel(marking_res.at<cv::Vec3b>(y, x));
      bool is_ground_truth_pixel_black =
          IsBlackPixel(ground_truth.at<cv::Vec3b>(y, x));
      if (is_marking_res_pixel_black) {
        if (is_ground_truth_pixel_black) {
          num_true_negative++;
        } else {
          num_false_negative++;
        }
      } else {
        if (!is_ground_truth_pixel_black) {
          num_true_positive++;
        } else {
          num_false_positive++;
        }
      }
    }
  }
  return static_cast<double>(num_true_positive + num_true_negative) /
         static_cast<double>(num_true_positive + num_true_negative +
                             num_false_positive + num_false_negative);
}

void MetricsManager::Process() {
  double sum_IoU = 0;
  double sum_accuracy = 0;
  for (size_t i = 0; i < marking_res_reader_.GetSize(); i++) {
    auto marking_res = marking_res_reader_.Read();
    auto ground_truth = ground_truth_reader_.Read();
    sum_IoU +=
        EvaluateIoU(marking_res, ground_truth);
    sum_accuracy += EvaluateAccuracy(marking_res,
                                     ground_truth);
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
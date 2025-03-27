#pragma once
#include "../image_upscale/upscale.hpp"
#include "flow_handler.hpp"
#include "reader.hpp"
#include "recognition.hpp"
#include "writer.hpp"

class Manager {
public:
  virtual ~Manager() {}
  virtual void Process() = 0;

protected:
  Manager(Reader &reader, Writer &writer);
  Reader &reader_;
  Writer &writer_;
};

class DistantRoadRecognitionManager : public Manager {
public:
  DistantRoadRecognitionManager(Reader &reader, Writer &writer,
                                DistantRoadRecognition &marker);
  ~DistantRoadRecognitionManager();

  void Process() override;
  std::vector<Points> GetPixelsVectors();

private:
  DistantRoadRecognition &marker_;
  std::vector<Points> pixels_vectors_;
};

class UpscaleManager : public Manager {
public:
  UpscaleManager(Reader &reader, Writer &writer, Upscale &improver);
  void Process() override;

private:
  Upscale &improver_;
};

class RoiManager : public Manager {
public:
  RoiManager(Reader &reader, Writer &writer);
  void Process() override;

private:
  void SetRoi(const cv::Mat &sample);
  cv::Rect roi_;
};

class FPSManagerDistantRoadRecognition {
public:
  FPSManagerDistantRoadRecognition(DistantRoadRecognition &marker,
                                   const std::string &path_to_txt);
  void Process();

private:
  DistantRoadRecognition &marker_;
  std::string path_to_txt_;
};

class FPSManagerTwinLiteNet {
public:
  FPSManagerTwinLiteNet(const std::string &path_to_txt);
  void Process();

private:
  std::string path_to_txt_;
};

class AutomatedRoiManagerDistantRoadRecognition {
public:
  AutomatedRoiManagerDistantRoadRecognition(DistantRoadRecognition &marker);
  double Process();

private:
  DistantRoadRecognition &marker_;
};

class MetricsManager {
public:
  MetricsManager(std::string path_to_txt);
  void Process();
  double GetIoU();
  double GetAccuracy();

private:
  double EvaluateIoU(const cv::Mat &marking_res, const cv::Mat &ground_truth);
  double EvaluateAccuracy(const cv::Mat &marking_res,
                          const cv::Mat &ground_truth);
  void PrintHistogram(const std::vector<double> &data, int num_bins);

  std::string path_to_txt_;
  double res_IoU_;
  double res_accuracy_;
};

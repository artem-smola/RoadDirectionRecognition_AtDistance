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

class ManagerUpscale : public Manager {
public:
  ManagerUpscale(Reader &reader, Writer &writer, Upscale &improver);
  void Process() override;

private:
  Upscale &improver_;
};

class ManagerDRR : public Manager {
public:
  ManagerDRR(Reader &reader, Writer &writer, DistantRoadRecognition &marker);
  void Process() override;
  std::vector<Points> GetPixelsVectors();

private:
  DistantRoadRecognition &marker_;
  FlowHandler handler_;
  std::vector<Points> pixels_vectors_;
};

class FPSManagerDRR {
public:
  FPSManagerDRR(DistantRoadRecognition &marker, const std::string &path_to_txt);
  void Process();

private:
  DistantRoadRecognition &marker_;
  std::string path_to_txt_;
};

class FPSManagerTLN {
public:
  FPSManagerTLN(const std::string &path_to_txt);
  void Process();

private:
  std::string path_to_txt_;
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

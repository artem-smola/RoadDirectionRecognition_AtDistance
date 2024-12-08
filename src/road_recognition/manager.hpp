#pragma once
#include "reader.hpp"
#include "recognition.hpp"
#include "metrics.hpp"

class Manager {
public:
  virtual ~Manager() {}
  Manager() {}
  virtual void Process() = 0;
};

class DistantRoadRecognitionManager : public Manager {
public:
  DistantRoadRecognitionManager(Reader &reader, DistantRoadRecognition &marker);
  ~DistantRoadRecognitionManager();

  void Process();
  std::vector<Points> GetPixelsVectors();

private:
  Reader &reader_;
  DistantRoadRecognition &marker_;
  std::vector<Points> pixels_vectors_;
};

class MetricsManager : public Manager {
public:
  MetricsManager(Reader &marking_res_reader, Reader &ground_truth_reader, MetricsEvaluator &evaluator);
  void Process();
  double GetIoU();
  double GetAccuracy();
private:
  Reader &marking_res_reader_;
  Reader &ground_truth_reader_;
  MetricsEvaluator evaluator_;
  double res_IoU_;
  double res_accuracy_;
};
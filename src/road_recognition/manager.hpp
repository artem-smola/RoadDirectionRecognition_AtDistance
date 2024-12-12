#pragma once
#include "../image_upscale/upscale.hpp"
#include "metrics.hpp"
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

  void Process();
  std::vector<Points> GetPixelsVectors();

private:
  DistantRoadRecognition &marker_;
  std::vector<Points> pixels_vectors_;
};

class UpscaleManager : public Manager {
public:
  UpscaleManager(Reader &reader, Writer &writer, Upscale &improver);
  void Process();

private:
  Upscale &improver_;
};

class MetricsManager {
public:
  MetricsManager(Reader &marking_res_reader, Reader &ground_truth_reader,
                 MetricsEvaluator &evaluator);
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

class RoiManager : public Manager {
public:
  RoiManager(Reader &reader, Writer &writer);
  void Process() override;

private:
  void SetRoi(const cv::Mat &sample);
  cv::Rect roi_;
};
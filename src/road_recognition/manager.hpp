#pragma once
#include "reader.hpp"
#include "recognition.hpp"

class Manager {
public:
  virtual ~Manager() {}
  virtual void Process() = 0;

protected:
  Manager(Reader &reader);
  Reader &reader_;
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
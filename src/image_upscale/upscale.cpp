#include "upscale.hpp"

void UpscaleESPCN::Execute(cv::Mat &img) {
  std::string path = "../../models/ESPCN_x4.pb";
  std::string model_name = "espcn";
  int scale = 4;
  cv::dnn_superres::DnnSuperResImpl sr;
  sr.readModel(path);
  sr.setModel(model_name, scale);
  sr.upsample(img, img);
}
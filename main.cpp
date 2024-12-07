#include "src/road_recognition/recognition.hpp"
#include <filesystem>

int main() {
  std::filesystem::path cwd = std::filesystem::current_path();
  std::string dir_path = cwd.parent_path().string() + "/images/";
  DistantRoadDirectionRecognitionTwinLiteNet detector(dir_path,
                                                      PhotoExtension::jpg);
  detector.MarkLaneAtDistance();

  return 0;
}
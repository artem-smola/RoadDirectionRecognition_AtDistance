#include "manager.hpp"
#include <filesystem>

int main() {
  std::filesystem::path cwd = std::filesystem::current_path();
  std::string dir_path = cwd.parent_path().string() + "/images/";

  FolderReader reader(dir_path, PhotoExtension::jpg);
  DistantRoadRecognitionTwinLiteNet marker;
  DistantRoadRecognitionManager manager(
      static_cast<Reader &>(reader),
      static_cast<DistantRoadRecognition &>(marker));
  manager.Process();
  return 0;
}
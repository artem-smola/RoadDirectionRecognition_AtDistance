#include "manager.hpp"
#include <filesystem>

int main() {
  std::filesystem::path cwd = std::filesystem::current_path();
  std::string in_path = cwd.parent_path().string() + "/input_images/";
  std::string out_path = cwd.parent_path().string() + "/output_images/";

  if (!std::filesystem::exists(in_path)) {
    throw std::runtime_error("Input directory does not exist: " + in_path);
  }

  if (!std::filesystem::exists(out_path)) {
    std::filesystem::create_directories(out_path);
  }

  FolderReader reader(in_path, PhotoExtension::jpg);
  FolderWriter writer(out_path, PhotoExtension::jpg);
  DistantRoadRecognitionTwinLiteNet marker;
  DistantRoadRecognitionManager manager(
      static_cast<Reader &>(reader), static_cast<Writer &>(writer),
      static_cast<DistantRoadRecognition &>(marker));

  manager.Process();
  std::cout << "Processing completed successfully." << std::endl;

  return 0;
}
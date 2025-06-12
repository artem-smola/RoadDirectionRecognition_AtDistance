#include "manager.hpp"
#include <filesystem>

int main() {
  std::string path =
      "/home/artem/Загрузки/ColorImage_road02/ColorImage/Record016/Camera 5/";
  FolderReader reader(path, PhotoExtension::jpg);
  RoiManager manager(reader);
  manager.Process();

  return 0;
}
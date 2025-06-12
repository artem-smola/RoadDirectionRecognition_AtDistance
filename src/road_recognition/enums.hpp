#pragma once
#include <array>
#include <string>

enum class PhotoExtension { jpg, png, count };

const std::array<std::string, static_cast<size_t>(PhotoExtension::count)>
    kExtensionNames = {".jpg", ".png"};

enum class Keys { up = 72, down = 80, left = 75, right = 77, enter = 13 };
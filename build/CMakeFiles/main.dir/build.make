# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.28

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:

#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:

# Disable VCS-based implicit rules.
% : %,v

# Disable VCS-based implicit rules.
% : RCS/%

# Disable VCS-based implicit rules.
% : RCS/%,v

# Disable VCS-based implicit rules.
% : SCCS/s.%

# Disable VCS-based implicit rules.
% : s.%

.SUFFIXES: .hpux_make_needs_suffix_list

# Command-line flag to silence nested $(MAKE).
$(VERBOSE)MAKESILENT = -s

#Suppress display of executed commands.
$(VERBOSE).SILENT:

# A target that is always out of date.
cmake_force:
.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /usr/bin/cmake

# The command to remove a file.
RM = /usr/bin/cmake -E rm -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/artem/RoadDirectionRecognition_AtDistance

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/artem/RoadDirectionRecognition_AtDistance/build

# Include any dependencies generated for this target.
include CMakeFiles/main.dir/depend.make
# Include any dependencies generated by the compiler for this target.
include CMakeFiles/main.dir/compiler_depend.make

# Include the progress variables for this target.
include CMakeFiles/main.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/main.dir/flags.make

CMakeFiles/main.dir/main.cpp.o: CMakeFiles/main.dir/flags.make
CMakeFiles/main.dir/main.cpp.o: /home/artem/RoadDirectionRecognition_AtDistance/main.cpp
CMakeFiles/main.dir/main.cpp.o: CMakeFiles/main.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --progress-dir=/home/artem/RoadDirectionRecognition_AtDistance/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/main.dir/main.cpp.o"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/main.dir/main.cpp.o -MF CMakeFiles/main.dir/main.cpp.o.d -o CMakeFiles/main.dir/main.cpp.o -c /home/artem/RoadDirectionRecognition_AtDistance/main.cpp

CMakeFiles/main.dir/main.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Preprocessing CXX source to CMakeFiles/main.dir/main.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/artem/RoadDirectionRecognition_AtDistance/main.cpp > CMakeFiles/main.dir/main.cpp.i

CMakeFiles/main.dir/main.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Compiling CXX source to assembly CMakeFiles/main.dir/main.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/artem/RoadDirectionRecognition_AtDistance/main.cpp -o CMakeFiles/main.dir/main.cpp.s

CMakeFiles/main.dir/TwinLiteNet-onnxruntime/src/twinlitenet_onnxruntime.cpp.o: CMakeFiles/main.dir/flags.make
CMakeFiles/main.dir/TwinLiteNet-onnxruntime/src/twinlitenet_onnxruntime.cpp.o: /home/artem/RoadDirectionRecognition_AtDistance/TwinLiteNet-onnxruntime/src/twinlitenet_onnxruntime.cpp
CMakeFiles/main.dir/TwinLiteNet-onnxruntime/src/twinlitenet_onnxruntime.cpp.o: CMakeFiles/main.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --progress-dir=/home/artem/RoadDirectionRecognition_AtDistance/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Building CXX object CMakeFiles/main.dir/TwinLiteNet-onnxruntime/src/twinlitenet_onnxruntime.cpp.o"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/main.dir/TwinLiteNet-onnxruntime/src/twinlitenet_onnxruntime.cpp.o -MF CMakeFiles/main.dir/TwinLiteNet-onnxruntime/src/twinlitenet_onnxruntime.cpp.o.d -o CMakeFiles/main.dir/TwinLiteNet-onnxruntime/src/twinlitenet_onnxruntime.cpp.o -c /home/artem/RoadDirectionRecognition_AtDistance/TwinLiteNet-onnxruntime/src/twinlitenet_onnxruntime.cpp

CMakeFiles/main.dir/TwinLiteNet-onnxruntime/src/twinlitenet_onnxruntime.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Preprocessing CXX source to CMakeFiles/main.dir/TwinLiteNet-onnxruntime/src/twinlitenet_onnxruntime.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/artem/RoadDirectionRecognition_AtDistance/TwinLiteNet-onnxruntime/src/twinlitenet_onnxruntime.cpp > CMakeFiles/main.dir/TwinLiteNet-onnxruntime/src/twinlitenet_onnxruntime.cpp.i

CMakeFiles/main.dir/TwinLiteNet-onnxruntime/src/twinlitenet_onnxruntime.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Compiling CXX source to assembly CMakeFiles/main.dir/TwinLiteNet-onnxruntime/src/twinlitenet_onnxruntime.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/artem/RoadDirectionRecognition_AtDistance/TwinLiteNet-onnxruntime/src/twinlitenet_onnxruntime.cpp -o CMakeFiles/main.dir/TwinLiteNet-onnxruntime/src/twinlitenet_onnxruntime.cpp.s

CMakeFiles/main.dir/src/road_recognition/recognition.cpp.o: CMakeFiles/main.dir/flags.make
CMakeFiles/main.dir/src/road_recognition/recognition.cpp.o: /home/artem/RoadDirectionRecognition_AtDistance/src/road_recognition/recognition.cpp
CMakeFiles/main.dir/src/road_recognition/recognition.cpp.o: CMakeFiles/main.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --progress-dir=/home/artem/RoadDirectionRecognition_AtDistance/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_3) "Building CXX object CMakeFiles/main.dir/src/road_recognition/recognition.cpp.o"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/main.dir/src/road_recognition/recognition.cpp.o -MF CMakeFiles/main.dir/src/road_recognition/recognition.cpp.o.d -o CMakeFiles/main.dir/src/road_recognition/recognition.cpp.o -c /home/artem/RoadDirectionRecognition_AtDistance/src/road_recognition/recognition.cpp

CMakeFiles/main.dir/src/road_recognition/recognition.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Preprocessing CXX source to CMakeFiles/main.dir/src/road_recognition/recognition.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/artem/RoadDirectionRecognition_AtDistance/src/road_recognition/recognition.cpp > CMakeFiles/main.dir/src/road_recognition/recognition.cpp.i

CMakeFiles/main.dir/src/road_recognition/recognition.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Compiling CXX source to assembly CMakeFiles/main.dir/src/road_recognition/recognition.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/artem/RoadDirectionRecognition_AtDistance/src/road_recognition/recognition.cpp -o CMakeFiles/main.dir/src/road_recognition/recognition.cpp.s

CMakeFiles/main.dir/src/road_recognition/reader.cpp.o: CMakeFiles/main.dir/flags.make
CMakeFiles/main.dir/src/road_recognition/reader.cpp.o: /home/artem/RoadDirectionRecognition_AtDistance/src/road_recognition/reader.cpp
CMakeFiles/main.dir/src/road_recognition/reader.cpp.o: CMakeFiles/main.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --progress-dir=/home/artem/RoadDirectionRecognition_AtDistance/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_4) "Building CXX object CMakeFiles/main.dir/src/road_recognition/reader.cpp.o"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/main.dir/src/road_recognition/reader.cpp.o -MF CMakeFiles/main.dir/src/road_recognition/reader.cpp.o.d -o CMakeFiles/main.dir/src/road_recognition/reader.cpp.o -c /home/artem/RoadDirectionRecognition_AtDistance/src/road_recognition/reader.cpp

CMakeFiles/main.dir/src/road_recognition/reader.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Preprocessing CXX source to CMakeFiles/main.dir/src/road_recognition/reader.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/artem/RoadDirectionRecognition_AtDistance/src/road_recognition/reader.cpp > CMakeFiles/main.dir/src/road_recognition/reader.cpp.i

CMakeFiles/main.dir/src/road_recognition/reader.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Compiling CXX source to assembly CMakeFiles/main.dir/src/road_recognition/reader.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/artem/RoadDirectionRecognition_AtDistance/src/road_recognition/reader.cpp -o CMakeFiles/main.dir/src/road_recognition/reader.cpp.s

CMakeFiles/main.dir/src/road_recognition/writer.cpp.o: CMakeFiles/main.dir/flags.make
CMakeFiles/main.dir/src/road_recognition/writer.cpp.o: /home/artem/RoadDirectionRecognition_AtDistance/src/road_recognition/writer.cpp
CMakeFiles/main.dir/src/road_recognition/writer.cpp.o: CMakeFiles/main.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --progress-dir=/home/artem/RoadDirectionRecognition_AtDistance/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_5) "Building CXX object CMakeFiles/main.dir/src/road_recognition/writer.cpp.o"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/main.dir/src/road_recognition/writer.cpp.o -MF CMakeFiles/main.dir/src/road_recognition/writer.cpp.o.d -o CMakeFiles/main.dir/src/road_recognition/writer.cpp.o -c /home/artem/RoadDirectionRecognition_AtDistance/src/road_recognition/writer.cpp

CMakeFiles/main.dir/src/road_recognition/writer.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Preprocessing CXX source to CMakeFiles/main.dir/src/road_recognition/writer.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/artem/RoadDirectionRecognition_AtDistance/src/road_recognition/writer.cpp > CMakeFiles/main.dir/src/road_recognition/writer.cpp.i

CMakeFiles/main.dir/src/road_recognition/writer.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Compiling CXX source to assembly CMakeFiles/main.dir/src/road_recognition/writer.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/artem/RoadDirectionRecognition_AtDistance/src/road_recognition/writer.cpp -o CMakeFiles/main.dir/src/road_recognition/writer.cpp.s

CMakeFiles/main.dir/src/road_recognition/manager.cpp.o: CMakeFiles/main.dir/flags.make
CMakeFiles/main.dir/src/road_recognition/manager.cpp.o: /home/artem/RoadDirectionRecognition_AtDistance/src/road_recognition/manager.cpp
CMakeFiles/main.dir/src/road_recognition/manager.cpp.o: CMakeFiles/main.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --progress-dir=/home/artem/RoadDirectionRecognition_AtDistance/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_6) "Building CXX object CMakeFiles/main.dir/src/road_recognition/manager.cpp.o"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/main.dir/src/road_recognition/manager.cpp.o -MF CMakeFiles/main.dir/src/road_recognition/manager.cpp.o.d -o CMakeFiles/main.dir/src/road_recognition/manager.cpp.o -c /home/artem/RoadDirectionRecognition_AtDistance/src/road_recognition/manager.cpp

CMakeFiles/main.dir/src/road_recognition/manager.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Preprocessing CXX source to CMakeFiles/main.dir/src/road_recognition/manager.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/artem/RoadDirectionRecognition_AtDistance/src/road_recognition/manager.cpp > CMakeFiles/main.dir/src/road_recognition/manager.cpp.i

CMakeFiles/main.dir/src/road_recognition/manager.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Compiling CXX source to assembly CMakeFiles/main.dir/src/road_recognition/manager.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/artem/RoadDirectionRecognition_AtDistance/src/road_recognition/manager.cpp -o CMakeFiles/main.dir/src/road_recognition/manager.cpp.s

CMakeFiles/main.dir/src/road_recognition/metrics.cpp.o: CMakeFiles/main.dir/flags.make
CMakeFiles/main.dir/src/road_recognition/metrics.cpp.o: /home/artem/RoadDirectionRecognition_AtDistance/src/road_recognition/metrics.cpp
CMakeFiles/main.dir/src/road_recognition/metrics.cpp.o: CMakeFiles/main.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --progress-dir=/home/artem/RoadDirectionRecognition_AtDistance/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_7) "Building CXX object CMakeFiles/main.dir/src/road_recognition/metrics.cpp.o"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/main.dir/src/road_recognition/metrics.cpp.o -MF CMakeFiles/main.dir/src/road_recognition/metrics.cpp.o.d -o CMakeFiles/main.dir/src/road_recognition/metrics.cpp.o -c /home/artem/RoadDirectionRecognition_AtDistance/src/road_recognition/metrics.cpp

CMakeFiles/main.dir/src/road_recognition/metrics.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Preprocessing CXX source to CMakeFiles/main.dir/src/road_recognition/metrics.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/artem/RoadDirectionRecognition_AtDistance/src/road_recognition/metrics.cpp > CMakeFiles/main.dir/src/road_recognition/metrics.cpp.i

CMakeFiles/main.dir/src/road_recognition/metrics.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Compiling CXX source to assembly CMakeFiles/main.dir/src/road_recognition/metrics.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/artem/RoadDirectionRecognition_AtDistance/src/road_recognition/metrics.cpp -o CMakeFiles/main.dir/src/road_recognition/metrics.cpp.s

CMakeFiles/main.dir/src/image_upscale/upscale.cpp.o: CMakeFiles/main.dir/flags.make
CMakeFiles/main.dir/src/image_upscale/upscale.cpp.o: /home/artem/RoadDirectionRecognition_AtDistance/src/image_upscale/upscale.cpp
CMakeFiles/main.dir/src/image_upscale/upscale.cpp.o: CMakeFiles/main.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --progress-dir=/home/artem/RoadDirectionRecognition_AtDistance/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_8) "Building CXX object CMakeFiles/main.dir/src/image_upscale/upscale.cpp.o"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/main.dir/src/image_upscale/upscale.cpp.o -MF CMakeFiles/main.dir/src/image_upscale/upscale.cpp.o.d -o CMakeFiles/main.dir/src/image_upscale/upscale.cpp.o -c /home/artem/RoadDirectionRecognition_AtDistance/src/image_upscale/upscale.cpp

CMakeFiles/main.dir/src/image_upscale/upscale.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Preprocessing CXX source to CMakeFiles/main.dir/src/image_upscale/upscale.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/artem/RoadDirectionRecognition_AtDistance/src/image_upscale/upscale.cpp > CMakeFiles/main.dir/src/image_upscale/upscale.cpp.i

CMakeFiles/main.dir/src/image_upscale/upscale.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Compiling CXX source to assembly CMakeFiles/main.dir/src/image_upscale/upscale.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/artem/RoadDirectionRecognition_AtDistance/src/image_upscale/upscale.cpp -o CMakeFiles/main.dir/src/image_upscale/upscale.cpp.s

# Object files for target main
main_OBJECTS = \
"CMakeFiles/main.dir/main.cpp.o" \
"CMakeFiles/main.dir/TwinLiteNet-onnxruntime/src/twinlitenet_onnxruntime.cpp.o" \
"CMakeFiles/main.dir/src/road_recognition/recognition.cpp.o" \
"CMakeFiles/main.dir/src/road_recognition/reader.cpp.o" \
"CMakeFiles/main.dir/src/road_recognition/writer.cpp.o" \
"CMakeFiles/main.dir/src/road_recognition/manager.cpp.o" \
"CMakeFiles/main.dir/src/road_recognition/metrics.cpp.o" \
"CMakeFiles/main.dir/src/image_upscale/upscale.cpp.o"

# External object files for target main
main_EXTERNAL_OBJECTS =

main: CMakeFiles/main.dir/main.cpp.o
main: CMakeFiles/main.dir/TwinLiteNet-onnxruntime/src/twinlitenet_onnxruntime.cpp.o
main: CMakeFiles/main.dir/src/road_recognition/recognition.cpp.o
main: CMakeFiles/main.dir/src/road_recognition/reader.cpp.o
main: CMakeFiles/main.dir/src/road_recognition/writer.cpp.o
main: CMakeFiles/main.dir/src/road_recognition/manager.cpp.o
main: CMakeFiles/main.dir/src/road_recognition/metrics.cpp.o
main: CMakeFiles/main.dir/src/image_upscale/upscale.cpp.o
main: CMakeFiles/main.dir/build.make
main: /usr/local/lib/libopencv_gapi.so.4.10.0
main: /usr/local/lib/libopencv_stitching.so.4.10.0
main: /usr/local/lib/libopencv_alphamat.so.4.10.0
main: /usr/local/lib/libopencv_aruco.so.4.10.0
main: /usr/local/lib/libopencv_bgsegm.so.4.10.0
main: /usr/local/lib/libopencv_bioinspired.so.4.10.0
main: /usr/local/lib/libopencv_ccalib.so.4.10.0
main: /usr/local/lib/libopencv_dnn_objdetect.so.4.10.0
main: /usr/local/lib/libopencv_dnn_superres.so.4.10.0
main: /usr/local/lib/libopencv_dpm.so.4.10.0
main: /usr/local/lib/libopencv_face.so.4.10.0
main: /usr/local/lib/libopencv_freetype.so.4.10.0
main: /usr/local/lib/libopencv_fuzzy.so.4.10.0
main: /usr/local/lib/libopencv_hdf.so.4.10.0
main: /usr/local/lib/libopencv_hfs.so.4.10.0
main: /usr/local/lib/libopencv_img_hash.so.4.10.0
main: /usr/local/lib/libopencv_intensity_transform.so.4.10.0
main: /usr/local/lib/libopencv_line_descriptor.so.4.10.0
main: /usr/local/lib/libopencv_mcc.so.4.10.0
main: /usr/local/lib/libopencv_quality.so.4.10.0
main: /usr/local/lib/libopencv_rapid.so.4.10.0
main: /usr/local/lib/libopencv_reg.so.4.10.0
main: /usr/local/lib/libopencv_rgbd.so.4.10.0
main: /usr/local/lib/libopencv_saliency.so.4.10.0
main: /usr/local/lib/libopencv_sfm.so.4.10.0
main: /usr/local/lib/libopencv_signal.so.4.10.0
main: /usr/local/lib/libopencv_stereo.so.4.10.0
main: /usr/local/lib/libopencv_structured_light.so.4.10.0
main: /usr/local/lib/libopencv_superres.so.4.10.0
main: /usr/local/lib/libopencv_surface_matching.so.4.10.0
main: /usr/local/lib/libopencv_tracking.so.4.10.0
main: /usr/local/lib/libopencv_videostab.so.4.10.0
main: /usr/local/lib/libopencv_wechat_qrcode.so.4.10.0
main: /usr/local/lib/libopencv_xfeatures2d.so.4.10.0
main: /usr/local/lib/libopencv_xobjdetect.so.4.10.0
main: /usr/local/lib/libopencv_xphoto.so.4.10.0
main: /usr/local/lib/libopencv_shape.so.4.10.0
main: /usr/local/lib/libopencv_highgui.so.4.10.0
main: /usr/local/lib/libopencv_datasets.so.4.10.0
main: /usr/local/lib/libopencv_plot.so.4.10.0
main: /usr/local/lib/libopencv_text.so.4.10.0
main: /usr/local/lib/libopencv_ml.so.4.10.0
main: /usr/local/lib/libopencv_phase_unwrapping.so.4.10.0
main: /usr/local/lib/libopencv_optflow.so.4.10.0
main: /usr/local/lib/libopencv_ximgproc.so.4.10.0
main: /usr/local/lib/libopencv_video.so.4.10.0
main: /usr/local/lib/libopencv_videoio.so.4.10.0
main: /usr/local/lib/libopencv_imgcodecs.so.4.10.0
main: /usr/local/lib/libopencv_objdetect.so.4.10.0
main: /usr/local/lib/libopencv_calib3d.so.4.10.0
main: /usr/local/lib/libopencv_dnn.so.4.10.0
main: /usr/local/lib/libopencv_features2d.so.4.10.0
main: /usr/local/lib/libopencv_flann.so.4.10.0
main: /usr/local/lib/libopencv_photo.so.4.10.0
main: /usr/local/lib/libopencv_imgproc.so.4.10.0
main: /usr/local/lib/libopencv_core.so.4.10.0
main: CMakeFiles/main.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --bold --progress-dir=/home/artem/RoadDirectionRecognition_AtDistance/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_9) "Linking CXX executable main"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/main.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/main.dir/build: main
.PHONY : CMakeFiles/main.dir/build

CMakeFiles/main.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/main.dir/cmake_clean.cmake
.PHONY : CMakeFiles/main.dir/clean

CMakeFiles/main.dir/depend:
	cd /home/artem/RoadDirectionRecognition_AtDistance/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/artem/RoadDirectionRecognition_AtDistance /home/artem/RoadDirectionRecognition_AtDistance /home/artem/RoadDirectionRecognition_AtDistance/build /home/artem/RoadDirectionRecognition_AtDistance/build /home/artem/RoadDirectionRecognition_AtDistance/build/CMakeFiles/main.dir/DependInfo.cmake "--color=$(COLOR)"
.PHONY : CMakeFiles/main.dir/depend


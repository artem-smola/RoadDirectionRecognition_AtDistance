# RoadDirectionRecognition_AtDistance

## Requirements

- OpenCV (>= 4.10.0)
- ONNX Runtime (>= 1.20.0)

## Usage

1. Clone the repository:

```bash
git clone --recurse-submodule https://github.com/artem-smola/RoadDirectionRecognition_AtDistance.git
```

2. Replace the path to ONNX Runtime in the following line in `CMakeLists.txt` with your path to ONNX Runtime:
```
set(ONNXRUNTIME_ROOTDIR "/home/artem/onnxruntime-linux-x64-1.20.0")
``` 

3. Build the project using CMake.
```bash
mkdir build
cd build
cmake ..
cmake --build .
```

4. Execute `./main` from the build directory to try distant road recognition.


## License

This project is licensed under the terms of the **MIT** license. See the [LICENSE](LICENSE) for more information.
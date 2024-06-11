# win-libtorch-yolov3
A Libtorch-based implementation of the YOLO v3 object detection algorithm working in Windows environment, written with pure C++. It's easy to be reproduced in your Windows environment. CPU is supported for now. Please enjoy yourself.

This repository is derived from [libtorch-yolov3](https://github.com/walktree/libtorch-yolov3), with adaption to Windows platform and association of annotation to bounding box.

## Preparation

### Requirements
1. LibTorch(CPU version) v1.6.0 and newer
2. OpenCV 3.3 and newer (used only in images input/output phases)
3. VSCode + C/C+ Extension
4. Visual Studio 

### Toolchains
1. CMake 3.8+
2. Visual Studio Community 20xx Release - amd64(dependent on your computer configuration)

   Note: In place once Visual Studio is installed. No need of additional action.

### Adaptation to your environment
1. Modify TORCH_PREFIX_PATH to point to your Torch installation directory.

   Example：

   ```set(TORCH_PREFIX_PATH "<your path>/libtorch")```
2. Modify OPENCV_PREFIX_PATH to point to your Torch installation directory.

   Example：

   ```set(OPENCV_PREFIX_PATH "<your path>/opencv")```

## Building executable
Trigger Build command in VSCode UI to get build/Debug/yolo-app.exe

## Detection from images

The first thing you need to do is to get the weights file for v3:

```
Download https://pjreddie.com/media/files/yolov3.weights to directory models
```

On Single image:
```
cd build/Debug
yolo-app.exe ../../imgs/person.jpg
```
On Multiple images under a directory imgs:
```
cd build/Debug
yolo-app.exe ../../imgs
```
Generated images with bounding box and annotation are kept under directory det.

## Debugging in VS Code IDE
If you would like to debug your code in VSCode, please follow following steps.
1. Start VS Code from Developer Command Prompt. An example as below.
   ```
   D:\src>cd win-libtorch-yolov3
   D:\src\win-libtorch-yolov3>code .
   ```
2. Update includePath field via UI or c_cpp_properties.json to contain your OpenCV and Libtorch include paths.
   ```
   "includePath": [
                "${workspaceFolder}/**",
                "<your path>/opencv/build/include",
                "<your path>/libtorch/include/torch/csrc/api/include"
            ]
   ```
3. Create launch.json with your actual values, like
   ```
   "configurations": [
        {
            "name": "(Windows) Launch",
            "type": "cppvsdbg",
            "request": "launch",
            "program": "${fileDirname}/../build/Debug/<your app>.exe",
            "args": ["${fileDirname}/../imgs"],
            "stopAtEntry": false,
            "cwd": "${fileDirname}",
            "environment": [],
            "console": "externalTerminal"
        }

    ]
   ```

For cooresponding Pytorch version, please move to [eng-pytorch-yolov3](https://github.com/RyanFeiluX/eng-pytorch-yolov3)

cmake_minimum_required(VERSION 3.8 FATAL_ERROR)
project(yolo-app)

set(TORCH_PREFIX_PATH "D:/src/LibTorch/libtorch-win-shared-with-deps-debug-1.6.0+cpu/libtorch")
# set(TORCH_PREFIX_PATH "D:/ProgramData/anaconda3/envs/condaenv-torch/Lib/site-packages/torch")
set(Torch_DIR "${TORCH_PREFIX_PATH}/share/cmake/Torch")
find_package(Torch REQUIRED)
include_directories( ${TORCH_INCLUDE_DIRS} )
message([DEBUG] "Torch include path: ${TORCH_INCLUDE_DIRS}")

set(OPENCV_PREFIX_PATH "D:/src/opencv-3.3.0-vc14/opencv")
set(OpenCV_DIR "${OPENCV_PREFIX_PATH}/build/x64/vc14/lib")
find_package( OpenCV REQUIRED )
include_directories( ${OpenCV_INCLUDE_DIRS} )
message([DEBUG] "OpenCV include path: ${OpenCV_INCLUDE_DIRS}")

aux_source_directory(. DIR_SRCS)

if (MSVC)
	# Add compilation option to set codeset to utf-8
	add_compile_options("/source-charset:utf-8")
endif(MSVC)

add_executable(yolo-app ${DIR_SRCS})
target_link_libraries(yolo-app "${TORCH_LIBRARIES}" "${OpenCV_LIBS}")
set_property(TARGET yolo-app PROPERTY CXX_STANDARD 17)

# The following code block is suggested to be used on Windows.
# According to https://github.com/pytorch/pytorch/issues/25457,
# the DLLs need to be copied to avoid memory errors.
if (MSVC)
  set(TORCH_BIN "${TORCH_PREFIX_PATH}/lib")
  file(GLOB TORCH_DLLS "${TORCH_BIN}/*.dll")
  set(OpenCV_BIN "${OPENCV_PREFIX_PATH}/build/x64/vc14/bin")
  file(GLOB OPENCV_DLLS "${OpenCV_BIN}/*.dll")
  add_custom_command(TARGET yolo-app
                     POST_BUILD
                     COMMAND ${CMAKE_COMMAND} -E copy_if_different
                     ${TORCH_DLLS} ${OPENCV_DLLS}
                     $<TARGET_FILE_DIR:yolo-app>)
endif (MSVC)
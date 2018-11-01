cmake_minimum_required(VERSION 2.6 FATAL_ERROR)
project(facelock)

### Stuff for MacOS

set(CMAKE_MACOSX_RPATH 1)

### Add your version from cmake

# The version number.
set (facelock_VERSION_MAJOR 1)
set (facelock_VERSION_MINOR 0)

### Add external libraries

# Tell cmake we will need dlib.  This command will pull in dlib and compile it
# into your project.  Note that you don't need to compile or install dlib.  All
# cmake needs is the dlib source code folder and it will take care of everything.

add_subdirectory(../../dlib dlib_build)
#include_directories(../../dlib/tools/json)
include_directories(../../dlib/)
#include_directories(../../dlib/)
#get_property(dlib_opts TARGET dlib PROPERTY COMPILE_OPTIONS)
#list(APPEND CUDA_NVCC_FLAGS "${dlib_opts}")
#cuda_include_directories(../../dlib/)

# Tell cmake we will need opencv.

find_package(OpenCV REQUIRED)
message(STATUS "Found OpenCV version ${OpenCV_VERSION}")
include_directories( ${OpenCV_INCLUDE_DIRS} )

### Now add your executable

# add the executable
add_executable(facelock_exe facelock.cpp )

# You need to tell CMake that this program
# depends on dlib and opencv.
# You do that with the below statements:
target_link_libraries(facelock_exe dlib::dlib)
target_link_libraries( facelock_exe ${OpenCV_LIBS} )

### And cpack for your cpp code distributions
### binary and source distributions
### (usage) to distribute binary:
### cpack --config CPackConfig.cmake
### (usage) to distribute source code:
### cpack --config CPackSourceConfig.cmake

# build a CPack driven installer package
include (InstallRequiredSystemLibraries)
set (CPACK_RESOURCE_FILE_LICENSE "${CMAKE_CURRENT_SOURCE_DIR}/License.txt")
set (CPACK_PACKAGE_VERSION_MAJOR "${facelock_VERSION_MAJOR}")
set (CPACK_PACKAGE_VERSION_MINOR "${facelock_VERSION_MINOR}")
include (CPack)

### Add and get your shared library here

add_library(facelock_shared SHARED facelock.cpp ../../dlib/dlib/all/source.cpp)

### Add and get your object library here

add_library(facelock_object OBJECT facelock.cpp ../../dlib/dlib/all/source.cpp)

### Add SWIG to port cpp to Python

# This is a CMake example for Python

FIND_PACKAGE(SWIG REQUIRED)
INCLUDE(${SWIG_USE_FILE})

FIND_PACKAGE(PythonLibs 2.7 EXACT REQUIRED)
INCLUDE_DIRECTORIES(${PYTHON_INCLUDE_PATH})

message(STATUS "PYTHONLIBS_FOUND:    ${PYTHONLIBS_FOUND}")
message(STATUS "PYTHON_LIBRARIES:    ${PYTHON_LIBRARIES}")
message(STATUS "PYTHON_INCLUDE_PATH:    ${PYTHON_INCLUDE_PATH}")
message(STATUS "PYTHON_INCLUDE_DIRS:    ${PYTHON_INCLUDE_DIRS}")
message(STATUS "PYTHON_DEBUG_LIBRARIES:    ${PYTHON_DEBUG_LIBRARIES}")
message(STATUS "PYTHONLIBS_VERSION_STRING:    ${PYTHONLIBS_VERSION_STRING}")

INCLUDE_DIRECTORIES(${CMAKE_CURRENT_SOURCE_DIR})

SET(CMAKE_SWIG_FLAGS "")

SET_SOURCE_FILES_PROPERTIES(facelock.i PROPERTIES CPLUSPLUS ON)
SET_SOURCE_FILES_PROPERTIES(facelock.i PROPERTIES SWIG_FLAGS "-includeall")
SWIG_ADD_MODULE(facelock python facelock.i facelock.cpp ../../dlib/dlib/all/source.cpp)
#SWIG_ADD_LIBRARY(facelock python facelock.i facelock.cpp ../../dlib/dlib/all/source.cpp)
SWIG_LINK_LIBRARIES(facelock ${PYTHON_LIBRARIES})
SWIG_LINK_LIBRARIES(facelock dlib::dlib)
SWIG_LINK_LIBRARIES(facelock ${OpenCV_LIBS} )


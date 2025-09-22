cmake_minimum_required(VERSION 3.10)
project(kmeans)

set(CMAKE_CXX_STANDARD 17)

find_package(ZLIB REQUIRED)
include_directories(${ZLIB_INCLUDE_DIRS})

add_executable(kmeans
    src/main.cpp
    src/NpyLoader.cpp
    src/cnpy.cpp
    src/KMeans.cpp
)

target_link_libraries(kmeans PRIVATE ${ZLIB_LIBRARIES})
target_include_directories(kmeans PRIVATE src)
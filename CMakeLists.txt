cmake_minimum_required(VERSION 3.10)
project(kmeans)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_BUILD_TYPE Debug)

find_package(ZLIB REQUIRED)
include_directories(${ZLIB_INCLUDE_DIRS})

add_subdirectory(external/xtl)
add_subdirectory(external/xtensor)
# Add this line to make ghc filesystem headers available:
set(ghc-filesystem-include "${CMAKE_SOURCE_DIR}/external/ghc-filesystem/include")
add_subdirectory(external/xtensor-io)

add_executable(kmeans
    src/main.cpp
    src/NpyLoader.cpp
    src/cnpy.cpp
    src/KMeans.cpp
)
target_link_libraries(kmeans PRIVATE ${ZLIB_LIBRARIES} xtl xtensor xtensor-io)
target_include_directories(kmeans PRIVATE src external/ghc-filesystem/include)

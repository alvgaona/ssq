cmake_minimum_required(VERSION 3.18)
project(ssq LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Find required packages
find_package(PkgConfig REQUIRED)
pkg_check_modules(FFTW3 REQUIRED fftw3)
find_package(Eigen3 REQUIRED)

# Option for building tests
option(SSQ_BUILD_TESTS "Build unit tests" OFF)

# Main library
add_library(ssq_core STATIC
    src/fftw_wrapper.cpp
    src/stft.cpp
    src/fsst.cpp
    src/wavelet.cpp
    src/cwt.cpp
    src/wsst.cpp
)

target_include_directories(ssq_core PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
    ${FFTW3_INCLUDE_DIRS}
)

target_link_libraries(ssq_core PUBLIC ${FFTW3_LIBRARIES} Eigen3::Eigen)
target_link_directories(ssq_core PUBLIC ${FFTW3_LIBRARY_DIRS})

# Python bindings
find_package(pybind11 CONFIG)
if(pybind11_FOUND)
    pybind11_add_module(ssq python/bindings.cpp)
    target_link_libraries(ssq PRIVATE ssq_core)

    # Get Python site-packages directory
    execute_process(
        COMMAND ${PYTHON_EXECUTABLE} -c "import site; print(site.getsitepackages()[0])"
        OUTPUT_VARIABLE PYTHON_SITE_PACKAGES
        OUTPUT_STRIP_TRAILING_WHITESPACE
    )

    # Install Python module to site-packages
    install(TARGETS ssq LIBRARY DESTINATION ${PYTHON_SITE_PACKAGES})
endif()

# Tests
if(SSQ_BUILD_TESTS)
    enable_testing()
    find_package(GTest REQUIRED)

    add_executable(test_stft test/test_stft.cpp)
    target_link_libraries(test_stft PRIVATE ssq_core GTest::gtest GTest::gtest_main)
    add_test(NAME test_stft COMMAND test_stft)

    add_executable(test_fsst test/test_fsst.cpp)
    target_link_libraries(test_fsst PRIVATE ssq_core GTest::gtest GTest::gtest_main)
    add_test(NAME test_fsst COMMAND test_fsst)

    add_executable(test_wsst test/test_wsst.cpp)
    target_link_libraries(test_wsst PRIVATE ssq_core GTest::gtest GTest::gtest_main)
    add_test(NAME test_wsst COMMAND test_wsst)
endif()

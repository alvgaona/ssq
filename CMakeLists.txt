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
find_package(OpenMP)

# Options
option(SSQ_BUILD_TESTS "Build unit tests" OFF)
option(SSQ_NATIVE_ARCH "Optimize for host CPU (-march=native)" OFF)

# Main library
add_library(ssq_core STATIC
    src/fftw_wrapper.cpp
    src/stft.cpp
    src/fsst.cpp
    src/wavelet.cpp
    src/cwt.cpp
    src/wsst.cpp
    src/windows.cpp
)

target_include_directories(ssq_core PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
    ${FFTW3_INCLUDE_DIRS}
)

target_link_libraries(ssq_core PUBLIC ${FFTW3_LIBRARIES} Eigen3::Eigen)
target_link_directories(ssq_core PUBLIC ${FFTW3_LIBRARY_DIRS})

if(OpenMP_CXX_FOUND)
    target_link_libraries(ssq_core PUBLIC OpenMP::OpenMP_CXX)
endif()

# Optional optimization flags
if(SSQ_NATIVE_ARCH)
    target_compile_options(ssq_core PRIVATE -march=native)
endif()


# Install C++ library and headers
install(TARGETS ssq_core LIBRARY DESTINATION lib ARCHIVE DESTINATION lib)
install(DIRECTORY include/ssq DESTINATION include)

# Python bindings
find_package(Python COMPONENTS Interpreter Development QUIET)
find_package(pybind11 CONFIG)
if(pybind11_FOUND AND Python_FOUND)
    pybind11_add_module(ssq_python python/bindings.cpp)
    target_link_libraries(ssq_python PRIVATE ssq_core)
    set_target_properties(ssq_python PROPERTIES OUTPUT_NAME ssq)

    # Get Python site-packages for install
    execute_process(
        COMMAND "${Python_EXECUTABLE}" -c "import sysconfig; print(sysconfig.get_path('platlib'))"
        OUTPUT_VARIABLE PYTHON_SITE_PACKAGES
        OUTPUT_STRIP_TRAILING_WHITESPACE
    )
    if(PYTHON_SITE_PACKAGES)
        install(TARGETS ssq_python LIBRARY DESTINATION "${PYTHON_SITE_PACKAGES}")
    endif()
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

    add_executable(test_windows test/test_windows.cpp)
    target_link_libraries(test_windows PRIVATE ssq_core GTest::gtest GTest::gtest_main)
    add_test(NAME test_windows COMMAND test_windows)
endif()

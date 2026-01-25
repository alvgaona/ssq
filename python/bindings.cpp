#include <pybind11/eigen.h>
#include <pybind11/pybind11.h>

#include "ssq/fsst.hpp"
#include "ssq/wsst.hpp"

#include <stdexcept>

namespace py = pybind11;

// Python wrapper for fsst function
// Returns tuple of (spectrum, frequencies, times) as numpy arrays
// pybind11's Eigen support automatically converts between numpy and Eigen types
py::tuple py_fsst(Eigen::Ref<const Eigen::VectorXd> signal, double sample_rate,
                  Eigen::Ref<const Eigen::VectorXd> window, double threshold = 1e-6) {
    // Compute FSST
    ssq::FsstResult result = ssq::fsst(signal, sample_rate, window, threshold);

    // Return as tuple - pybind11 automatically converts Eigen types to numpy
    return py::make_tuple(std::move(result.spectrum), std::move(result.frequencies), std::move(result.times));
}

// Python wrapper for wsst function (MATLAB-compatible API)
// Returns tuple of (spectrum, frequencies, times) as numpy arrays
py::tuple py_wsst(Eigen::Ref<const Eigen::VectorXd> signal, double sample_rate, const std::string& wavelet = "amor",
                  int num_voices = 32, double threshold = 1e-6) {
    // Parse wavelet type (MATLAB naming convention)
    ssq::WaveletType wavelet_type;
    if (wavelet == "amor" || wavelet == "morlet") {
        wavelet_type = ssq::WaveletType::Morlet;
    } else if (wavelet == "bump") {
        wavelet_type = ssq::WaveletType::Bump;
    } else {
        throw std::invalid_argument("Unknown wavelet type: " + wavelet + ". Supported: 'amor', 'morlet', 'bump'");
    }

    // Compute WSST
    ssq::WsstResult result = ssq::wsst(signal, sample_rate, wavelet_type, num_voices, threshold);

    // Return as tuple - pybind11 automatically converts Eigen types to numpy
    return py::make_tuple(std::move(result.spectrum), std::move(result.frequencies), std::move(result.times));
}

PYBIND11_MODULE(ssq, m) {
    m.doc() = "Synchrosqueezed Transform implementations (FSST and WSST)";

    m.def("fsst", &py_fsst, "Compute the Fourier Synchrosqueezed Transform", py::arg("signal"),
          py::arg("sample_rate"), py::arg("window"), py::arg("threshold") = 1e-6);

    m.def("wsst", &py_wsst,
          "Compute the Wavelet Synchrosqueezed Transform\n\n"
          "Args:\n"
          "    signal: Input signal (1D array)\n"
          "    sample_rate: Sampling frequency in Hz\n"
          "    wavelet: Wavelet type - 'amor' (analytic Morlet, default), 'bump'\n"
          "    num_voices: Voices per octave (default 32)\n"
          "    threshold: Numerical stability threshold (default 1e-6)\n\n"
          "Returns:\n"
          "    Tuple of (spectrum, frequencies, times)",
          py::arg("signal"), py::arg("sample_rate"), py::arg("wavelet") = "amor", py::arg("num_voices") = 32,
          py::arg("threshold") = 1e-6);
}

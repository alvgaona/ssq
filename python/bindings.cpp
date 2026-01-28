#include <pybind11/eigen.h>
#include <pybind11/pybind11.h>

#include "ssq/fsst.hpp"
#include "ssq/windows.hpp"
#include "ssq/wsst.hpp"

#include <stdexcept>

namespace py = pybind11;

// Python wrapper for fsst function
// Returns tuple of (spectrum, frequencies, times) as numpy arrays
py::tuple py_fsst(Eigen::Ref<const Eigen::VectorXd> signal, double sample_rate,
                  Eigen::Ref<const Eigen::VectorXd> window, double threshold = 1e-6) {
    ssq::FsstResult result = ssq::fsst(signal, sample_rate, window, threshold);
    return py::make_tuple(std::move(result.spectrum), std::move(result.frequencies), std::move(result.times));
}

// Python wrapper for ifsst function (full reconstruction)
Eigen::VectorXd py_ifsst(Eigen::Ref<const Eigen::MatrixXcd> spectrum, Eigen::Ref<const Eigen::VectorXd> window) {
    return ssq::ifsst(spectrum, window);
}

// Python wrapper for ifsst function with frequency range
Eigen::VectorXd py_ifsst_freqrange(Eigen::Ref<const Eigen::MatrixXcd> spectrum,
                                   Eigen::Ref<const Eigen::VectorXd> window,
                                   Eigen::Ref<const Eigen::VectorXd> frequencies,
                                   std::pair<double, double> freqrange) {
    return ssq::ifsst(spectrum, window, frequencies, freqrange);
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

// Python wrapper for iwsst function (full reconstruction)
Eigen::VectorXd py_iwsst(Eigen::Ref<const Eigen::MatrixXcd> spectrum, Eigen::Ref<const Eigen::VectorXd> frequencies) {
    return ssq::iwsst(spectrum, frequencies);
}

// Python wrapper for iwsst function with frequency range
Eigen::VectorXd py_iwsst_freqrange(Eigen::Ref<const Eigen::MatrixXcd> spectrum,
                                   Eigen::Ref<const Eigen::VectorXd> frequencies,
                                   std::pair<double, double> freqrange) {
    return ssq::iwsst(spectrum, frequencies, freqrange);
}

PYBIND11_MODULE(ssq, m) {
    m.doc() = "Synchrosqueezed Transform implementations (FSST and WSST)";

    m.def("fsst", &py_fsst, "Compute the Fourier Synchrosqueezed Transform", py::arg("signal"),
          py::arg("sample_rate"), py::arg("window"), py::arg("threshold") = 1e-6);

    m.def("ifsst", &py_ifsst,
          "Inverse FSST: reconstruct signal from synchrosqueezed spectrum\n\n"
          "Args:\n"
          "    spectrum: Synchrosqueezed spectrum from fsst()\n"
          "    window: Analysis window used in forward transform\n\n"
          "Returns:\n"
          "    Reconstructed signal",
          py::arg("spectrum"), py::arg("window"));

    m.def("ifsst", &py_ifsst_freqrange,
          "Inverse FSST with frequency range: reconstruct only specified frequency range\n\n"
          "Args:\n"
          "    spectrum: Synchrosqueezed spectrum from fsst()\n"
          "    window: Analysis window used in forward transform\n"
          "    frequencies: Frequency axis from fsst()\n"
          "    freqrange: Tuple (fmin, fmax) in Hz - only reconstruct this range\n\n"
          "Returns:\n"
          "    Reconstructed signal containing only the specified frequency range",
          py::arg("spectrum"), py::arg("window"), py::arg("frequencies"), py::arg("freqrange"));

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

    m.def("iwsst", &py_iwsst,
          "Inverse WSST: reconstruct signal from synchrosqueezed spectrum\n\n"
          "Args:\n"
          "    spectrum: Synchrosqueezed spectrum from wsst()\n"
          "    frequencies: Frequency axis from wsst()\n\n"
          "Returns:\n"
          "    Reconstructed signal",
          py::arg("spectrum"), py::arg("frequencies"));

    m.def("iwsst", &py_iwsst_freqrange,
          "Inverse WSST with frequency range: reconstruct only specified frequency range\n\n"
          "Args:\n"
          "    spectrum: Synchrosqueezed spectrum from wsst()\n"
          "    frequencies: Frequency axis from wsst()\n"
          "    freqrange: Tuple (fmin, fmax) in Hz - only reconstruct this range\n\n"
          "Returns:\n"
          "    Reconstructed signal containing only the specified frequency range",
          py::arg("spectrum"), py::arg("frequencies"), py::arg("freqrange"));

    // Windows submodule
    py::module_ windows = m.def_submodule("windows", "Window functions for FSST");

    windows.def("kaiser", &ssq::windows::kaiser,
                "Kaiser window\n\n"
                "Args:\n"
                "    length: Window length\n"
                "    beta: Shape parameter (default 8.6)\n\n"
                "Returns:\n"
                "    Window array",
                py::arg("length"), py::arg("beta") = 8.6);

    windows.def("hamming", &ssq::windows::hamming,
                "Hamming window\n\n"
                "Args:\n"
                "    length: Window length\n\n"
                "Returns:\n"
                "    Window array",
                py::arg("length"));

    windows.def("hann", &ssq::windows::hann,
                "Hann window\n\n"
                "Args:\n"
                "    length: Window length\n\n"
                "Returns:\n"
                "    Window array",
                py::arg("length"));

    windows.def("gaussian", &ssq::windows::gaussian,
                "Gaussian window\n\n"
                "Args:\n"
                "    length: Window length\n"
                "    sigma: Standard deviation (default length/6)\n\n"
                "Returns:\n"
                "    Window array",
                py::arg("length"), py::arg("sigma") = 0.0);
}

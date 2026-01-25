#ifndef SSQ_CWT_HPP
#define SSQ_CWT_HPP

#include "ssq/fftw_wrapper.hpp"
#include "ssq/wavelet.hpp"

#include <Eigen/Dense>

namespace ssq {

// CWT result containing the transform and derivative transform
struct CwtResult {
    Eigen::MatrixXcd cwt;      // W_psi: CWT coefficients (num_scales x time_steps)
    Eigen::MatrixXcd cwt_d;    // W_psi': CWT with wavelet derivative (for phase transform)
    Eigen::VectorXd scales;    // Scale values used
    Eigen::VectorXd frequencies;  // Corresponding frequencies
    Eigen::VectorXd times;     // Time axis
};

class Cwt {
public:
    // Constructor with wavelet type and optional parameters
    explicit Cwt(WaveletType wavelet = WaveletType::Morlet, int num_voices = 32);

    // Compute CWT of signal
    // Returns both the CWT and the CWT with derivative wavelet (for synchrosqueezing)
    CwtResult compute(const Eigen::VectorXd& signal, double sample_rate) const;

    // Compute CWT with custom scales
    CwtResult compute(const Eigen::VectorXd& signal, double sample_rate, const Eigen::VectorXd& scales) const;

    // Accessors
    WaveletType wavelet_type() const { return wavelet_type_; }
    int num_voices() const { return num_voices_; }
    double omega0() const { return omega0_; }

private:
    WaveletType wavelet_type_;
    int num_voices_;
    double omega0_;  // Morlet center frequency parameter
};

}  // namespace ssq

#endif  // SSQ_CWT_HPP

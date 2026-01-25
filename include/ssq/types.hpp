#ifndef SSQ_TYPES_HPP
#define SSQ_TYPES_HPP

#include <Eigen/Dense>

namespace ssq {

// STFT result containing the transform and derivative transform
struct StftResult {
    Eigen::MatrixXcd stft;     // V_g: STFT with window g (freq_bins x time_steps)
    Eigen::MatrixXcd stft_dg;  // V_dg: STFT with window derivative dg
    Eigen::VectorXd frequencies;
    Eigen::VectorXd times;
};

// FSST result
struct FsstResult {
    Eigen::MatrixXcd spectrum;  // Synchrosqueezed spectrum (freq_bins x time_steps)
    Eigen::VectorXd frequencies;
    Eigen::VectorXd times;
};

}  // namespace ssq

#endif  // SSQ_TYPES_HPP

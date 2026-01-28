#ifndef SSQ_WINDOWS_HPP
#define SSQ_WINDOWS_HPP

#include <Eigen/Dense>

namespace ssq {
namespace windows {

// Kaiser window with shape parameter beta
// beta controls the trade-off between main-lobe width and side-lobe level
// Typical values: beta=5 (similar to Hamming), beta=8.6 (similar to Blackman)
Eigen::VectorXd kaiser(Eigen::Index length, double beta = 8.6);

// Hamming window: 0.54 - 0.46 * cos(2*pi*n/(N-1))
// Good for spectral analysis, minimizes nearest side-lobe
Eigen::VectorXd hamming(Eigen::Index length);

// Hann window: 0.5 * (1 - cos(2*pi*n/(N-1)))
// Good general-purpose window, also called "raised cosine"
Eigen::VectorXd hann(Eigen::Index length);

// Gaussian window: exp(-0.5 * ((n - N/2) / sigma)^2)
// sigma controls width; smaller sigma = narrower window
// Default sigma = length/6 gives good time-frequency resolution
Eigen::VectorXd gaussian(Eigen::Index length, double sigma = 0.0);

}  // namespace windows
}  // namespace ssq

#endif  // SSQ_WINDOWS_HPP

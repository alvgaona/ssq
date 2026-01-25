#include <gtest/gtest.h>

#include "ssq/stft.hpp"

#include <cmath>

namespace {

const double PI = 3.14159265358979323846;

// Generate a rectangular window
Eigen::VectorXd rect_window(Eigen::Index n) {
    return Eigen::VectorXd::Ones(n);
}

// Generate a Hann window
Eigen::VectorXd hann_window(Eigen::Index n) {
    Eigen::VectorXd w(n);
    for (Eigen::Index i = 0; i < n; ++i) {
        w(i) = 0.5 * (1.0 - std::cos(2.0 * PI * static_cast<double>(i) / (n - 1)));
    }
    return w;
}

// Generate a DC signal (constant value)
Eigen::VectorXd dc_signal(Eigen::Index n, double value = 1.0) {
    return Eigen::VectorXd::Constant(n, value);
}

// Generate a sinusoid
Eigen::VectorXd sine_signal(Eigen::Index n, double freq, double sample_rate) {
    Eigen::VectorXd sig(n);
    for (Eigen::Index i = 0; i < n; ++i) {
        sig(i) = std::sin(2.0 * PI * freq * static_cast<double>(i) / sample_rate);
    }
    return sig;
}

}  // namespace

class StftTest : public ::testing::Test {
protected:
    const Eigen::Index window_size = 64;
    const double sample_rate = 1000.0;
};

TEST_F(StftTest, WindowDerivativeCentralDifference) {
    // Test that window derivative is computed correctly using central difference
    Eigen::VectorXd win = hann_window(window_size);
    ssq::Stft stft(win);

    const auto& dg = stft.window_derivative();

    // Window derivative should have same size as window
    ASSERT_EQ(dg.size(), window_size);

    // Check a few interior points using central difference formula
    for (Eigen::Index i = 1; i < window_size - 1; ++i) {
        double expected = (win(i + 1) - win(i - 1)) / 2.0;
        EXPECT_NEAR(dg(i), expected, 1e-10);
    }
}

TEST_F(StftTest, OutputDimensions) {
    Eigen::VectorXd win = rect_window(window_size);
    ssq::Stft stft(win);

    Eigen::VectorXd sig = dc_signal(200);
    auto result = stft.compute(sig, sample_rate);

    // Number of frequency bins: nfft/2 + 1
    Eigen::Index expected_freqs = window_size / 2 + 1;
    // Number of time steps: signal length (hop = 1)
    Eigen::Index expected_times = sig.size();

    EXPECT_EQ(result.stft.rows(), expected_freqs);
    EXPECT_EQ(result.stft.cols(), expected_times);
    EXPECT_EQ(result.stft_dg.rows(), expected_freqs);
    EXPECT_EQ(result.stft_dg.cols(), expected_times);
    EXPECT_EQ(result.frequencies.size(), expected_freqs);
    EXPECT_EQ(result.times.size(), expected_times);
}

TEST_F(StftTest, FrequencyAxis) {
    Eigen::VectorXd win = rect_window(window_size);
    ssq::Stft stft(win);

    Eigen::VectorXd sig = dc_signal(100);
    auto result = stft.compute(sig, sample_rate);

    // Frequency axis should go from 0 to Nyquist
    EXPECT_NEAR(result.frequencies(0), 0.0, 1e-10);

    double nyquist = sample_rate / 2.0;
    EXPECT_NEAR(result.frequencies(result.frequencies.size() - 1), nyquist, 1e-10);

    // Check frequency spacing
    double df = sample_rate / static_cast<double>(window_size);
    for (Eigen::Index k = 0; k < result.frequencies.size() - 1; ++k) {
        EXPECT_NEAR(result.frequencies(k + 1) - result.frequencies(k), df, 1e-10);
    }
}

TEST_F(StftTest, TimeAxis) {
    Eigen::VectorXd win = rect_window(window_size);
    ssq::Stft stft(win);

    Eigen::VectorXd sig = dc_signal(100);
    auto result = stft.compute(sig, sample_rate);

    // Time axis should start at 0
    EXPECT_NEAR(result.times(0), 0.0, 1e-10);

    // Check time spacing (hop = 1 sample)
    double dt = 1.0 / sample_rate;
    for (Eigen::Index t = 0; t < result.times.size() - 1; ++t) {
        EXPECT_NEAR(result.times(t + 1) - result.times(t), dt, 1e-10);
    }
}

TEST_F(StftTest, DcSignalConcentration) {
    // DC signal should concentrate energy at 0 Hz
    Eigen::VectorXd win = hann_window(window_size);
    ssq::Stft stft(win);

    Eigen::VectorXd sig = dc_signal(200, 1.0);
    auto result = stft.compute(sig, sample_rate);

    // Check a time point in the middle (away from boundaries)
    Eigen::Index t_mid = sig.size() / 2;

    // DC bin should have significant energy
    double dc_mag = std::abs(result.stft(0, t_mid));

    // Higher frequency bins should have much less energy
    for (Eigen::Index k = 2; k < result.stft.rows(); ++k) {
        double mag = std::abs(result.stft(k, t_mid));
        EXPECT_LT(mag, dc_mag * 0.1) << "Frequency bin " << k << " should have much less energy than DC";
    }
}

TEST_F(StftTest, SinusoidPeakAtExpectedFrequency) {
    // Single sinusoid should peak at its frequency
    double test_freq = 100.0;  // Hz
    Eigen::VectorXd win = hann_window(window_size);
    ssq::Stft stft(win);

    Eigen::VectorXd sig = sine_signal(500, test_freq, sample_rate);
    auto result = stft.compute(sig, sample_rate);

    // Check a time point in the middle
    Eigen::Index t_mid = sig.size() / 2;

    // Find the frequency bin with maximum magnitude
    Eigen::Index max_bin = 0;
    double max_mag = 0.0;
    for (Eigen::Index k = 0; k < result.stft.rows(); ++k) {
        double mag = std::abs(result.stft(k, t_mid));
        if (mag > max_mag) {
            max_mag = mag;
            max_bin = k;
        }
    }

    // Peak should be near the expected frequency
    double peak_freq = result.frequencies(max_bin);
    double df = sample_rate / static_cast<double>(window_size);
    EXPECT_NEAR(peak_freq, test_freq, df) << "Peak frequency should be near " << test_freq << " Hz";
}

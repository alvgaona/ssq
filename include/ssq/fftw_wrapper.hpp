#ifndef SSQ_FFTW_WRAPPER_HPP
#define SSQ_FFTW_WRAPPER_HPP

#include <cstddef>
#include <fftw3.h>
#include <memory>
#include <mutex>

namespace ssq {

// Custom deleter for fftw_plan
struct FftwPlanDeleter {
    void operator()(fftw_plan* plan) const {
        if (plan && *plan) {
            fftw_destroy_plan(*plan);
        }
        delete plan;
    }
};

// RAII wrapper for fftw_plan
class FftwPlan {
public:
    FftwPlan() : plan_(nullptr) {}

    explicit FftwPlan(fftw_plan plan) : plan_(new fftw_plan(plan), FftwPlanDeleter{}) {}

    fftw_plan get() const {
        return plan_ ? *plan_ : nullptr;
    }

    explicit operator bool() const {
        return plan_ && *plan_;
    }

private:
    std::unique_ptr<fftw_plan, FftwPlanDeleter> plan_;
};

// Custom deleter for fftw_malloc'd memory
template <typename T>
struct FftwArrayDeleter {
    void operator()(T* ptr) const {
        if (ptr) {
            fftw_free(ptr);
        }
    }
};

// RAII wrapper for fftw-allocated arrays
template <typename T>
class FftwArray {
public:
    FftwArray() : data_(nullptr), size_(0) {}

    explicit FftwArray(size_t size)
        : data_(static_cast<T*>(fftw_malloc(size * sizeof(T))), FftwArrayDeleter<T>{}), size_(size) {
        if (!data_) {
            throw std::bad_alloc();
        }
    }

    T* get() {
        return data_.get();
    }

    const T* get() const {
        return data_.get();
    }

    T& operator[](size_t i) {
        return data_.get()[i];
    }

    const T& operator[](size_t i) const {
        return data_.get()[i];
    }

    size_t size() const {
        return size_;
    }

private:
    std::unique_ptr<T, FftwArrayDeleter<T>> data_;
    size_t size_;
};

// Singleton manager for thread-safe FFTW plan creation
class FftwManager {
public:
    static FftwManager& instance();

    // Create a forward DFT plan (real to complex)
    FftwPlan create_r2c_plan(int n, double* in, fftw_complex* out, unsigned flags = FFTW_ESTIMATE);

    // Create an inverse DFT plan (complex to real)
    FftwPlan create_c2r_plan(int n, fftw_complex* in, double* out, unsigned flags = FFTW_ESTIMATE);

    // Create a forward DFT plan (complex to complex)
    FftwPlan create_dft_plan(int n, fftw_complex* in, fftw_complex* out, int sign, unsigned flags = FFTW_ESTIMATE);

    // Execute a plan
    void execute(const FftwPlan& plan);

    // Cleanup all FFTW resources (call at program exit if needed)
    void cleanup();

private:
    FftwManager();
    ~FftwManager();

    FftwManager(const FftwManager&) = delete;
    FftwManager& operator=(const FftwManager&) = delete;

    std::mutex mutex_;
};

}  // namespace ssq

#endif  // SSQ_FFTW_WRAPPER_HPP

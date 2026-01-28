#ifndef SSQ_FFTW_WRAPPER_HPP
#define SSQ_FFTW_WRAPPER_HPP

#include <cstddef>
#include <fftw3.h>
#include <memory>
#include <mutex>
#include <unordered_map>

namespace ssq {

// Custom deleter for fftw_plan
struct FftwPlanDeleter {
    void operator()(fftw_plan_s* plan) const {
        if (plan) {
            fftw_destroy_plan(plan);
        }
    }
};

// RAII wrapper for fftw_plan using shared_ptr for cache sharing
class FftwPlan {
public:
    FftwPlan() : plan_(nullptr) {}

    explicit FftwPlan(fftw_plan plan) : plan_(plan, FftwPlanDeleter{}) {}

    fftw_plan get() const { return plan_.get(); }

    explicit operator bool() const { return plan_ != nullptr; }

private:
    std::shared_ptr<fftw_plan_s> plan_;
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

// Plan types for cache key
enum class PlanType { R2C, C2R, DFT_FORWARD, DFT_BACKWARD };

// Cache key for FFTW plans
struct PlanKey {
    int size;
    PlanType type;

    bool operator==(const PlanKey& other) const { return size == other.size && type == other.type; }
};

struct PlanKeyHash {
    std::size_t operator()(const PlanKey& key) const {
        return std::hash<int>()(key.size) ^ (std::hash<int>()(static_cast<int>(key.type)) << 1);
    }
};

// Singleton manager for thread-safe FFTW plan creation with caching
class FftwManager {
public:
    static FftwManager& instance();

    // Get or create plans (cached by size and type)
    FftwPlan get_r2c_plan(int n, double* in, fftw_complex* out);
    FftwPlan get_c2r_plan(int n, fftw_complex* in, double* out);
    FftwPlan get_dft_plan(int n, fftw_complex* in, fftw_complex* out, int sign);

    // Execute plan with new arrays (uses cached plan)
    void execute_r2c(int n, double* in, fftw_complex* out);
    void execute_c2r(int n, fftw_complex* in, double* out);
    void execute_dft(int n, fftw_complex* in, fftw_complex* out, int sign);

    // Legacy methods (now use caching internally)
    FftwPlan create_r2c_plan(int n, double* in, fftw_complex* out, unsigned flags = FFTW_ESTIMATE);
    FftwPlan create_c2r_plan(int n, fftw_complex* in, double* out, unsigned flags = FFTW_ESTIMATE);
    FftwPlan create_dft_plan(int n, fftw_complex* in, fftw_complex* out, int sign, unsigned flags = FFTW_ESTIMATE);
    void execute(const FftwPlan& plan);

    // Cleanup all FFTW resources
    void cleanup();

private:
    FftwManager();
    ~FftwManager();

    FftwManager(const FftwManager&) = delete;
    FftwManager& operator=(const FftwManager&) = delete;

    std::mutex mutex_;
    std::unordered_map<PlanKey, FftwPlan, PlanKeyHash> plan_cache_;
};

}  // namespace ssq

#endif  // SSQ_FFTW_WRAPPER_HPP

#ifndef SOKUDO_CUDA_HELPER_H
#define SOKUDO_CUDA_HELPER_H
#include <exception>
#include <string>

class CudaException : public std::exception {
private:
    std::string msg;
public:
    CudaException(const std::string &msg) {
        this->msg = "sokudo.cuda: " + msg;
    }

    const char* what() const noexcept override {
        return msg.c_str();
    }
};

class CudaAbstractTask {
private:
    struct CudaAllocation {
        void *device_ptr;
        CudaAllocation *prev;
    };

    void *_stream;
    void *_cublas_handle;
    CudaAllocation *_top{};
public:
    CudaAbstractTask(void *stream, void *cublasHandle = nullptr);

    CudaAbstractTask(const CudaAbstractTask &c) : _top(0) {
        _stream = c._stream;
        _top = c._top;
        _cublas_handle = c._cublas_handle;
    }

    CudaAbstractTask &operator=(const CudaAbstractTask &c) {
        if (this != &c) {
            _stream = c._stream;
            _top = c._top;
            _cublas_handle = c._cublas_handle;
        }

        return *this;
    }

    CudaAbstractTask &operator<<(void *device_ptr) {
        auto *alloc = new CudaAllocation;
        alloc->device_ptr = device_ptr;
        alloc->prev = _top;
        _top = alloc;
        return *this;
    }

    void destroy();

    void sync();
};

class CudaError {
public:
    void operator<<(int err);
};

class CublasError {
public:
    void operator<<(int err);
};

#endif

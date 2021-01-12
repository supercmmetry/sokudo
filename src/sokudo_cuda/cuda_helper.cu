
#include "cuda_helper.h"
#include <cuda_runtime.h>
#include <cublas.h>
#include <device_launch_parameters.h>

CudaAbstractTask::CudaAbstractTask(void *stream, void *cublas_handle) {
    _stream = stream;
    _cublas_handle = cublas_handle;
}

void CudaAbstractTask::destroy() {
    CudaError err;
    CublasError berr;
    while(_top) {
        err << cudaFree(_top->device_ptr);
        CudaAllocation *tmp = _top;
        _top = _top->prev;
        delete tmp;
    }

    err << cudaStreamDestroy((cudaStream_t) _stream);
    if (_cublas_handle) {
        berr << cublasDestroy_v2((cublasHandle_t) _cublas_handle);
    }
}

void CudaAbstractTask::sync() {
    if (cudaStreamSynchronize((cudaStream_t) _stream) != cudaSuccess) {
        throw CudaException("cudaStreamSynchronize failed");
    }
}

void CudaError::operator<<(int err) {
    auto error = (cudaError_t) err;
    if (error != cudaSuccess) {
        throw CudaException(std::string("CUDA operation failed with error=") + std::to_string(err));
    }
}

void CublasError::operator<<(int err) {
    auto error = (cublasStatus_t) err;
    if (error != CUBLAS_STATUS_SUCCESS) {
        throw CudaException(std::string("cuBLAS operation failed with error=") + std::to_string(err));
    }
}

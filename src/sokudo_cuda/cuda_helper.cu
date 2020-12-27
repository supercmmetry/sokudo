
#include "cuda_helper.h"
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

CudaAbstractTask::CudaAbstractTask(void *stream) {
    _stream = stream;
}

void CudaAbstractTask::destroy() {
    while(_top) {
        cudaFree(_top->device_ptr);
        CudaAllocation *tmp = _top;
        _top = _top->prev;
        delete tmp;
    }

    cudaStreamDestroy((cudaStream_t) _stream);
}

void CudaAbstractTask::sync() {
    if (cudaStreamSynchronize((cudaStream_t) _stream) != cudaSuccess) {
        throw CudaException("cudaStreamSynchronize failed");
    }
}

#ifndef SOKUDO_TASK_H
#define SOKUDO_TASK_H

#include <vector>
#include <shared_mutex>
#include <errors.h>

#ifdef SOKUDO_OPENCL

#include <sokudo_opencl/cl_helper.h>

#endif

#ifdef SOKUDO_CUDA

#include <sokudo_cuda/cuda_helper.h>

#endif

namespace sokudo {
    enum TaskExecutor {
        CUDA,
        OPENCL,
        CPU
    };

    template<class Type>
    class DataBuffer {
    private:
        Type *_data{};
        uint64_t _size{};
        std::shared_mutex _mutex;
        uint64_t *_refs{};

        inline void increment_ref() {
            *_refs = *_refs + 1;
        }

        inline void decrement_ref() {
            if (*_refs) {
                *_refs = *_refs - 1;
            }
        }

    public:
        DataBuffer() {
            _refs = new uint64_t;
        }

        explicit DataBuffer(std::vector<Type> data) : DataBuffer() {
            _size = data.size();
            _data = new Type[_size];

            for (uint64_t i = 0; i < _size; i++) {
                _data[i] = data[i];
            }
        }

        DataBuffer(Type *data, uint64_t size) : DataBuffer() {
            _size = size;
            _data = new Type[_size];

            for (uint64_t i = 0; i < _size; i++) {
                _data[i] = data[i];
            }
        }

        DataBuffer(const DataBuffer<Type> &buffer) {
            buffer._mutex.lock();
            _mutex = buffer._mutex;
            _refs = buffer._refs;
            _data = buffer._data;
            _size = buffer._size;

            increment_ref();
            _mutex.unlock();
        }

        DataBuffer<Type> &operator=(const DataBuffer<Type> &buffer) {
            if (this != &buffer) {
                buffer._mutex.lock();
                _mutex = buffer._mutex;
                _refs = buffer._refs;
                _data = buffer._data;
                _size = buffer._size;

                increment_ref();
                _mutex.unlock();
            }

            return *this;
        }

        Type &operator[](uint64_t index) {
            if (index >= _size) {
                throw sokudo::errors::InvalidOperationException("DataBuffer index out of bounds");
            }

            return _data[index];
        }

        [[nodiscard]] uint64_t size() const {
            return _size;
        }

        [[nodiscard]] uint64_t bsize() const {
            return _size * sizeof(Type);
        }

        Type *inner() const {
            return _data;
        }

        ~DataBuffer() {
            _mutex.lock();
            if (!*_refs) {
                delete[] _data;
                delete _refs;
            } else {
                decrement_ref();
            }
            _mutex.unlock();
        }
    };

    template<class Type>
    class DataValue {
    private:
        Type *_data{};
        std::shared_mutex _mutex;
        uint64_t *_refs{};

        inline void increment_ref() {
            *_refs = *_refs + 1;
        }

        inline void decrement_ref() {
            if (*_refs) {
                *_refs = *_refs - 1;
            }
        }

    public:
        DataValue() {
            _refs = new uint64_t;
        }

        explicit DataValue(Type data) : DataValue() {
            _data = new Type[1];
            *_data = data;
        }

        DataValue(const DataValue<Type> &value) {
            value._mutex.lock();
            _mutex = value._mutex;
            _refs = value._refs;
            _data = value._data;

            increment_ref();
            _mutex.unlock();
        }

        DataValue<Type> &operator=(const DataValue<Type> &value) {
            if (this != &value) {
                value._mutex.lock();
                _mutex = value._mutex;
                _refs = value._refs;
                _data = value._data;

                increment_ref();
                _mutex.unlock();
            }

            return *this;
        }

        friend bool operator==(const DataValue<Type> &value1, const DataValue<Type> &value2) {
            return *(value1._data) == *(value2._data);
        }

        friend bool operator==(const Type &value1, const DataValue<Type> &value2) {
            return value1 == *(value2._data);
        }

        [[nodiscard]] uint64_t size() const {
            return 1;
        }

        [[nodiscard]] uint64_t bsize() const {
            return sizeof(Type);
        }

        Type *inner() const {
            return _data;
        }

        ~DataValue() {
            _mutex.lock();
            if (!*_refs) {
                delete[] _data;
                delete _refs;
            } else {
                decrement_ref();
            }
            _mutex.unlock();
        }
    };

    class Task {
    protected:
        TaskExecutor _executor = TaskExecutor::CPU;
    public:
        virtual void sync() = 0;
    };

#ifdef SOKUDO_CUDA

    class CUDATask : public Task {
    private:
        CudaAbstractTask _task;
    public:
        explicit CUDATask(const CudaAbstractTask &task) : _task(nullptr) {
            _executor = CUDA;
            _task = task;
        }

        void sync() override {
            _task.sync();
            _task.destroy();
        }
    };

#endif

#ifdef SOKUDO_OPENCL

    class CLTask : public Task {
    private:
        cl::CommandQueue _queue;
    public:
        explicit CLTask(const cl::CommandQueue &queue) {
            _queue = queue;
            _executor = TaskExecutor::OPENCL;
        }

        void sync() override {
            _queue.flush();
            _queue.finish();
        }
    };

#endif

    inline TaskExecutor get_fallback_executor(TaskExecutor executor) {
        auto fallback_executor = executor;

#if defined(SOKUDO_OPENCL) and defined(SOKUDO_CUDA)
        if (fallback_executor == OPENCL) {
            if (sokudo::opencl::DeviceProvider::empty()) {
                fallback_executor = CUDA;
            }
        }
#endif
#if defined(SOKUDO_OPENCL) and !defined(SOKUDO_CUDA)
        if (fallback_executor == OPENCL) {
            if (sokudo::opencl::DeviceProvider::empty()) {
                fallback_executor = CPU;
            }
        }
#endif
#if !defined(SOKUDO_OPENCL) and defined(SOKUDO_CUDA)
        if (fallback_executor == OPENCL) {
            fallback_executor = CUDA;
        }
#endif
#if !defined(SOKUDO_OPENCL) and !defined(SOKUDO_CUDA)
        fallback_executor = CPU;
#endif

        return fallback_executor;
    }

    class TaskGraph {

    };
}

#endif

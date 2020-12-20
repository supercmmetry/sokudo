#ifndef SOKUDO_TASK_H
#define SOKUDO_TASK_H

#include <shared_mutex>
#include "kernel.h"
#include "errors.h"

#ifdef SOKUDO_OPENCL

#include <CL/cl2.hpp>

#endif

namespace sokudo {
    enum TaskExecutor {
        CUDA,
        OPENCL,
        CPU
    };

    class GenericTask {
    public:
        virtual void sync() = 0;
    };

    enum BufferPermissions {
        READ = 0x1,
        WRITE = 0x2
    };

    template<class Type>
    class DataBuffer {
    private:
        uint64_t _permissions{};
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

        DataBuffer(std::vector<Type> data) : DataBuffer() {
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
                _permissions = buffer._permissions;
                _size = buffer._size;

                increment_ref();
                _mutex.unlock();
            }

            return *this;
        }

        Type& operator[](uint64_t index) {
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

        Type* inner() const {
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
}

#endif

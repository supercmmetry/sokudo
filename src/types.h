#ifndef SOKUDO_TYPES_H
#define SOKUDO_TYPES_H

#include <memory>
#include <vector>
#include <mutex>
#include <errors.h>

namespace sokudo {
    template<class Type>
    class Buffer {
    private:
        std::string _name = "BUF_UNDEFINED";
        Type *_data{};
        uint64_t _size{};
        std::shared_ptr<std::mutex> _mutex;
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
        Buffer() {
            _refs = new uint64_t;
            _mutex = std::make_shared<std::mutex>();
        }

        explicit Buffer(std::vector<Type> data) : Buffer() {
            _size = data.size();
            _data = new Type[_size];

            for (uint64_t i = 0; i < _size; i++) {
                _data[i] = data[i];
            }
        }

        Buffer(Type *data, uint64_t size, bool clone = true) : Buffer() {
            _size = size;

            if (clone) {
                _data = new Type[_size];

                for (uint64_t i = 0; i < _size; i++) {
                    _data[i] = data[i];
                }
            } else {
                _data = data;
            }
        }

        Buffer(Buffer<Type> &buffer) {
            buffer._mutex->lock();
            _mutex = buffer._mutex;
            _refs = buffer._refs;
            _data = buffer._data;
            _size = buffer._size;
            _name = buffer._name;

            increment_ref();
            _mutex->unlock();
        }

        Buffer<Type> &operator=(const Buffer<Type> &buffer) {
            if (this != &buffer) {
                buffer._mutex->lock();
                _mutex = buffer._mutex;
                _refs = buffer._refs;
                _data = buffer._data;
                _size = buffer._size;
                _name = buffer._name;

                increment_ref();
                _mutex->unlock();
            }

            return *this;
        }

        Type &operator[](uint64_t index) {
            _mutex->lock();
            if (index >= _size) {
                throw sokudo::errors::InvalidOperationException("DataBuffer index out of bounds");
            }

            auto v = _data[index];
            _mutex->unlock();
            return v;
        }

        [[nodiscard]] uint64_t size() const {
            _mutex->lock();
            auto sz = _size;
            _mutex->unlock();
            return sz;
        }

        [[nodiscard]] uint64_t bsize() const {
            _mutex->lock();
            auto sz = _size * sizeof(Type);;
            _mutex->unlock();
            return sz;
        }

        Type *inner() const {
            _mutex->lock();
            auto data = _data;
            _mutex->unlock();
            return data;
        }

        Buffer<Type> &to_named(const std::string &name) {
            _mutex->lock();
            _name = name;
            _mutex->unlock();
            return *this;
        }

        [[nodiscard]] std::string name() const {
            _mutex->lock();
            auto name = _name;
            _mutex->unlock();
            return name;
        }

        ~Buffer() {
            _mutex->lock();
            if (!*_refs) {
                delete[] _data;
                delete _refs;
            } else {
                decrement_ref();
            }
            _mutex->unlock();
        }
    };

    template<class Type>
    class Value {
    private:
        std::string _name = "VAL_UNDEFINED";
        Type *_data{};
        std::shared_ptr<std::mutex> _mutex;
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
        Value() {
            _refs = new uint64_t;
            _mutex = std::make_shared<std::mutex>();
        }

        Value(Type data) : Value() {
            _data = new Type[1];
            *_data = data;
        }

        Value(const Value<Type> &value) {
            value._mutex->lock();
            _mutex = value._mutex;
            _refs = value._refs;
            _data = value._data;
            _name = value._name;

            increment_ref();
            _mutex->unlock();
        }

        Value<Type> &operator=(const Value<Type> &value) {
            if (this != &value) {
                value._mutex->lock();
                _mutex = value._mutex;
                _refs = value._refs;
                _data = value._data;
                _name = value._name;

                increment_ref();
                _mutex->unlock();
            }

            return *this;
        }

        Value<Type> &operator=(const Type &value) {
            _mutex->lock();
            *_data = value;
            _mutex->unlock();
            return *this;
        }

        friend bool operator==(const Value<Type> &value1, const Value<Type> &value2) {
            value1._mutex->lock();
            auto lhs = *(value1._data);
            value1._mutex->unlock();

            value2._mutex->lock();
            auto rhs = *(value2._data);
            value2._mutex->unlock();

            return lhs == rhs;
        }

        friend bool operator==(const Type &value1, const Value<Type> &value2) {
            value2._mutex->lock();
            auto rhs = *(value2._data);
            value2._mutex->unlock();

            return rhs == value1;
        }

        [[nodiscard]] uint64_t size() const {
            return 1;
        }

        [[nodiscard]] uint64_t bsize() const {
            return sizeof(Type);
        }

        Type *inner() const {
            _mutex->lock();
            auto data = _data;
            _mutex->unlock();
            return data;
        }

        Type value() const {
            _mutex->lock();
            auto v = *_data;
            _mutex->unlock();
            return v;
        }

        Value &to_named(const std::string &name) {
            _mutex->lock();
            _name = name;
            _mutex->unlock();
            return *this;
        }

        [[nodiscard]] std::string name() const {
            _mutex->lock();
            auto name = _name;
            _mutex->unlock();
            return name;
        }

        ~Value() {
            _mutex->lock();
            if (!*_refs) {
                delete[] _data;
                delete _refs;
            } else {
                decrement_ref();
            }
            _mutex->unlock();
        }
    };

    struct float2 {
        float x;
        float y;
    };

    struct double2 {
        double x;
        double y;
    };
}

#endif

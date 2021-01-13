#ifndef SOKUDO_TASK_H
#define SOKUDO_TASK_H

#include <vector>
#include <shared_mutex>
#include <types.h>
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

    class Task {
    protected:
        TaskExecutor _executor = TaskExecutor::CPU;
        Kernel _kernel = KERNEL_UNDEFINED;
        std::string _name = "undefined";
        std::unordered_map<std::string, std::string> _params;
        std::vector<uint64_t> _input_shape;
        std::vector<uint64_t> _output_shape;
    public:
        virtual void sync() = 0;

        virtual ~Task() = default;

        Task *then(Task *t) {
            sync();
            return t;
        }

        Task *set(
                Kernel kernel,
                const std::string &name = "undefined"
        ) {
            _kernel = kernel;
            _name = name;
            return this;
        }

        std::unordered_map<std::string, std::string> &params() {
            return _params;
        }

        std::vector<uint64_t> &input_shape() {
            return _input_shape;
        }

        std::vector<uint64_t> &output_shape() {
            return _output_shape;
        }
    };

    class TaskGroup {
    private:
        std::vector<Task *> _tasks;
    public:
        enum TASKGROUP {
            SYNC
        };

        TaskGroup() = default;

        TaskGroup(const TaskGroup &task_group) {
            _tasks = task_group._tasks;
        }

        TaskGroup &operator=(const TaskGroup &task_group) {
            if (this != &task_group) {
                _tasks = task_group._tasks;
            }

            return *this;
        }

        void sync() {
            for (auto &_task : _tasks) {
                _task->sync();
                delete _task;
            }
        }

        TaskGroup &operator<<(Task *task) {
            _tasks.push_back(task);
            return *this;
        }

        TaskGroup &operator<<(TASKGROUP marker) {
            if (marker == SYNC) {
                sync();
            }
            return *this;
        }

        TaskGroup &add(Task *task) {
            _tasks.push_back(task);
            return *this;
        }

        TaskGroup &then(TaskGroup &task_group) {
            sync();
            return task_group;
        }

        TaskGroup then(TaskGroup task_group) {
            sync();
            return task_group;
        }

        TaskGroup operator>>(TaskGroup task_group) {
            sync();
            return task_group;
        }
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
}

#endif

#ifndef SOKUDO_TASK_H
#define SOKUDO_TASK_H

#include <functional>
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
    };

    class TaskGroup {
    private:
        std::vector<Task *> _tasks;
    public:
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
            _tasks.clear();
        }

        TaskGroup &add(Task *task) {
            _tasks.push_back(task);
            return *this;
        }

        TaskGroup &operator()(Task *task) {
            _tasks.push_back(task);
            return *this;
        }

        TaskGroup &operator()() {
            sync();
            return *this;
        }

        TaskGroup then(const std::function<TaskGroup ()> &func) {
            sync();
            return func();
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

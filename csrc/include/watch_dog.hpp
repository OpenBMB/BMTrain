#pragma once
#include <thread>
#include <mutex>
#include <queue>
#include <vector>
#include <condition_variable>
#include <chrono>
#include <iostream>
#include <cuda_runtime_api.h>

class CUDAEvent {

private:
    cudaEvent_t startCUDAEvent_;
    cudaEvent_t endCUDAEvent_;
    std::chrono::time_point<std::chrono::steady_clock> startTime_;
    const std::string name;
    cudaStream_t stream;
    bool isStarted;

public:
    friend class WatchDog;
    CUDAEvent(cudaStream_t stream, const std::string& name)
        : stream(stream)
        , name(name)
        , isStarted(false) { }
    void recordStart() {
        cudaEventCreate(&startCUDAEvent_);
        cudaEventRecord(startCUDAEvent_, stream);
    }
    void recordEnd() {
        cudaEventCreate(&endCUDAEvent_);
        cudaEventRecord(endCUDAEvent_, stream);
    }

    void updateStartTime() {
        if (!isStarted) {
            startTime_ = std::chrono::steady_clock::now();
            isStarted = true;
        }
    }

    void destroyAll() {
        cudaEventDestroy(startCUDAEvent_);
        cudaEventDestroy(endCUDAEvent_);
    }
};

class WatchDog {
private:
    std::chrono::milliseconds timeout;

    std::mutex mutex_;
    std::queue<CUDAEvent> events_;

    std::condition_variable event_notifier_;
    std::condition_variable stop_notifier_;
    std::thread thread_;
    bool finished_;

public:
    WatchDog(int timeout = 18 * 60 * 1000)
        : timeout(timeout)
        , finished_(false)
    {
        thread_ = std::thread(&WatchDog::execution_loop, this);
    }
    ~WatchDog() {
        std::unique_lock<std::mutex> lock(mutex_);
        finished_ = true;
        event_notifier_.notify_all();
        lock.unlock();
        thread_.join();
    }
    void watch(const CUDAEvent& evt)
    {
        std::unique_lock<std::mutex> lock(mutex_);
        events_.emplace(std::move(evt));
        event_notifier_.notify_all();
    }

    void wait()
    {
        std::unique_lock<std::mutex> lock(mutex_);
        stop_notifier_.wait(lock, [this] { return this->events_.empty(); });
        return;
    }

    bool isReady(const cudaEvent_t& evt)
    {
        cudaError_t err = cudaEventQuery(evt);
        if (err == cudaSuccess) {
            return true;
        } else if (err != cudaErrorNotReady) {
            throw std::logic_error(
                std::string("CUDA Error: ") + cudaGetErrorString(err));
        } else {
            cudaGetLastError();
        }
        return false;
    }

    void execution_loop()
    {
        std::unique_lock<std::mutex> lock(mutex_);
        for (;;) {

            event_notifier_.wait(lock, [this] { return !this->events_.empty() || finished_; });

            if (finished_) {
                stop_notifier_.notify_all();
                return;
            }
            auto& event = events_.front();

            if (isReady(event.startCUDAEvent_)) {
                event.updateStartTime();
                if (isReady(event.endCUDAEvent_)) {
                    event.destroyAll();
                    events_.pop();
                } else {
                    if (std::chrono::steady_clock::now() - event.startTime_ > timeout) {
                        std::cerr << "NCCL timeout detected: " << event.name << std::endl;
                        throw std::runtime_error("NCCL timeout");
                    }
                }
            }
            lock.unlock();
            std::this_thread::sleep_for(
                std::chrono::milliseconds(1));
            lock.lock();
            if (this->events_.empty()) {
                stop_notifier_.notify_all();
            }
        }
    }
};
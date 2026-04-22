#pragma once

#include <nlohmann/json.hpp>
#include <phos/phos.h>

#include <sys/types.h>

#include <atomic>
#include <cstdint>
#include <mutex>
#include <string>
#include <thread>
#include <unordered_map>

namespace mp_studio {

struct RunHandle {
    std::string run_id;
    pid_t pid;
    int monitor_port;
};

class TrainingManager {
public:
    explicit TrainingManager(phos::Window& win);
    ~TrainingManager();

    // Spawn a new training sidecar. config is the JSON training config.
    // Returns a RunHandle with the assigned run_id and monitor port.
    RunHandle start(const nlohmann::json& config);

    // Send SIGTERM to the sidecar for run_id. Returns false if not found.
    // After SIGTERM, a background cleanup thread waits up to 10s, SIGKILLs if
    // needed, closes pipe fds, joins reader threads, and erases the map entry.
    bool stop(const std::string& run_id);

    // List all active runs as JSON array.
    nlohmann::json list() const;

    TrainingManager(const TrainingManager&) = delete;
    TrainingManager& operator=(const TrainingManager&) = delete;

private:
    struct ActiveRun {
        std::string run_id;
        pid_t pid = -1;
        int monitor_port = 0;
        int stdout_fd = -1;              // read end of stdout pipe
        int stderr_fd = -1;              // read end of stderr pipe
        std::thread stdout_thread;
        std::thread stderr_thread;

        ActiveRun() = default;
        ActiveRun(ActiveRun&&) = default;
        ActiveRun& operator=(ActiveRun&&) = default;
        ActiveRun(const ActiveRun&) = delete;
        ActiveRun& operator=(const ActiveRun&) = delete;
    };

    phos::Window& win_;
    mutable std::mutex mutex_;
    std::unordered_map<std::string, ActiveRun> runs_;

    // Monotonic counter for run-id disambiguation (protects against
    // millisecond collisions when two runs start within the same ms).
    std::atomic<uint64_t> run_counter_{0};

    // Self-pipe for async-signal-safe SIGCHLD handling.
    // sigchld_pipe_[0] = read end (reaper thread reads wakeups)
    // sigchld_pipe_[1] = write end (signal handler writes one byte)
    static int sigchld_pipe_[2];
    static TrainingManager* instance_;  // referenced by static signal handler

    std::atomic<bool> shutting_down_{false};
    std::thread reaper_thread_;

    // Helpers.
    static int find_free_port();
    std::string make_run_id();
    void read_pipe(int fd, const std::string& run_id, const char* stream);

    // SIGCHLD plumbing.
    void start_reaper();
    void reaper_loop();
    static void sigchld_handler(int);  // async-signal-safe: writes one byte
    void on_child_exit(pid_t pid, int status);  // runs on reaper thread

    // Resolve the mud-puppy sidecar path. Throws if not found.
    static std::string resolve_sidecar_path();

    // Close fds, join reader threads. Caller must NOT hold mutex_.
    static void cleanup_run(ActiveRun& run);
};

}  // namespace mp_studio

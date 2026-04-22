#pragma once

#include <nlohmann/json.hpp>
#include <phos/phos.h>

#include <atomic>
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
    bool stop(const std::string& run_id);

    // List all active runs as JSON array.
    nlohmann::json list() const;

    TrainingManager(const TrainingManager&) = delete;
    TrainingManager& operator=(const TrainingManager&) = delete;

private:
    struct ActiveRun {
        std::string run_id;
        pid_t pid;
        int monitor_port;
        int stderr_fd;       // read end of pipe
        std::thread log_thread;
    };

    phos::Window& win_;
    mutable std::mutex mutex_;
    std::unordered_map<std::string, ActiveRun> runs_;

    // Allocate a free TCP port by binding to :0, reading it, closing.
    static int find_free_port();

    // Generate a short unique run ID.
    static std::string make_run_id();

    // Background thread: reads stderr from child, emits log_line events.
    void read_stderr(int fd, const std::string& run_id);

    // SIGCHLD handler reaps zombies and emits run_complete.
    static void install_sigchld(TrainingManager* self);
    static TrainingManager* instance_;  // one manager per process

    static void sigchld_handler(int);
    void on_child_exit(pid_t pid, int status);
};

}  // namespace mp_studio

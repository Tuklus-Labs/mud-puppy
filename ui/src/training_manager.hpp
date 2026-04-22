#pragma once

#include <nlohmann/json.hpp>
#include <phos/phos.h>

#include <sys/types.h>

#include <atomic>
#include <cstdint>
#include <functional>
#include <mutex>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

namespace mp_studio {

// Callback fired when a sidecar announces its bound monitor port via its
// stdout marker line "MUD_PUPPY_MONITOR_PORT=<n>". Runs on the stdout
// reader thread; handler must be thread-safe and non-blocking.
using PortReadyCb = std::function<void(const std::string& run_id, int port)>;

// Run status as tracked by TrainingManager. Updated by reaper when a child exits.
enum class RunStatus {
    Running,
    Complete,
    Failed,
    Stopped,
};

struct RunHandle {
    std::string run_id;
    pid_t pid;
    int monitor_port;
};

class TrainingManager {
public:
    explicit TrainingManager(phos::Window& win);
    ~TrainingManager();

    // Install a callback invoked when a sidecar announces its bound
    // monitor port. Typically WsBridge::connect. Must be set before
    // start() is called; calling start() before this is set means the
    // port announcement is silently dropped.
    void set_port_ready_callback(PortReadyCb cb);

    // Spawn a new training sidecar. config is the JSON training config.
    // Returns a RunHandle with the assigned run_id. Note: monitor_port
    // is 0 at return time — the child binds to port 0 and announces its
    // actual port via the stdout marker, which fires the PortReadyCb
    // on the stdout reader thread.
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

        // Config snapshot stored at start() time for list() / RunSummary.
        std::string model;
        std::string method;
        std::string dataset;
        int64_t start_time_ms = 0;       // ms since epoch at fork time
        int64_t end_time_ms = 0;         // 0 = still running
        int exit_code = 0;
        RunStatus status = RunStatus::Running;

        ActiveRun() = default;
        ActiveRun(ActiveRun&&) = default;
        ActiveRun& operator=(ActiveRun&&) = default;
        ActiveRun(const ActiveRun&) = delete;
        ActiveRun& operator=(const ActiveRun&) = delete;
    };

    // Completed runs kept for RunSummary display (capped at MAX_COMPLETED_RUNS).
    static constexpr size_t MAX_COMPLETED_RUNS = 20;

    phos::Window& win_;
    mutable std::mutex mutex_;
    std::unordered_map<std::string, ActiveRun> runs_;
    // Completed runs stored as RunSummary JSON, newest first. Capped at MAX_COMPLETED_RUNS.
    std::vector<nlohmann::json> completed_runs_;
    PortReadyCb port_ready_cb_;  // fires when child announces its port

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
    std::string make_run_id();
    void read_pipe(int fd, const std::string& run_id, const char* stream);
    // Record a child's bound port and invoke port_ready_cb_.
    void on_port_announced(const std::string& run_id, int port);

    // SIGCHLD plumbing.
    void start_reaper();
    void reaper_loop();
    static void sigchld_handler(int);  // async-signal-safe: writes one byte
    void on_child_exit(pid_t pid, int status);  // runs on reaper thread

    // How the sidecar should be launched.
    struct SidecarLauncher {
        // Program to exec (an absolute path).
        std::string program;
        // Argv prefix to insert between program and user args. Empty
        // for the venv case; ["-m", "mud_puppy.cli"] for the system
        // python fallback so the repo's package is loaded as a module.
        std::vector<std::string> argv_prefix;
        // cwd to chdir into before exec, or empty to keep current cwd.
        // Used by the system-python fallback so Python finds mud_puppy
        // on sys.path via the current directory.
        std::string cwd;
    };

    // Resolve how to launch the mud-puppy training sidecar. Prefers the
    // repo-local venv entry point when installed; falls back to `python3
    // -m mud_puppy.cli` run from the repo root so training works even
    // when the user has not set up the venv. Throws if neither is usable.
    static SidecarLauncher resolve_sidecar_launcher();

    // Close fds, join reader threads. Caller must NOT hold mutex_.
    static void cleanup_run(ActiveRun& run);
};

}  // namespace mp_studio

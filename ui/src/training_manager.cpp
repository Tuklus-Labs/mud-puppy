#include "training_manager.hpp"

#include <phos/phos.h>

#include <arpa/inet.h>
#include <errno.h>
#include <fcntl.h>
#include <netinet/in.h>
#include <pwd.h>
#include <signal.h>
#include <sys/socket.h>
#include <sys/stat.h>
#include <sys/wait.h>
#include <unistd.h>

#include <chrono>
#include <cstring>
#include <sstream>
#include <stdexcept>

namespace mp_studio {

// Fallback port when bind-to-port-0 fails; WsBridge's retry loop will still
// surface the failure via run_complete if this port is already taken.
constexpr int DEFAULT_MONITOR_PORT_FALLBACK = 5980;

// Static members.
TrainingManager* TrainingManager::instance_ = nullptr;
int TrainingManager::sigchld_pipe_[2] = {-1, -1};

// ---------------------------------------------------------------------------
// Construction / destruction
// ---------------------------------------------------------------------------

TrainingManager::TrainingManager(phos::Window& win) : win_(win) {
    instance_ = this;

    // Create the self-pipe for SIGCHLD wakeups.
    if (::pipe2(sigchld_pipe_, O_CLOEXEC | O_NONBLOCK) != 0) {
        throw std::runtime_error("TrainingManager: pipe2(sigchld) failed: " +
                                 std::string(strerror(errno)));
    }

    // Install SIGCHLD handler that only writes one byte.
    struct sigaction sa{};
    sa.sa_handler = &TrainingManager::sigchld_handler;
    sigemptyset(&sa.sa_mask);
    sa.sa_flags = SA_RESTART | SA_NOCLDSTOP;
    if (::sigaction(SIGCHLD, &sa, nullptr) != 0) {
        throw std::runtime_error("TrainingManager: sigaction(SIGCHLD) failed: " +
                                 std::string(strerror(errno)));
    }

    start_reaper();
}

TrainingManager::~TrainingManager() {
    // Move all active runs out from under the lock so we can clean them up
    // without holding mutex_ while joining threads.
    std::unordered_map<std::string, ActiveRun> to_clean;
    {
        std::lock_guard<std::mutex> lock(mutex_);
        to_clean = std::move(runs_);
        runs_.clear();
    }

    // SIGTERM every live child in parallel, then poll per-run with a 10s budget.
    for (auto& [id, run] : to_clean) {
        if (run.pid > 0) {
            ::kill(run.pid, SIGTERM);
        }
    }

    // Per-run 10s deadline (not a shared budget across all runs).
    for (auto& [id, run] : to_clean) {
        if (run.pid <= 0) continue;
        auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(10);
        bool reaped = false;
        while (std::chrono::steady_clock::now() < deadline) {
            int status;
            pid_t r = ::waitpid(run.pid, &status, WNOHANG);
            if (r == run.pid) { reaped = true; break; }
            if (r < 0 && errno == ECHILD) { reaped = true; break; }
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
        }
        if (!reaped) {
            ::kill(run.pid, SIGKILL);
            ::waitpid(run.pid, nullptr, 0);
        }
    }

    // Now close pipes (unblocks reader threads) and join them.
    for (auto& [id, run] : to_clean) {
        cleanup_run(run);
    }

    // Stop the reaper thread by closing the self-pipe write end.
    shutting_down_.store(true);
    if (sigchld_pipe_[1] >= 0) {
        ::close(sigchld_pipe_[1]);
        sigchld_pipe_[1] = -1;
    }
    if (reaper_thread_.joinable()) {
        reaper_thread_.join();
    }
    if (sigchld_pipe_[0] >= 0) {
        ::close(sigchld_pipe_[0]);
        sigchld_pipe_[0] = -1;
    }

    // Restore default SIGCHLD handler so a later manager re-install works.
    struct sigaction sa{};
    sa.sa_handler = SIG_DFL;
    sigemptyset(&sa.sa_mask);
    ::sigaction(SIGCHLD, &sa, nullptr);

    instance_ = nullptr;
}

// ---------------------------------------------------------------------------
// SIGCHLD: self-pipe pattern
// ---------------------------------------------------------------------------

void TrainingManager::sigchld_handler(int /*sig*/) {
    // Async-signal-safe body: just write a single byte to the pipe.
    // Ignore EAGAIN/EINTR -- reaper thread will catch up via the waitpid loop.
    if (sigchld_pipe_[1] >= 0) {
        char c = 1;
        // ::write is listed as async-signal-safe by POSIX.
        while (true) {
            ssize_t n = ::write(sigchld_pipe_[1], &c, 1);
            if (n >= 0) break;
            if (errno == EINTR) continue;
            break;  // EAGAIN/EPIPE etc. -- don't block the signal handler.
        }
    }
}

void TrainingManager::start_reaper() {
    reaper_thread_ = std::thread([this]() { reaper_loop(); });
}

void TrainingManager::reaper_loop() {
    // Normal thread context -- free to lock, free to call win_.emit().
    char buf[64];
    while (!shutting_down_.load()) {
        ssize_t n = ::read(sigchld_pipe_[0], buf, sizeof(buf));
        if (n < 0) {
            if (errno == EINTR) continue;
            if (errno == EAGAIN) {
                // Shouldn't happen: read is blocking when O_NONBLOCK not set on
                // read end, but we set it on both. Poll via short sleep.
                std::this_thread::sleep_for(std::chrono::milliseconds(50));
                continue;
            }
            break;  // fd closed or fatal error
        }
        if (n == 0) break;  // EOF: destructor closed write end.

        // Reap every exited child that's ready.
        pid_t pid;
        int status;
        while ((pid = ::waitpid(-1, &status, WNOHANG)) > 0) {
            on_child_exit(pid, status);
        }
    }
}

void TrainingManager::on_child_exit(pid_t pid, int status) {
    // Runs on reaper thread -- safe to lock, safe to emit.
    std::string run_id;
    int exit_code = -1;
    {
        std::lock_guard<std::mutex> lock(mutex_);
        for (auto& [id, run] : runs_) {
            if (run.pid == pid) {
                run.pid = -1;  // mark reaped
                run_id = run.run_id;
                exit_code = WIFEXITED(status)  ? WEXITSTATUS(status)
                          : WIFSIGNALED(status) ? -WTERMSIG(status)
                          : -1;
                break;
            }
        }
    }
    if (!run_id.empty()) {
        win_.emit("run_complete", {{"run_id", run_id}, {"exit_code", exit_code}});
        PHOS_LOG_INFO("TrainingManager: run_id={} exited code={}", run_id, exit_code);
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

int TrainingManager::find_free_port() {
    // Classic bind-to-port-0 trick.  NOTE: there is an unavoidable TOCTOU race
    // between the close(fd) here and the child's call to listen(); another
    // process could grab this port in the window between them.  WsBridge's
    // retry loop (with its retry cap) surfaces the failure via a
    // run_complete error payload.
    //
    // TODO: have the Python child print its actual bound port to stdout on
    // startup and have the parent read the port from the stdout pipe instead
    // of assigning it here.
    int fd = ::socket(AF_INET, SOCK_STREAM, 0);
    if (fd < 0) return DEFAULT_MONITOR_PORT_FALLBACK;

    sockaddr_in addr{};
    addr.sin_family = AF_INET;
    addr.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
    addr.sin_port = 0;

    if (::bind(fd, reinterpret_cast<sockaddr*>(&addr), sizeof(addr)) != 0) {
        ::close(fd);
        return DEFAULT_MONITOR_PORT_FALLBACK;
    }

    socklen_t len = sizeof(addr);
    ::getsockname(fd, reinterpret_cast<sockaddr*>(&addr), &len);
    int port = ntohs(addr.sin_port);
    ::close(fd);
    return port;
}

std::string TrainingManager::make_run_id() {
    // Append a monotonic counter so two runs in the same millisecond don't
    // collide. Format: "run-<ms>-<counter>".
    auto now = std::chrono::system_clock::now().time_since_epoch();
    auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(now).count();
    uint64_t n = run_counter_.fetch_add(1);
    std::ostringstream ss;
    ss << "run-" << ms << "-" << n;
    return ss.str();
}

std::string TrainingManager::resolve_sidecar_path() {
    // 1. $HOME/Projects/mud-puppy/venv/bin/mud-puppy
    // 2. getpwuid()->pw_dir/Projects/mud-puppy/venv/bin/mud-puppy
    // 3. throw
    std::string home;
    if (const char* h = ::getenv("HOME"); h && *h) {
        home = h;
    } else {
        if (struct passwd* pw = ::getpwuid(::getuid()); pw && pw->pw_dir) {
            home = pw->pw_dir;
        } else {
            throw std::runtime_error(
                "cannot resolve HOME for mud-puppy sidecar lookup");
        }
    }

    std::string candidate = home + "/Projects/mud-puppy/venv/bin/mud-puppy";
    if (::access(candidate.c_str(), X_OK) != 0) {
        throw std::runtime_error("mud-puppy sidecar not found at: " + candidate);
    }
    return candidate;
}

void TrainingManager::cleanup_run(ActiveRun& run) {
    if (run.stdout_fd >= 0) { ::close(run.stdout_fd); run.stdout_fd = -1; }
    if (run.stderr_fd >= 0) { ::close(run.stderr_fd); run.stderr_fd = -1; }
    if (run.stdout_thread.joinable()) run.stdout_thread.join();
    if (run.stderr_thread.joinable()) run.stderr_thread.join();
}

// ---------------------------------------------------------------------------
// start / stop / list
// ---------------------------------------------------------------------------

RunHandle TrainingManager::start(const nlohmann::json& config) {
    std::string venv_python = resolve_sidecar_path();

    int monitor_port = find_free_port();
    std::string run_id = make_run_id();

    // Build argv from config JSON fields.
    std::vector<std::string> args_storage;
    std::vector<const char*> argv;

    args_storage.push_back(venv_python);

    if (config.contains("model_name_or_path"))
        args_storage.push_back(config["model_name_or_path"].get<std::string>());
    else
        throw std::runtime_error("config missing model_name_or_path");

    if (config.contains("dataset_path"))
        args_storage.push_back(config["dataset_path"].get<std::string>());
    else
        throw std::runtime_error("config missing dataset_path");

    if (config.contains("finetuning_method")) {
        args_storage.push_back("--method");
        args_storage.push_back(config["finetuning_method"].get<std::string>());
    }
    if (config.contains("output_dir")) {
        args_storage.push_back("--output");
        args_storage.push_back(config["output_dir"].get<std::string>());
    }

    auto add_flag = [&](const std::string& key, const std::string& flag) {
        if (config.contains(key)) {
            args_storage.push_back(flag);
            args_storage.push_back(std::to_string(config[key].get<double>()));
        }
    };
    auto add_bool_flag = [&](const std::string& key, const std::string& flag) {
        if (config.contains(key) && config[key].get<bool>()) {
            args_storage.push_back(flag);
        }
    };

    add_flag("num_epochs", "--epochs");
    add_flag("batch_size", "--batch-size");
    add_flag("learning_rate", "--learning-rate");
    add_flag("max_seq_length", "--max-seq-length");
    add_bool_flag("pack_sequences", "--pack-sequences");
    add_bool_flag("stream", "--stream");

    args_storage.push_back("--monitor");
    args_storage.push_back("--monitor-port");
    args_storage.push_back(std::to_string(monitor_port));

    for (const auto& a : args_storage) argv.push_back(a.c_str());
    argv.push_back(nullptr);

    // Two pipes for stdout and stderr.
    int out_pipe[2];
    int err_pipe[2];
    if (::pipe2(out_pipe, O_CLOEXEC) != 0) {
        throw std::runtime_error("pipe2(stdout) failed: " + std::string(strerror(errno)));
    }
    if (::pipe2(err_pipe, O_CLOEXEC) != 0) {
        ::close(out_pipe[0]); ::close(out_pipe[1]);
        throw std::runtime_error("pipe2(stderr) failed: " + std::string(strerror(errno)));
    }

    pid_t pid = ::fork();
    if (pid < 0) {
        ::close(out_pipe[0]); ::close(out_pipe[1]);
        ::close(err_pipe[0]); ::close(err_pipe[1]);
        throw std::runtime_error("fork failed: " + std::string(strerror(errno)));
    }

    if (pid == 0) {
        // Child.
        ::dup2(out_pipe[1], STDOUT_FILENO);
        ::dup2(err_pipe[1], STDERR_FILENO);
        ::close(out_pipe[0]); ::close(out_pipe[1]);
        ::close(err_pipe[0]); ::close(err_pipe[1]);
        ::execv(argv[0], const_cast<char* const*>(argv.data()));
        ::_exit(127);  // exec failed
    }

    // Parent: close write ends.
    ::close(out_pipe[1]);
    ::close(err_pipe[1]);

    PHOS_LOG_INFO("TrainingManager: spawned run_id={} pid={} monitor_port={}",
                  run_id, pid, monitor_port);

    ActiveRun run;
    run.run_id = run_id;
    run.pid = pid;
    run.monitor_port = monitor_port;
    run.stdout_fd = out_pipe[0];
    run.stderr_fd = err_pipe[0];

    run.stdout_thread = std::thread([this, fd = out_pipe[0], rid = run_id]() {
        read_pipe(fd, rid, "stdout");
    });
    run.stderr_thread = std::thread([this, fd = err_pipe[0], rid = run_id]() {
        read_pipe(fd, rid, "stderr");
    });

    {
        std::lock_guard<std::mutex> lock(mutex_);
        runs_.emplace(run_id, std::move(run));
    }

    return RunHandle{run_id, pid, monitor_port};
}

bool TrainingManager::stop(const std::string& run_id) {
    pid_t pid = -1;
    {
        std::lock_guard<std::mutex> lock(mutex_);
        auto it = runs_.find(run_id);
        if (it == runs_.end()) return false;
        pid = it->second.pid;
    }

    if (pid > 0) {
        ::kill(pid, SIGTERM);
    }

    // Detached cleanup thread: keep stop() responsive for the UI.
    std::thread([this, run_id, pid]() {
        if (pid > 0) {
            auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(10);
            bool reaped = false;
            while (std::chrono::steady_clock::now() < deadline) {
                int status;
                pid_t r = ::waitpid(pid, &status, WNOHANG);
                if (r == pid) { reaped = true; break; }
                if (r < 0 && errno == ECHILD) { reaped = true; break; }
                std::this_thread::sleep_for(std::chrono::milliseconds(200));
            }
            if (!reaped) {
                ::kill(pid, SIGKILL);
                ::waitpid(pid, nullptr, 0);
            }
            // NOTE: the reaper thread (on_child_exit) may already have emitted
            // run_complete by the time we get here; that's fine -- it marks
            // run.pid = -1 but leaves the entry for us to finish cleanup.
        }

        // Remove the entry from the map and clean up its pipes / threads.
        // We must NOT hold mutex_ while joining reader threads.
        ActiveRun run;
        bool found = false;
        {
            std::lock_guard<std::mutex> lock(mutex_);
            auto it = runs_.find(run_id);
            if (it != runs_.end()) {
                run = std::move(it->second);
                runs_.erase(it);
                found = true;
            }
        }
        if (found) {
            cleanup_run(run);
        }
    }).detach();

    return true;
}

nlohmann::json TrainingManager::list() const {
    std::lock_guard<std::mutex> lock(mutex_);
    nlohmann::json result = nlohmann::json::array();
    for (const auto& [id, run] : runs_) {
        result.push_back({
            {"run_id", run.run_id},
            {"pid", run.pid},
            {"monitor_port", run.monitor_port}
        });
    }
    return result;
}

// ---------------------------------------------------------------------------
// Pipe reader
// ---------------------------------------------------------------------------

void TrainingManager::read_pipe(int fd, const std::string& run_id, const char* stream) {
    std::string buf;
    char chunk[4096];
    while (true) {
        ssize_t n = ::read(fd, chunk, sizeof(chunk) - 1);
        if (n <= 0) break;
        chunk[n] = '\0';
        buf += chunk;

        size_t pos;
        while ((pos = buf.find('\n')) != std::string::npos) {
            std::string line = buf.substr(0, pos);
            buf.erase(0, pos + 1);
            if (!line.empty()) {
                win_.emit("log_line", {
                    {"run_id", run_id},
                    {"stream", stream},
                    {"line", line}
                });
            }
        }
    }
    if (!buf.empty()) {
        win_.emit("log_line", {
            {"run_id", run_id},
            {"stream", stream},
            {"line", buf}
        });
    }
    // fd is closed by caller (cleanup_run) to avoid double-close.
}

}  // namespace mp_studio

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
#include <deque>
#include <sstream>
#include <stdexcept>

namespace mp_studio {

// Marker prefix printed by the Python sidecar as its first stdout line
// once the monitor's aiohttp server has bound a port. We parse it out of
// the stdout stream and route the port to the WsBridge via the
// registered callback. See mud_puppy/trainer.py.
static constexpr const char* MONITOR_PORT_MARKER = "MUD_PUPPY_MONITOR_PORT=";

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

    // B3 fix: reset SIGCHLD disposition BEFORE closing the self-pipe write end.
    // A SIGCHLD arriving between shutting_down_.store and pipe close would write
    // to a file descriptor we just closed (potentially recycled). Reset to SIG_DFL
    // first so any late SIGCHLD is handled by the kernel default (no-op for us).
    {
        struct sigaction sa{};
        sa.sa_handler = SIG_DFL;
        sigemptyset(&sa.sa_mask);
        ::sigaction(SIGCHLD, &sa, nullptr);
    }

    // Now close the self-pipe to wake and exit the reaper thread.
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

    instance_ = nullptr;
}

// Note: SIGCHLD handler was already reset to SIG_DFL in the destructor before
// closing the self-pipe (B3 fix). No secondary reset needed here.

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
                run.exit_code = exit_code;
                run.status = (exit_code == 0) ? RunStatus::Complete : RunStatus::Failed;
                auto now_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                    std::chrono::system_clock::now().time_since_epoch()).count();
                run.end_time_ms = static_cast<int64_t>(now_ms);

                // Stash a RunSummary JSON in completed_runs_ for list().
                const char* status_str = (exit_code == 0) ? "complete" : "failed";
                nlohmann::json summary = {
                    {"run_id",     run.run_id},
                    {"model",      run.model},
                    {"method",     run.method},
                    {"dataset",    run.dataset},
                    {"status",     status_str},
                    {"start_time", run.start_time_ms},
                    {"end_time",   run.end_time_ms},
                };
                completed_runs_.insert(completed_runs_.begin(), std::move(summary));
                if (completed_runs_.size() > MAX_COMPLETED_RUNS) {
                    completed_runs_.resize(MAX_COMPLETED_RUNS);
                }
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

void TrainingManager::set_port_ready_callback(PortReadyCb cb) {
    std::lock_guard<std::mutex> lock(mutex_);
    port_ready_cb_ = std::move(cb);
}

void TrainingManager::on_port_announced(const std::string& run_id, int port) {
    PortReadyCb cb;
    {
        std::lock_guard<std::mutex> lock(mutex_);
        auto it = runs_.find(run_id);
        if (it != runs_.end()) {
            it->second.monitor_port = port;
        }
        cb = port_ready_cb_;
    }
    PHOS_LOG_INFO("TrainingManager: run_id={} bound monitor port {}",
                  run_id, port);
    if (cb) cb(run_id, port);
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
    // B4 fix: join reader threads BEFORE closing the read-end fds.
    // The reader calls read(fd, ...) in a blocking loop; read() returns 0
    // (EOF) when the write end of the pipe is closed (which happens when
    // the child exits). Once the thread has seen EOF and returned, we can
    // safely close our (read-end) copy of the fd. Closing before join()
    // risks a use-after-free if the fd number is recycled and the still-
    // running reader reads from a different file entirely.
    if (run.stdout_thread.joinable()) run.stdout_thread.join();
    if (run.stderr_thread.joinable()) run.stderr_thread.join();
    if (run.stdout_fd >= 0) { ::close(run.stdout_fd); run.stdout_fd = -1; }
    if (run.stderr_fd >= 0) { ::close(run.stderr_fd); run.stderr_fd = -1; }
}

// ---------------------------------------------------------------------------
// start / stop / list
// ---------------------------------------------------------------------------

RunHandle TrainingManager::start(const nlohmann::json& config) {
    std::string venv_python = resolve_sidecar_path();

    // Pass --monitor-port 0 so the child OS-binds an ephemeral port;
    // it announces the bound port via a stdout marker line which we
    // parse in read_pipe(). This eliminates the bind(0)/close/rebind
    // TOCTOU race the previous find_free_port() path had.
    int monitor_port = 0;
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
    auto add_str_flag = [&](const std::string& key, const std::string& flag) {
        if (config.contains(key) && config[key].is_string()) {
            const auto val = config[key].get<std::string>();
            if (!val.empty()) {
                args_storage.push_back(flag);
                args_storage.push_back(val);
            }
        }
    };
    auto add_bool_flag = [&](const std::string& key, const std::string& flag) {
        if (config.contains(key) && config[key].get<bool>()) {
            args_storage.push_back(flag);
        }
    };

    // A6: wire the flags that were previously missing.
    add_flag("num_epochs", "--epochs");
    add_flag("batch_size", "--batch-size");
    add_flag("gradient_accumulation", "--gradient-accumulation");
    add_flag("learning_rate", "--learning-rate");
    add_flag("max_seq_length", "--max-seq-length");
    add_bool_flag("pack_sequences", "--pack-sequences");
    add_bool_flag("stream", "--stream");
    add_str_flag("precision", "--precision");
    add_bool_flag("compile", "--compile");
    add_str_flag("compile_mode", "--compile-mode");
    add_flag("lora_r", "--lora-r");
    add_flag("lora_alpha", "--lora-alpha");
    add_flag("lora_dropout", "--lora-dropout");
    add_str_flag("quant_backend", "--quant-backend");
    add_bool_flag("trust_remote_code", "--trust-remote-code");
    add_bool_flag("zero_offload", "--zero-offload");
    add_bool_flag("resume", "--resume");
    add_flag("prefetch_layers", "--prefetch-layers");
    add_bool_flag("merge_lora", "--merge-lora");
    add_str_flag("merge_precision", "--merge-precision");
    // preference sub-type (DPO/IPO/KTO/ORPO) -- only meaningful when method == "preference"
    add_str_flag("preference", "--preference");

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
        // Child: reset signal dispositions modified by the parent before execv.
        // The SIGCHLD handler writes to the parent's sigchld_pipe_; if the child
        // inherits it and receives SIGCHLD between fork and execv it would
        // corrupt the parent's self-pipe. SIG_DFL is safe because the child
        // process will install its own handlers after execv.
        struct sigaction sa_dfl{};
        sa_dfl.sa_handler = SIG_DFL;
        sigemptyset(&sa_dfl.sa_mask);
        ::sigaction(SIGCHLD, &sa_dfl, nullptr);
        ::signal(SIGPIPE, SIG_DFL);

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

    // Capture config snapshot for RunSummary (list() / Runs pane).
    std::string snap_model;
    std::string snap_method;
    std::string snap_dataset;
    if (config.contains("model_name_or_path") && config["model_name_or_path"].is_string())
        snap_model = config["model_name_or_path"].get<std::string>();
    if (config.contains("finetuning_method") && config["finetuning_method"].is_string())
        snap_method = config["finetuning_method"].get<std::string>();
    if (config.contains("dataset_path") && config["dataset_path"].is_string())
        snap_dataset = config["dataset_path"].get<std::string>();

    auto now_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::system_clock::now().time_since_epoch()).count();

    // B1 fix: emplace the ActiveRun under the lock BEFORE spawning reader
    // threads. If the child prints MUD_PUPPY_MONITOR_PORT immediately after
    // fork, on_port_announced must be able to find the entry in runs_.
    // Previously the emplace happened AFTER thread spawn, creating a window
    // where the port announcement would silently be dropped.
    {
        std::lock_guard<std::mutex> lock(mutex_);
        ActiveRun run;
        run.run_id = run_id;
        run.pid = pid;
        run.monitor_port = monitor_port;
        run.stdout_fd = out_pipe[0];
        run.stderr_fd = err_pipe[0];
        run.model = snap_model;
        run.method = snap_method;
        run.dataset = snap_dataset;
        run.start_time_ms = static_cast<int64_t>(now_ms);
        run.status = RunStatus::Running;
        runs_.emplace(run_id, std::move(run));
    }

    // Spawn reader threads after emplace so they can safely call
    // on_port_announced and find the entry.
    {
        std::lock_guard<std::mutex> lock(mutex_);
        auto it = runs_.find(run_id);
        if (it != runs_.end()) {
            it->second.stdout_thread = std::thread(
                [this, fd = out_pipe[0], rid = run_id]() {
                    read_pipe(fd, rid, "stdout");
                });
            it->second.stderr_thread = std::thread(
                [this, fd = err_pipe[0], rid = run_id]() {
                    read_pipe(fd, rid, "stderr");
                });
        }
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

    // B5 fix: stop() must NOT call waitpid. The reaper thread is the sole
    // authority for all waitpid calls. Double-waitpid between the detached
    // cleanup thread and the reaper is a race that causes ECHILD errors and
    // can reap an unrelated process if the pid was recycled.
    //
    // Strategy: send SIGTERM, then after 10 seconds send SIGKILL if the
    // process still exists (kill(pid, 0) probes liveness without waitpid).
    // The reaper thread handles the actual waitpid when the child exits.
    // After we are confident the child is dead (or the reaper has cleaned up),
    // we remove the runs_ entry and join reader threads.
    std::thread([this, run_id, pid]() {
        if (pid > 0) {
            auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(10);
            while (std::chrono::steady_clock::now() < deadline) {
                // Check if child still exists (no waitpid -- just a signal 0 probe).
                if (::kill(pid, 0) != 0 && errno == ESRCH) {
                    break;  // process is gone (reaped by reaper or never existed)
                }
                // Also check if the reaper has already marked pid=-1.
                {
                    std::lock_guard<std::mutex> lock(mutex_);
                    auto it = runs_.find(run_id);
                    if (it != runs_.end() && it->second.pid < 0) break;
                    if (it == runs_.end()) return;  // already fully cleaned up
                }
                std::this_thread::sleep_for(std::chrono::milliseconds(200));
            }
            // Deadline expired: send SIGKILL. Let the reaper catch the exit.
            if (::kill(pid, 0) == 0) {
                ::kill(pid, SIGKILL);
                // Brief wait for reaper to catch up (not waitpid -- reaper owns that).
                std::this_thread::sleep_for(std::chrono::milliseconds(500));
            }
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

    // Active runs first.
    for (const auto& [id, run] : runs_) {
        nlohmann::json entry = {
            {"run_id",     run.run_id},
            {"model",      run.model},
            {"method",     run.method},
            {"dataset",    run.dataset},
            {"status",     "running"},
            {"start_time", run.start_time_ms},
        };
        result.push_back(std::move(entry));
    }

    // Then completed runs (newest first, up to MAX_COMPLETED_RUNS).
    for (const auto& summary : completed_runs_) {
        result.push_back(summary);
    }

    return result;
}

// ---------------------------------------------------------------------------
// Pipe reader
// ---------------------------------------------------------------------------

void TrainingManager::read_pipe(int fd, const std::string& run_id, const char* stream) {
    // Only scan for the monitor-port marker on the stdout stream; we
    // don't expect Python to print it on stderr, and scanning stderr
    // would be extra work on every error line.
    const bool scan_port = (std::strcmp(stream, "stdout") == 0);

    std::string buf;
    char chunk[4096];
    auto dispatch_line = [&](const std::string& line) {
        // Check for the bound-port announcement before emitting. We
        // still emit the marker line as a regular log_line so the user
        // sees it in the Logs pane — it's a perfectly valid progress
        // message.
        if (scan_port && line.rfind(MONITOR_PORT_MARKER, 0) == 0) {
            try {
                int port = std::stoi(
                    line.substr(std::strlen(MONITOR_PORT_MARKER)));
                if (port > 0 && port < 65536) {
                    on_port_announced(run_id, port);
                }
            } catch (const std::exception& exc) {
                PHOS_LOG_WARN("TrainingManager: bad port marker '{}' ({})",
                              line, exc.what());
            }
        }

        if (!line.empty()) {
            win_.emit("log_line", {
                {"run_id", run_id},
                {"stream", stream},
                {"line", line}
            });
        }
    };

    while (true) {
        ssize_t n = ::read(fd, chunk, sizeof(chunk) - 1);
        if (n <= 0) break;
        chunk[n] = '\0';
        buf += chunk;

        size_t pos;
        while ((pos = buf.find('\n')) != std::string::npos) {
            std::string line = buf.substr(0, pos);
            buf.erase(0, pos + 1);
            dispatch_line(line);
        }
    }
    if (!buf.empty()) {
        dispatch_line(buf);
    }
    // fd is closed by caller (cleanup_run) to avoid double-close.
}

}  // namespace mp_studio

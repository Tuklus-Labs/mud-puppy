#include "training_manager.hpp"

#include <phos/phos.h>

#include <arpa/inet.h>
#include <fcntl.h>
#include <netinet/in.h>
#include <signal.h>
#include <sys/socket.h>
#include <sys/wait.h>
#include <unistd.h>

#include <chrono>
#include <cstring>
#include <sstream>
#include <stdexcept>

namespace mp_studio {

// Static instance pointer for SIGCHLD handler.
TrainingManager* TrainingManager::instance_ = nullptr;

TrainingManager::TrainingManager(phos::Window& win) : win_(win) {
    instance_ = this;
    install_sigchld(this);
}

TrainingManager::~TrainingManager() {
    // Terminate all running sidecars on destruction.
    std::unique_lock<std::mutex> lock(mutex_);
    for (auto& [id, run] : runs_) {
        if (run.pid > 0) {
            ::kill(run.pid, SIGTERM);
        }
    }
    // Give sidecars up to 10 seconds to exit, then SIGKILL.
    auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(10);
    for (auto& [id, run] : runs_) {
        if (run.pid <= 0) continue;
        while (std::chrono::steady_clock::now() < deadline) {
            int status;
            pid_t r = ::waitpid(run.pid, &status, WNOHANG);
            if (r != 0) break;
            std::this_thread::sleep_for(std::chrono::milliseconds(200));
        }
        // Force-kill if still alive.
        ::kill(run.pid, SIGKILL);
        ::waitpid(run.pid, nullptr, 0);
    }
    // Join log threads.
    for (auto& [id, run] : runs_) {
        if (run.log_thread.joinable()) {
            close(run.stderr_fd);
            run.log_thread.join();
        }
    }
    instance_ = nullptr;
}

int TrainingManager::find_free_port() {
    int fd = ::socket(AF_INET, SOCK_STREAM, 0);
    if (fd < 0) return 5980;  // fallback

    sockaddr_in addr{};
    addr.sin_family = AF_INET;
    addr.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
    addr.sin_port = 0;

    if (::bind(fd, reinterpret_cast<sockaddr*>(&addr), sizeof(addr)) != 0) {
        ::close(fd);
        return 5980;
    }

    socklen_t len = sizeof(addr);
    ::getsockname(fd, reinterpret_cast<sockaddr*>(&addr), &len);
    int port = ntohs(addr.sin_port);
    ::close(fd);
    return port;
}

std::string TrainingManager::make_run_id() {
    // A short timestamp-based ID.
    auto now = std::chrono::system_clock::now().time_since_epoch();
    auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(now).count();
    std::ostringstream ss;
    ss << "run-" << ms;
    return ss.str();
}

RunHandle TrainingManager::start(const nlohmann::json& config) {
    // Determine project root: binary dir -> up two levels (build/ -> ui/ -> repo/).
    // We resolve the path to venv/bin/mud-puppy relative to the manifest's directory.
    // For now we assume the venv is at $HOME/Projects/mud-puppy/venv.
    const char* home = ::getenv("HOME");
    if (!home) home = "/home/aegis";

    std::string venv_python = std::string(home) + "/Projects/mud-puppy/venv/bin/mud-puppy";

    // Check executable exists.
    if (::access(venv_python.c_str(), X_OK) != 0) {
        throw std::runtime_error("mud-puppy sidecar not found at: " + venv_python);
    }

    int monitor_port = find_free_port();
    std::string run_id = make_run_id();

    // Build argv from config JSON fields.
    std::vector<std::string> args_storage;
    std::vector<const char*> argv;

    args_storage.push_back(venv_python);

    // Required positional args: model and dataset.
    if (config.contains("model_name_or_path"))
        args_storage.push_back(config["model_name_or_path"].get<std::string>());
    else
        throw std::runtime_error("config missing model_name_or_path");

    if (config.contains("dataset_path"))
        args_storage.push_back(config["dataset_path"].get<std::string>());
    else
        throw std::runtime_error("config missing dataset_path");

    // Method.
    if (config.contains("finetuning_method")) {
        args_storage.push_back("--method");
        args_storage.push_back(config["finetuning_method"].get<std::string>());
    }

    // Output dir.
    if (config.contains("output_dir")) {
        args_storage.push_back("--output");
        args_storage.push_back(config["output_dir"].get<std::string>());
    }

    // Optional training knobs.
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

    // Monitor flags.
    args_storage.push_back("--monitor");
    args_storage.push_back("--monitor-port");
    args_storage.push_back(std::to_string(monitor_port));

    // Build argv.
    for (const auto& a : args_storage) {
        argv.push_back(a.c_str());
    }
    argv.push_back(nullptr);

    // Create stderr pipe (stdout will be inherited).
    int pipefd[2];
    if (::pipe2(pipefd, O_CLOEXEC) != 0) {
        throw std::runtime_error("pipe2 failed: " + std::string(strerror(errno)));
    }

    pid_t pid = ::fork();
    if (pid < 0) {
        ::close(pipefd[0]);
        ::close(pipefd[1]);
        throw std::runtime_error("fork failed: " + std::string(strerror(errno)));
    }

    if (pid == 0) {
        // Child process.
        // Redirect stderr to write end of pipe.
        ::dup2(pipefd[1], STDERR_FILENO);
        ::close(pipefd[0]);
        ::close(pipefd[1]);

        // Exec the sidecar.
        ::execv(argv[0], const_cast<char* const*>(argv.data()));
        // If we get here, exec failed.
        ::_exit(127);
    }

    // Parent.
    ::close(pipefd[1]);  // close write end

    PHOS_LOG_INFO("TrainingManager: spawned run_id={} pid={} monitor_port={}",
                  run_id, pid, monitor_port);

    // Store the run.
    ActiveRun run;
    run.run_id = run_id;
    run.pid = pid;
    run.monitor_port = monitor_port;
    run.stderr_fd = pipefd[0];

    // Start log reader thread.
    run.log_thread = std::thread([this, fd = pipefd[0], rid = run_id]() {
        read_stderr(fd, rid);
    });

    {
        std::unique_lock<std::mutex> lock(mutex_);
        runs_.emplace(run_id, std::move(run));
    }

    return RunHandle{run_id, pid, monitor_port};
}

bool TrainingManager::stop(const std::string& run_id) {
    std::unique_lock<std::mutex> lock(mutex_);
    auto it = runs_.find(run_id);
    if (it == runs_.end()) return false;

    pid_t pid = it->second.pid;
    if (pid > 0) {
        ::kill(pid, SIGTERM);

        // Background thread waits and force-kills if needed.
        std::thread([pid]() {
            auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(10);
            while (std::chrono::steady_clock::now() < deadline) {
                int status;
                if (::waitpid(pid, &status, WNOHANG) != 0) return;
                std::this_thread::sleep_for(std::chrono::milliseconds(200));
            }
            ::kill(pid, SIGKILL);
            ::waitpid(pid, nullptr, 0);
        }).detach();
    }
    return true;
}

nlohmann::json TrainingManager::list() const {
    std::unique_lock<std::mutex> lock(mutex_);
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

void TrainingManager::read_stderr(int fd, const std::string& run_id) {
    // Read lines from the child's stderr and emit log_line events.
    std::string buf;
    char chunk[4096];
    while (true) {
        ssize_t n = ::read(fd, chunk, sizeof(chunk) - 1);
        if (n <= 0) break;  // EOF or error
        chunk[n] = '\0';
        buf += chunk;

        // Emit complete lines.
        size_t pos;
        while ((pos = buf.find('\n')) != std::string::npos) {
            std::string line = buf.substr(0, pos);
            buf.erase(0, pos + 1);
            if (!line.empty()) {
                win_.emit("log_line", {{"run_id", run_id}, {"line", line}});
            }
        }
    }
    // Emit any remaining partial line.
    if (!buf.empty()) {
        win_.emit("log_line", {{"run_id", run_id}, {"line", buf}});
    }
    ::close(fd);
}

void TrainingManager::install_sigchld(TrainingManager* self) {
    (void)self;
    struct sigaction sa{};
    sa.sa_handler = TrainingManager::sigchld_handler;
    sigemptyset(&sa.sa_mask);
    sa.sa_flags = SA_RESTART | SA_NOCLDSTOP;
    ::sigaction(SIGCHLD, &sa, nullptr);
}

void TrainingManager::sigchld_handler(int) {
    // Reap all available children.
    int status;
    pid_t pid;
    while ((pid = ::waitpid(-1, &status, WNOHANG)) > 0) {
        if (instance_) {
            instance_->on_child_exit(pid, status);
        }
    }
}

void TrainingManager::on_child_exit(pid_t pid, int status) {
    std::unique_lock<std::mutex> lock(mutex_);
    for (auto& [id, run] : runs_) {
        if (run.pid == pid) {
            run.pid = -1;
            int exit_code = WIFEXITED(status) ? WEXITSTATUS(status)
                          : (WIFSIGNALED(status) ? -WTERMSIG(status) : -1);
            win_.emit("run_complete", {
                {"run_id", run.run_id},
                {"exit_code", exit_code}
            });
            PHOS_LOG_INFO("TrainingManager: run_id={} exited code={}", id, exit_code);
            break;
        }
    }
}

}  // namespace mp_studio

#include <phos/phos.h>

#include "training_manager.hpp"
#include "ws_bridge.hpp"
#include "hf_client.hpp"
#include "dataset_peek.hpp"
#include "checkpoint_scan.hpp"

#include <curl/curl.h>

#include <cerrno>
#include <csignal>
#include <filesystem>
#include <unistd.h>
#include <fcntl.h>

// Self-pipe for SIGTERM/SIGINT wakeup.
// The signal handler only calls write() (async-signal-safe).
// A GLib FD source reads from the pipe on the main thread and calls
// g_application_quit() safely.
static int g_shutdown_pipe[2] = {-1, -1};

static void handle_sigterm(int) {
    // write() is listed as async-signal-safe by POSIX.
    // Ignore errors -- if the pipe is full or closed the process is
    // already shutting down.
    char c = 1;
    while (true) {
        ssize_t n = ::write(g_shutdown_pipe[1], &c, 1);
        if (n >= 0) break;
        if (errno == EINTR) continue;
        break;  // EAGAIN / EPIPE: don't block the signal handler.
    }
}

// GLib IO callback: runs on the GMainLoop thread (async-signal-safe is
// not required here). Reads the wakeup byte and quits the application.
static gboolean on_shutdown_pipe_ready(GIOChannel* /*chan*/,
                                       GIOCondition /*cond*/,
                                       gpointer /*data*/) {
    char buf[16];
    ssize_t n;
    do { n = ::read(g_shutdown_pipe[0], buf, sizeof(buf)); }
    while (n == -1 && errno == EINTR);

    GApplication* gapp = g_application_get_default();
    if (gapp) g_application_quit(gapp);
    return G_SOURCE_CONTINUE;  // keep the source alive until the loop exits
}

int main(int /*argc*/, char** argv) {
    // Phos resolves asset_dir and manifest paths relative to the CWD.
    // To make the binary runnable from any directory, chdir to the
    // directory containing argv[0] (where manifest.toml and web/dist/
    // are staged by CMake).
    try {
        std::filesystem::path exe_path = std::filesystem::canonical(argv[0]);
        std::filesystem::current_path(exe_path.parent_path());
    } catch (const std::exception& exc) {
        // Non-fatal: fall back to whatever CWD the user supplied.
        PHOS_LOG_WARN("Could not chdir to binary directory: {}", exc.what());
    }

    // Ignore SIGPIPE (broken pipe from sidecar stderr).
    signal(SIGPIPE, SIG_IGN);

    // Create the self-pipe used by the async-signal-safe SIGTERM/SIGINT handler.
    if (::pipe2(g_shutdown_pipe, O_CLOEXEC | O_NONBLOCK) != 0) {
        PHOS_LOG_ERROR("pipe2(shutdown_pipe) failed: {}", strerror(errno));
        return 1;
    }

    // Install SIGTERM/SIGINT handlers that only write one byte to the pipe.
    // The actual g_application_quit() runs on the GMainLoop thread via the
    // GLib IO watch registered below.
    struct sigaction sa_term{};
    sa_term.sa_handler = handle_sigterm;
    sigemptyset(&sa_term.sa_mask);
    sa_term.sa_flags = SA_RESTART;
    ::sigaction(SIGTERM, &sa_term, nullptr);
    ::sigaction(SIGINT,  &sa_term, nullptr);

    // libcurl requires one global init before any easy handle is created.
    if (curl_global_init(CURL_GLOBAL_ALL) != CURLE_OK) {
        PHOS_LOG_ERROR("curl_global_init failed");
        return 1;
    }

    auto manifest = phos::resolve_manifest();
    PHOS_LOG_INFO("mud-puppy-studio starting, manifest={}", manifest);

    phos::App app(manifest);
    auto& win = app.window(app.config().window);

    // Enable debug logging when devtools are on.
    if (app.config().window.devtools) {
        phos::log::set_level(phos::log::Level::Debug);
    }

    // Instantiate subsystems. WsBridge is constructed first so its ref
    // can be bound into the TrainingManager port-announcement callback.
    mp_studio::WsBridge ws(win);
    mp_studio::TrainingManager tm(win);
    mp_studio::HfClient hf;

    // When a child sidecar announces its bound monitor port (via the
    // stdout marker), connect the WsBridge. This replaces the older
    // pre-allocate-port-and-pass-it dance which had a TOCTOU race.
    tm.set_port_ready_callback(
        [&ws](const std::string& run_id, int port) {
            ws.connect(run_id, port);
        });

    // --- IPC handlers ---
    // Only handlers declared in manifest.toml are registered here.

    // run.start: spawn a training sidecar. The monitor port is not
    // known at return time; WsBridge is connected asynchronously via
    // the port_ready_cb once the child announces its bound port.
    win.handle("run.start", phos::safe_handler([&](const phos::Json& req) -> phos::Json {
        // Accept either { config: {...} } or the config object directly.
        const auto& config = req.contains("config") ? req["config"] : req;
        auto handle = tm.start(config);
        return {
            {"run_id", handle.run_id},
            {"pid", static_cast<int>(handle.pid)}
        };
    }));

    // run.stop: send SIGTERM to sidecar and disconnect WsBridge.
    win.handle("run.stop", phos::safe_handler([&](const phos::Json& req) -> phos::Json {
        std::string run_id = req.at("run_id").get<std::string>();
        ws.disconnect(run_id);
        bool ok = tm.stop(run_id);
        return phos::make_ok({{"stopped", ok}});
    }));

    // run.list: return all active runs as a JSON array.
    win.handle("run.list", phos::safe_handler([&](const phos::Json&) -> phos::Json {
        return tm.list();
    }));

    // hf.search_models: search HF Hub by query string.
    win.handle("hf.search_models", phos::safe_handler([&](const phos::Json& req) -> phos::Json {
        std::string query = req.at("query").get<std::string>();
        return hf.search_models(query);
    }));

    // checkpoint.list: enumerate checkpoints and LoRA adapters in output_dir.
    win.handle("checkpoint.list", phos::safe_handler([](const phos::Json& req) -> phos::Json {
        std::string output_dir = req.at("output_dir").get<std::string>();
        return mp_studio::scan_checkpoints(output_dir);
    }));

    // dataset.preview: read first N lines of a JSONL file and detect format.
    win.handle("dataset.preview", phos::safe_handler([](const phos::Json& req) -> phos::Json {
        std::string path = req.at("path").get<std::string>();
        int n = req.value("n", 5);
        return mp_studio::dataset_peek(path, n);
    }));

    // Register a GLib IO watch so the main thread reacts to the shutdown pipe.
    // When SIGTERM/SIGINT fires, handle_sigterm() writes one byte; GLib wakes
    // the main loop and on_shutdown_pipe_ready() calls g_application_quit().
    GIOChannel* shutdown_chan = g_io_channel_unix_new(g_shutdown_pipe[0]);
    g_io_channel_set_close_on_unref(shutdown_chan, FALSE);  // we close manually
    g_io_add_watch(shutdown_chan, G_IO_IN, on_shutdown_pipe_ready, nullptr);
    g_io_channel_unref(shutdown_chan);  // GLib retains the ref via the watch

    int rc = app.run();

    // libcurl cleanup mirrors the init at startup.
    curl_global_cleanup();

    // Close the shutdown self-pipe now that the main loop has exited.
    if (g_shutdown_pipe[0] >= 0) { ::close(g_shutdown_pipe[0]); g_shutdown_pipe[0] = -1; }
    if (g_shutdown_pipe[1] >= 0) { ::close(g_shutdown_pipe[1]); g_shutdown_pipe[1] = -1; }

    return rc;
}

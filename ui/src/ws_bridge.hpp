#pragma once

#include <nlohmann/json.hpp>
#include <phos/phos.h>

#include <glib-object.h>
#include <libsoup/soup.h>

#include <atomic>
#include <cstdint>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <unordered_map>

namespace mp_studio {

// WsBridge connects to the training sidecar's WebSocket monitor endpoint
// and forwards JSON frames to the webview as IPC events.
//
// Each active run gets its own libsoup WebSocket client, running on a
// dedicated GMainContext thread. All mutations of the per-run connection
// state (session, ws_conn, timers) happen on that GMainContext thread, via
// g_main_context_invoke -- never on the caller thread.
class WsBridge {
public:
    explicit WsBridge(phos::Window& win);
    ~WsBridge();

    // Connect to ws://127.0.0.1:<port>/ws for the given run_id.
    // Spawns a dedicated GMainLoop thread and retries with a 30-attempt cap.
    void connect(const std::string& run_id, int port);

    // Close the WebSocket connection for run_id. Blocks until the loop thread
    // has exited and all state is freed.
    void disconnect(const std::string& run_id);

    // Forward a parsed JSON frame to the webview as the appropriate event.
    // Callable from any thread; win_.emit() is itself thread-safe.
    void forward_frame(const std::string& run_id, const nlohmann::json& frame);

    // Public so the internal WsConnCtx in ws_bridge.cpp can reference it.
    // Lifetime: owned by WsBridge::conns_ (unique_ptr). The GMainContext
    // thread is the sole mutator of ws_conn / session / retry_count.
    struct RunConn {
        std::string run_id;
        int port = 0;
        GMainContext* ctx = nullptr;
        GMainLoop* loop = nullptr;
        SoupSession* session = nullptr;
        SoupWebsocketConnection* ws_conn = nullptr;
        std::thread loop_thread;
        std::atomic<bool> stopping{false};
        uint32_t retry_count = 0;  // mutated only on loop thread
        WsBridge* bridge = nullptr;

        RunConn() = default;
        RunConn(const RunConn&) = delete;
        RunConn& operator=(const RunConn&) = delete;
    };

    // Maximum number of connect attempts before giving up (2s per retry = 60s).
    static constexpr uint32_t MAX_RETRIES = 30;

    WsBridge(const WsBridge&) = delete;
    WsBridge& operator=(const WsBridge&) = delete;

private:
    phos::Window& win_;
    mutable std::mutex mutex_;
    std::unordered_map<std::string, std::unique_ptr<RunConn>> conns_;

    void run_loop(RunConn* conn);

    // All three of these run on the GMainLoop thread.
    static void on_ws_connected(GObject* src, GAsyncResult* res, gpointer user_data);
    static void on_ws_message(SoupWebsocketConnection* ws, SoupWebsocketDataType type,
                              GBytes* message, gpointer user_data);
    static void on_ws_closed(SoupWebsocketConnection* ws, gpointer user_data);

    // Schedule a connection attempt on the loop thread.
    static gboolean attempt_connect(gpointer user_data);

    // Emit a run_complete event with an error payload when retries exhaust.
    void emit_retry_exhausted(const std::string& run_id);
};

}  // namespace mp_studio

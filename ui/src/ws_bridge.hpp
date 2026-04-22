#pragma once

#include <nlohmann/json.hpp>
#include <phos/phos.h>

#include <glib-object.h>
#include <libsoup/soup.h>

#include <atomic>
#include <mutex>
#include <string>
#include <thread>
#include <unordered_map>

namespace mp_studio {

// WsBridge connects to the training sidecar's WebSocket monitor endpoint
// and forwards JSON frames to the webview as IPC events.
//
// Each active run gets its own libsoup WebSocket client, running on a
// dedicated GMainContext so the GTK main loop is not blocked.
class WsBridge {
public:
    // Public so that WsConnCtx (defined in ws_bridge.cpp) can reference it.
    struct RunConn {
        std::string run_id;
        int port;
        GMainContext* ctx;
        GMainLoop* loop;
        SoupSession* session;
        SoupWebsocketConnection* ws_conn;
        std::thread loop_thread;
        std::atomic<bool> stopping{false};

        RunConn() = default;
        RunConn(const RunConn&) = delete;
        RunConn& operator=(const RunConn&) = delete;
    };

    explicit WsBridge(phos::Window& win);
    ~WsBridge();

    // Connect to ws://127.0.0.1:<port>/ws for the given run_id.
    // Retries for up to 30 seconds if the sidecar is still starting.
    void connect(const std::string& run_id, int port);

    // Close the WebSocket connection for run_id.
    void disconnect(const std::string& run_id);

    // Forward a parsed JSON frame to the webview as the appropriate event.
    void forward_frame(const std::string& run_id, const nlohmann::json& frame);

    WsBridge(const WsBridge&) = delete;
    WsBridge& operator=(const WsBridge&) = delete;

private:
    phos::Window& win_;
    mutable std::mutex mutex_;
    std::unordered_map<std::string, RunConn*> conns_;

    void run_loop(RunConn* conn);
    static void on_ws_connected(GObject* src, GAsyncResult* res, gpointer user_data);
    static void on_ws_message(SoupWebsocketConnection* ws, SoupWebsocketDataType type,
                              GBytes* message, gpointer user_data);
    static void on_ws_closed(SoupWebsocketConnection* ws, gpointer user_data);
};

}  // namespace mp_studio

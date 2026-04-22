/**
 * Root ErrorBoundary — catches render errors and shows a diagnostic panel.
 *
 * Per Forge lesson: without this, a render throw leaves <div id="root" /> blank.
 * In dev mode shows the stack trace; in prod logs to console only.
 */
import React, { Component, ErrorInfo } from "react";

interface Props {
  children: React.ReactNode;
}

interface State {
  hasError: boolean;
  error: Error | null;
  errorInfo: ErrorInfo | null;
}

export class ErrorBoundary extends Component<Props, State> {
  constructor(props: Props) {
    super(props);
    this.state = { hasError: false, error: null, errorInfo: null };
  }

  static getDerivedStateFromError(error: Error): Partial<State> {
    return { hasError: true, error };
  }

  componentDidCatch(error: Error, info: ErrorInfo) {
    console.error("[mud-puppy-studio] Render error:", error, info);
    this.setState({ errorInfo: info });
  }

  render() {
    if (!this.state.hasError) {
      return this.props.children;
    }

    const isDev = import.meta.env.DEV;

    return (
      <div
        style={{
          position: "fixed",
          inset: 0,
          background: "#05070d",
          display: "flex",
          alignItems: "center",
          justifyContent: "center",
          padding: 32,
          fontFamily: "'Share Tech Mono', monospace",
        }}
      >
        <div
          style={{
            maxWidth: 640,
            width: "100%",
            border: "1px solid #ff2bd6",
            padding: 24,
            background: "#0b1220",
            clipPath: `polygon(
              8px 0, calc(100% - 8px) 0, 100% 8px,
              100% calc(100% - 8px), calc(100% - 8px) 100%,
              8px 100%, 0 calc(100% - 8px), 0 8px
            )`,
          }}
        >
          {/* Header */}
          <div
            style={{
              color: "#ff2bd6",
              fontSize: 14,
              letterSpacing: 3,
              textTransform: "uppercase",
              marginBottom: 16,
            }}
          >
            RENDER ERROR
          </div>

          {/* Error message */}
          <div
            style={{
              color: "#e6ecf5",
              fontSize: 12,
              marginBottom: 12,
              fontFamily: "'JetBrains Mono', monospace",
            }}
          >
            {this.state.error?.message || "Unknown error"}
          </div>

          {/* Stack trace (dev only) */}
          {isDev && this.state.error?.stack && (
            <pre
              style={{
                background: "#05070d",
                padding: 12,
                fontSize: 10,
                color: "#5a6d82",
                overflow: "auto",
                maxHeight: 200,
                fontFamily: "'JetBrains Mono', monospace",
                marginBottom: 16,
                whiteSpace: "pre-wrap",
                wordBreak: "break-all",
              }}
            >
              {this.state.error.stack}
            </pre>
          )}

          {isDev && this.state.errorInfo?.componentStack && (
            <pre
              style={{
                background: "#05070d",
                padding: 12,
                fontSize: 10,
                color: "#5a6d82",
                overflow: "auto",
                maxHeight: 160,
                fontFamily: "'JetBrains Mono', monospace",
                marginBottom: 16,
                whiteSpace: "pre-wrap",
                wordBreak: "break-all",
              }}
            >
              {this.state.errorInfo.componentStack}
            </pre>
          )}

          {/* Reload button */}
          <button
            onClick={() => window.location.reload()}
            style={{
              fontSize: "10px",
              padding: "8px 20px",
              letterSpacing: "2px",
              textTransform: "uppercase",
              background: "transparent",
              border: "1px solid #ff2bd6",
              color: "#ff2bd6",
              cursor: "pointer",
            }}
          >
            RELOAD APP
          </button>

          {/* Reset button */}
          <button
            onClick={() => this.setState({ hasError: false, error: null, errorInfo: null })}
            style={{
              fontSize: "10px",
              padding: "8px 20px",
              letterSpacing: "2px",
              textTransform: "uppercase",
              background: "transparent",
              border: "1px solid #1f3648",
              color: "#5a6d82",
              cursor: "pointer",
              marginLeft: 8,
            }}
          >
            TRY RESET
          </button>
        </div>
      </div>
    );
  }
}

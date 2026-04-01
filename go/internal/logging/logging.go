package logging

import (
	"log/slog"
	"os"
	"strings"
)

// Setup configures the default slog logger to write text-formatted output to
// stderr at the specified severity. The level string is case-insensitive and
// accepts "DEBUG", "INFO", "WARN" (or "WARNING"), and "ERROR"; any unrecognised
// value falls back to INFO.
func Setup(level string) {
	var lvl slog.Level
	switch strings.ToUpper(level) {
	case "DEBUG":
		lvl = slog.LevelDebug
	case "WARN", "WARNING":
		lvl = slog.LevelWarn
	case "ERROR":
		lvl = slog.LevelError
	default:
		lvl = slog.LevelInfo
	}

	handler := slog.NewTextHandler(os.Stderr, &slog.HandlerOptions{
		Level: lvl,
	})
	slog.SetDefault(slog.New(handler))
}

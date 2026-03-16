package siglip

import (
	"archive/tar"
	"compress/gzip"
	"fmt"
	"io"
	"net/http"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"strings"
	"time"
)

const ortVersion = "1.24.3"

// Default HuggingFace repo containing pre-exported ONNX models.
const defaultHFRepo = "SabiQG/siglip-go-onnx"

var modelFiles = []string{
	"vision_model.onnx",
	"vision_model.onnx.data",
	"text_model.onnx",
	"text_model.onnx.data",
	"vocab.json",
}

func ensureModels(dir string, hfRepoID string) error {
	if err := os.MkdirAll(dir, 0o755); err != nil {
		return err
	}

	var needed []string
	for _, name := range modelFiles {
		dest := filepath.Join(dir, name)
		if _, err := os.Stat(dest); err != nil {
			needed = append(needed, name)
		}
	}
	if len(needed) == 0 {
		return nil
	}

	fmt.Printf("siglip: fetching %d model file(s) from %s\n", len(needed), hfRepoID)
	for i, name := range needed {
		dest := filepath.Join(dir, name)
		url := fmt.Sprintf("https://huggingface.co/%s/resolve/main/%s?download=true", hfRepoID, name)
		fmt.Printf("  [%d/%d] ↓ %s ", i+1, len(needed), name)

		start := time.Now()
		n, err := downloadFile(url, dest)
		if err != nil {
			fmt.Println("✗")
			os.Remove(dest)
			return fmt.Errorf("downloading %s: %w", name, err)
		}
		elapsed := time.Since(start).Seconds()
		if elapsed > 0.01 {
			fmt.Printf("✓ %s (%.1f MB/s)\n", fmtBytes(n), float64(n)/1e6/elapsed)
		} else {
			fmt.Printf("✓ %s\n", fmtBytes(n))
		}
	}
	return nil
}

// ensureORT downloads the ONNX Runtime shared library if not found in any
// known location. Returns the path to the library.
func ensureORT(cacheDir string) (string, error) {
	// Check common system paths first
	if p := findRuntimeLib(); p != "" {
		return p, nil
	}

	// Check in cache dir
	libName := ortLibName()
	cached := filepath.Join(cacheDir, libName)
	if _, err := os.Stat(cached); err == nil {
		return cached, nil
	}

	// Download
	url, dirName := ortDownloadURL()
	if url == "" {
		return "", fmt.Errorf("unsupported platform %s/%s for auto-download; "+
			"install ONNX Runtime manually and use WithRuntimeLib()", runtime.GOOS, runtime.GOARCH)
	}

	fmt.Printf("siglip: fetching ONNX Runtime v%s (%s/%s)\n", ortVersion, runtime.GOOS, runtime.GOARCH)

	if err := os.MkdirAll(cacheDir, 0o755); err != nil {
		return "", err
	}

	tgzPath := filepath.Join(cacheDir, dirName+".tgz")
	fmt.Printf("  ↓ %s.tgz ", dirName)
	start := time.Now()
	n, err := downloadFile(url, tgzPath)
	if err != nil {
		fmt.Println("✗")
		os.Remove(tgzPath)
		return "", fmt.Errorf("downloading ONNX Runtime: %w", err)
	}
	elapsed := time.Since(start).Seconds()
	if elapsed > 0.01 {
		fmt.Printf("✓ %s (%.1f MB/s)\n", fmtBytes(n), float64(n)/1e6/elapsed)
	} else {
		fmt.Printf("✓ %s\n", fmtBytes(n))
	}

	fmt.Printf("  ⚙ extracting %s ... ", libName)
	if err := extractLibFromTgz(tgzPath, libName, cached); err != nil {
		fmt.Println("✗")
		os.Remove(cached)
		os.Remove(tgzPath)
		return "", fmt.Errorf("extracting ONNX Runtime: %w", err)
	}
	fmt.Println("✓")

	// macOS quarantine flag prevents dlopen on downloaded libraries.
	if runtime.GOOS == "darwin" {
		exec.Command("xattr", "-d", "com.apple.quarantine", cached).Run() //nolint:errcheck
	}

	os.Remove(tgzPath)
	fmt.Println("  ✓ runtime ready")
	return cached, nil
}

func ortLibName() string {
	switch runtime.GOOS {
	case "darwin":
		return "libonnxruntime.dylib"
	case "linux":
		return "libonnxruntime.so"
	case "windows":
		return "onnxruntime.dll"
	}
	return "libonnxruntime.so"
}

func ortDownloadURL() (url, dirName string) {
	goos := runtime.GOOS
	arch := runtime.GOARCH

	switch {
	case goos == "darwin" && arch == "arm64":
		dirName = fmt.Sprintf("onnxruntime-osx-arm64-%s", ortVersion)
	case goos == "darwin" && arch == "amd64":
		dirName = fmt.Sprintf("onnxruntime-osx-x86_64-%s", ortVersion)
	case goos == "linux" && arch == "amd64":
		dirName = fmt.Sprintf("onnxruntime-linux-x64-%s", ortVersion)
	case goos == "linux" && arch == "arm64":
		dirName = fmt.Sprintf("onnxruntime-linux-aarch64-%s", ortVersion)
	default:
		return "", ""
	}

	url = fmt.Sprintf("https://github.com/microsoft/onnxruntime/releases/download/v%s/%s.tgz", ortVersion, dirName)
	return url, dirName
}

func extractLibFromTgz(tgzPath, libName, dest string) error {
	f, err := os.Open(tgzPath)
	if err != nil {
		return err
	}
	defer f.Close()

	gz, err := gzip.NewReader(f)
	if err != nil {
		return err
	}
	defer gz.Close()

	tr := tar.NewReader(gz)
	for {
		hdr, err := tr.Next()
		if err == io.EOF {
			break
		}
		if err != nil {
			return err
		}

		// Skip symlinks, directories, and dSYM debug bundles.
		if hdr.Typeflag != tar.TypeReg || strings.Contains(hdr.Name, ".dSYM/") {
			continue
		}
		base := filepath.Base(hdr.Name)
		// Match the library file. Handles both naming conventions:
		//   Linux:  libonnxruntime.so  / libonnxruntime.so.1.21.0
		//   macOS:  libonnxruntime.dylib / libonnxruntime.1.21.0.dylib
		ext := filepath.Ext(libName) // .dylib or .so
		stem := strings.TrimSuffix(libName, ext)
		if base == libName ||
			strings.HasPrefix(base, libName+".") ||
			(strings.HasPrefix(base, stem+".") && strings.HasSuffix(base, ext)) {
			out, err := os.OpenFile(dest, os.O_CREATE|os.O_WRONLY|os.O_TRUNC, 0o755)
			if err != nil {
				return err
			}
			if _, err := io.Copy(out, tr); err != nil {
				out.Close()
				return err
			}
			out.Close()
			return nil
		}
	}
	return fmt.Errorf("library %s not found in archive", libName)
}

func fmtBytes(b int64) string {
	switch {
	case b >= 1_000_000_000:
		return fmt.Sprintf("%.2f GB", float64(b)/1e9)
	case b >= 1_000_000:
		return fmt.Sprintf("%.1f MB", float64(b)/1e6)
	case b >= 1_000:
		return fmt.Sprintf("%.1f KB", float64(b)/1e3)
	default:
		return fmt.Sprintf("%d B", b)
	}
}

func downloadFile(url, dest string) (int64, error) {
	resp, err := http.Get(url) //nolint:gosec // URLs are constructed from known domains
	if err != nil {
		return 0, err
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		return 0, fmt.Errorf("HTTP %d for %s", resp.StatusCode, url)
	}

	f, err := os.Create(dest)
	if err != nil {
		return 0, err
	}
	defer f.Close()

	return io.Copy(f, resp.Body)
}

package siglip

import (
	"archive/tar"
	"compress/gzip"
	"fmt"
	"io"
	"net/http"
	"os"
	"path/filepath"
	"runtime"
	"strings"
)

const ortVersion = "1.21.1"

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

	for _, name := range modelFiles {
		dest := filepath.Join(dir, name)
		if _, err := os.Stat(dest); err == nil {
			continue
		}

		url := fmt.Sprintf("https://huggingface.co/%s/resolve/main/%s?download=true", hfRepoID, name)
		fmt.Printf("siglip: downloading %s ...\n", name)

		if err := downloadFile(url, dest); err != nil {
			os.Remove(dest)
			return fmt.Errorf("downloading %s: %w", name, err)
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

	fmt.Printf("siglip: downloading ONNX Runtime %s for %s/%s ...\n", ortVersion, runtime.GOOS, runtime.GOARCH)

	if err := os.MkdirAll(cacheDir, 0o755); err != nil {
		return "", err
	}

	tgzPath := filepath.Join(cacheDir, dirName+".tgz")
	if err := downloadFile(url, tgzPath); err != nil {
		os.Remove(tgzPath)
		return "", fmt.Errorf("downloading ONNX Runtime: %w", err)
	}

	// Extract just the lib file from the tarball
	if err := extractLibFromTgz(tgzPath, libName, cached); err != nil {
		os.Remove(cached)
		os.Remove(tgzPath)
		return "", fmt.Errorf("extracting ONNX Runtime: %w", err)
	}

	os.Remove(tgzPath)
	fmt.Printf("siglip: ONNX Runtime ready at %s\n", cached)
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

		base := filepath.Base(hdr.Name)
		// Match the library file (also match versioned names like libonnxruntime.so.1.21.1)
		if base == libName || strings.HasPrefix(base, libName+".") {
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

func downloadFile(url, dest string) error {
	resp, err := http.Get(url)
	if err != nil {
		return err
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		return fmt.Errorf("HTTP %d for %s", resp.StatusCode, url)
	}

	f, err := os.Create(dest)
	if err != nil {
		return err
	}
	defer f.Close()

	_, err = io.Copy(f, resp.Body)
	return err
}

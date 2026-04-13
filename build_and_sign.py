#!/usr/bin/env python3
"""
Build FLIMKit with PyInstaller and sign for the current platform.
Supports macOS (Developer ID + notarization), Linux (GPG), and Windows (Signtool).
"""

import os
import sys
import subprocess
import platform
from pathlib import Path
from flimkit._version import __version__
# Configuration
APP_NAME = "FLIMKit"
MAIN_SCRIPT = "main.py"
VERSION = __version__

# Platform-specific config
MACOS_TEAM_ID = os.getenv("APPLE_TEAM_ID")
MACOS_DEVELOPER_ID = os.getenv("APPLE_DEVELOPER_ID")
MACOS_APPLE_ID = os.getenv("APPLE_ID")
MACOS_APP_PASSWORD = os.getenv("APPLE_APP_PASSWORD")

WINDOWS_CERT_PATH = os.getenv("WINDOWS_CERT_PATH")
WINDOWS_CERT_PASSWORD = os.getenv("WINDOWS_CERT_PASSWORD")

LINUX_GPG_KEY = os.getenv("GPG_KEY_ID")


def run_command(cmd, shell=False):
    """Run a shell command and return success status."""
    print(f"\n▶ {' '.join(cmd) if isinstance(cmd, list) else cmd}")
    try:
        subprocess.run(cmd, shell=shell, check=True, text=True)
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ Command failed with exit code {e.returncode}")
        return False


def prepare_icon(source_png):
    """
    Convert source PNG to platform-specific icon format.

    Returns the path to the icon file that should be passed to PyInstaller,
    or None if no conversion is needed.
    """
    if not source_png or not Path(source_png).exists():
        return None

    system = platform.system()
    source_png = Path(source_png)

    if system == "Darwin":
        iconset_dir = source_png.with_suffix(".iconset")
        iconset_dir.mkdir(exist_ok=True)
        sizes = [16, 32, 64, 128, 256, 512, 1024]
        for size in sizes:
            filename = f"icon_{size}x{size}.png"
            out_path = iconset_dir / filename
            subprocess.run(
                ["sips", "-z", str(size), str(size), str(source_png), "--out", str(out_path)],
                check=True, capture_output=True
            )
            if size <= 512:
                filename_2x = f"icon_{size}x{size}@2x.png"
                out_path_2x = iconset_dir / filename_2x
                subprocess.run(
                    ["sips", "-z", str(size*2), str(size*2), str(source_png), "--out", str(out_path_2x)],
                    check=True, capture_output=True
                )
        icns_path = source_png.with_suffix(".icns")
        subprocess.run(
            ["iconutil", "-c", "icns", "-o", str(icns_path), str(iconset_dir)],
            check=True, capture_output=True
        )
        subprocess.run(["rm", "-rf", str(iconset_dir)])
        return icns_path

    elif system == "Windows":
        try:
            from PIL import Image
            ico_path = source_png.with_suffix(".ico")
            img = Image.open(source_png)
            sizes = [(16, 16), (32, 32), (48, 48), (64, 64), (128, 128), (256, 256)]
            img.save(ico_path, format="ICO", sizes=sizes)
            return ico_path
        except ImportError:
            print("Warning: Pillow not installed; cannot create ICO. Using PNG (may not work).")
            return source_png

    else:  # Linux
        return source_png


def generate_mpl_cache():
    """
    Pre-warm the matplotlib font cache into mpl-cache/ so PyInstaller can
    bundle it.  This eliminates the 10-30 s font-scan delay on first launch
    of the frozen app.

    The cache only needs rebuilding when matplotlib or system fonts change,
    but regenerating it on every build is cheap (~2 s) and keeps it fresh.
    """
    cache_dir = Path("mpl-cache")

    print(f"\n{'='*60}")
    print("Pre-warming matplotlib font cache")
    print(f"{'='*60}")

    cache_dir.mkdir(exist_ok=True)

    # Run font manager rebuild in a subprocess so MPLCONFIGDIR is isolated
    # to mpl-cache/ and doesn't touch the developer's own ~/.config/matplotlib.
    script = (
        "import os; "
        "os.environ['MPLCONFIGDIR'] = 'mpl-cache'; "
        "import matplotlib; "
        "matplotlib.use('Agg'); "
        "import matplotlib.font_manager as fm; "
        "fm._load_fontmanager(try_read_cache=False); "
        "print('  font manager rebuilt'); "
        "import matplotlib.pyplot as plt; "
        "plt.figure(); plt.close(); "
        "print('  pyplot initialised')"
    )

    import subprocess
    result = subprocess.run(
        [sys.executable, "-c", script],
        text=True,
        capture_output=True,
    )

    if result.returncode != 0:
        print("✗ mpl-cache generation failed:")
        print(result.stderr)
        sys.exit(1)

    if result.stdout:
        for line in result.stdout.strip().splitlines():
            print(f"  {line}")

    # Sanity-check: font list JSON must exist after the rebuild.
    font_lists = list(cache_dir.glob("fontlist-*.json"))
    if not font_lists:
        print("✗ mpl-cache/ exists but contains no fontlist-*.json — rebuild failed.")
        sys.exit(1)

    print(f"✓ mpl-cache ready  ({len(list(cache_dir.iterdir()))} file(s))")


def build_app():
    """Build the app with PyInstaller."""
    print(f"\n{'='*60}")
    print(f"Building {APP_NAME} with PyInstaller")
    print(f"{'='*60}")

    system = platform.system()
    icon_path = prepare_icon("flimkit/UI/icon.png")
    if system == "Darwin":
        old_app = Path(f"dist/{APP_NAME}.app")
        if old_app.exists():
            print(f"\n▶ rm -rf {old_app}  (pre-clean old bundle)")
            subprocess.run(["rm", "-rf", str(old_app)], check=True)

    cmd = [
        sys.executable,
        "-m",
        "PyInstaller",
        "--noconfirm",   # skip the interactive "will be REMOVED" prompt
        "--name", APP_NAME,
        "--windowed",
        "--copy-metadata", "readchar",
        "--copy-metadata", "inquirer",
        "--copy-metadata", "blessed",
        "--copy-metadata", "tkinterdnd2",
        "--copy-metadata", "TKinterModernThemes",
        "--collect-data", "TKinterModernThemes",
        "--collect-data", "scipy",   # bundles .npz data files (sobol etc.)
        "--collect-data", "numpy",   # bundles numpy data files
        "--hidden-import", "tkinter",
        "--hidden-import", "tkinter.ttk",
        "--hidden-import", "PIL",
        "--hidden-import", "inquirer",
        "--hidden-import", "readchar",
        "--hidden-import", "blessed",
        "--hidden-import", "tqdm",
        "--hidden-import", "matplotlib",
        "--hidden-import", "numpy",
        "--hidden-import", "cv2",
        "--hidden-import", "opencv_python",
        "--hidden-import", "phasorpy",
        "--hidden-import", "pandas",
        "--hidden-import", "scipy",
        "--hidden-import", "xarray",
        "--hidden-import", "tifffile",
        "--hidden-import", "ptufile",
        "--hidden-import", "openpyxl",
        "--hidden-import", "tkinterdnd2",
        "--hidden-import", "TKinterModernThemes",
        "--add-data", "mpl-cache:mpl-cache",   # pre-warmed font cache
        "--add-data", "flimkit:flimkit",
        "--add-data", "flimkit/UI/icon.png:flimkit",
    ]

    # Platform-specific packaging mode
    if system == "Darwin":
        cmd.append("--onedir")   # single process, proper .app bundle
    else:
        cmd.append("--onefile")  # single portable binary on Windows/Linux

    if icon_path:
        cmd.extend(["--icon", str(icon_path)])

    cmd.append(MAIN_SCRIPT)
    return run_command(cmd)


def sign_macos():
    """Sign the macOS app bundle."""
    print(f"\n{'='*60}")
    print(f"Signing macOS app (self-signed)")
    print(f"{'='*60}")

    app_path = f"dist/{APP_NAME}.app"

    print("\n1 Code signing with self-signed certificate...")
    if not run_command(["codesign", "--deep", "--force", "--sign", "-", app_path]):
        return False

    print("\n2 Verifying signature...")
    run_command(["codesign", "-v", "--deep", app_path])

    print("\n✓ macOS app signed.")
    print("   Locally built apps are not quarantined so will open without a")
    print("   security prompt on this machine.")
    print("   For distribution, use a paid Apple Developer ID + notarization.")
    return True


def sign_windows():
    """Sign the Windows executable."""
    print(f"\n{'='*60}")
    print(f"Signing Windows executable")
    print(f"{'='*60}")

    if not WINDOWS_CERT_PATH or not WINDOWS_CERT_PASSWORD:
        print("✗ Missing environment variables:")
        print("  WINDOWS_CERT_PATH - Path to .pfx certificate")
        print("  WINDOWS_CERT_PASSWORD - Certificate password")
        print("⚠ App built but not signed")
        return True

    exe_path = f"dist/{APP_NAME}.exe"

    signtool_check = run_command("where signtool", shell=True)
    if not signtool_check:
        print("✗ signtool not found. Install Windows SDK or Visual Studio")
        print("⚠ App built but not signed")
        return True

    sign_cmd = [
        "signtool", "sign",
        "/f", WINDOWS_CERT_PATH,
        "/p", WINDOWS_CERT_PASSWORD,
        "/t", "http://timestamp.comodoca.com",
        "/d", APP_NAME,
        exe_path,
    ]

    if not run_command(sign_cmd):
        print("⚠ Signing failed. App built but not signed.")
        return True

    print("\n✓ Windows executable signed and ready!")
    return True


def sign_linux():
    """Sign the Linux executable with GPG."""
    print(f"\n{'='*60}")
    print(f"Signing Linux executable")
    print(f"{'='*60}")

    if not LINUX_GPG_KEY:
        print("⚠ GPG_KEY_ID not set. Skipping GPG signature.")
        return True

    exe_path = f"dist/{APP_NAME}"

    gpg_check = run_command("which gpg", shell=True)
    if not gpg_check:
        print("✗ GPG not found. Install gnupg package.")
        return True

    sign_cmd = [
        "gpg", "--detach-sign",
        "--default-key", LINUX_GPG_KEY,
        exe_path,
    ]

    if not run_command(sign_cmd):
        print("⚠ GPG signing failed.")
        return True

    print("\n✓ Linux executable signed and ready!")
    return True


def main():
    """Main build and sign process."""
    print(f"\n{'='*60}")
    print(f"FLIMKit Build & Sign Script")
    print(f"Platform: {platform.system()}")
    print(f"{'='*60}")

    if not Path("main.py").exists():
        print("✗ main.py not found. Run this script from the FLIMKit root directory.")
        sys.exit(1)

    generate_mpl_cache()

    if not build_app():
        print("✗ Build failed!")
        sys.exit(1)

    system = platform.system()
    success = False

    if system == "Darwin":
        success = sign_macos()
    elif system == "Windows":
        success = sign_windows()
    elif system == "Linux":
        success = sign_linux()
    else:
        print(f"✗ Unsupported platform: {system}")
        sys.exit(1)

    if not success:
        print("\n✗ Signing failed!")
        sys.exit(1)

    print(f"\n{'='*60}")
    print(f"✓ Build complete!")
    print(f"{'='*60}")

    if system == "Darwin":
        print(f"\nApp location: dist/{APP_NAME}.app")
        print(f"Run with: open dist/{APP_NAME}.app")
    elif system == "Windows":
        print(f"\nExecutable location: dist/{APP_NAME}.exe")
    else:
        print(f"\nExecutable location: dist/{APP_NAME}")
        print(f"Run with: ./dist/{APP_NAME}")


if __name__ == "__main__":
    main()

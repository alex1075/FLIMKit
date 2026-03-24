# Build and Sign Guide

## Quick Start

```bash
python build_and_sign.py
```

The script automatically:
1. Builds compiled binaries with PyInstaller
2. Self-signs the macOS app
3. Creates Windows and Linux executables

All binaries are in `dist/`

---

## What's Generated

| Platform | Output | File |
|----------|--------|------|
| macOS | Signed app | `dist/FLIMKit.app/` |
| Windows | Unsigned executable | `dist/FLIMKit.exe` |
| Linux | Unsigned executable | `dist/FLIMKit` |

---

## macOS Self-Signed Certificate

The macOS app is automatically signed with a self-signed certificate:

**Benefits:**
- ✅ No setup required
- ✅ One-time security warning for users
- ✅ Completely free
- ✅ Works great for beta and testing

**First-time user experience:**
1. User double-clicks the app
2. macOS shows: "Cannot open because the developer cannot be verified"
3. User goes to: System Preferences → Security & Privacy → Open Anyway
4. App opens and runs normally
5. macOS remembers the choice for future launches

**Or for developers/testers:**
```bash
xattr -d com.apple.quarantine dist/FLIMKit.app
./dist/FLIMKit.app/Contents/MacOS/FLIMKit
```

---

## Manual Signing

If you want to manually sign the app:

```bash
# Self-sign
codesign --deep --force --sign - dist/FLIMKit.app

# Verify
codesign -v --deep dist/FLIMKit.app

# Remove quarantine attribute
xattr -d com.apple.quarantine dist/FLIMKit.app
```

---

## Troubleshooting

**macOS: "Cannot open because the developer cannot be verified"**

This is expected with self-signed certs. Solutions:
1. Right-click → Open (one-time approval)
2. Or remove quarantine: `xattr -d com.apple.quarantine dist/FLIMKit.app`

**Windows: "Windows protected your PC"**

Also normal and expected for unsigned binaries. User clicks "More info" → "Run anyway"

**Linux: Permission denied**

```bash
chmod +x dist/FLIMKit
./dist/FLIMKit
```

**Rebuild fresh**

```bash
rm -rf dist build
python build_and_sign.py
```

---

## GitHub Actions

To enable automated builds on every push:

1. Ensure `.github/workflows/build.yml` exists (already configured)
2. Push to repository
3. Binaries automatically build and upload to Releases

No additional secrets needed for self-signed builds!

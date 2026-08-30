# Voice Detector - Expo Go App

## Setup (One-time)

### 1. Install Node.js
Download from: https://nodejs.org (LTS version)

### 2. Install Expo CLI
```bash
npm install -g expo-cli
```

### 3. Install Expo Go on your phone
- **Android:** Search "Expo Go" in Play Store
- **iPhone:** Search "Expo Go" in App Store

## Run the App

```bash
cd expo_app
npm install
npx expo start
```

A QR code will appear in your terminal.

### On your phone:
1. Open **Expo Go** app
2. Tap **Scan QR Code**
3. Scan the QR code from the terminal
4. The app will load on your phone instantly!

## How to Use

1. **Start the Python server** on your PC:
   ```bash
   python -m voice_detection_app.app
   ```

2. **Find your PC's IP:**
   ```bash
   ipconfig  # Look for "IPv4 Address"
   ```

3. **In the Expo app:**
   - Enter your PC IP (e.g. `192.168.1.100`)
   - API Key: `vd_dev_key_2024`
   - Tap **Connect & Start**

4. **Record & Detect:**
   - Tap the microphone button
   - Speak for a few seconds
   - Tap Stop
   - Results appear automatically!

## Features
- Real-time voice cloning detection
- Native microphone recording
- Risk score with color coding
- Scan history
- Haptic feedback for results

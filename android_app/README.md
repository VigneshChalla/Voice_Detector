# Voice Clone Detector - Android App

## How to Build APK

### Option 1: Android Studio (Recommended)
1. Download [Android Studio](https://developer.android.com/studio)
2. Open Android Studio → `File → Open` → select the `android_app` folder
3. Wait for Gradle sync (may take 5-10 minutes)
4. Click `Build → Build Bundle(s) / APK(s) → Build APK(s)`
5. APK will be at: `app/build/outputs/apk/debug/app-debug.apk`

### Option 2: Command Line
```bash
cd android_app
./gradlew assembleDebug
# APK at: app/build/outputs/apk/debug/app-debug.apk
```

## How to Install on Phone
1. Copy the APK to your phone
2. Open the APK file on your phone
3. Tap "Install" (may need to enable "Install from Unknown Sources")
4. Open "Voice Detector" app

## How to Use
1. Enter your PC's IP address (e.g. `http://192.168.1.100:8000`)
2. Enter API Key: `vd_dev_key_2024`
3. Tap "Connect & Start"
4. Tap the microphone button to record
5. Tap "Analyze Voice" to get results

## Requirements
- Python server running: `python -m voice_detection_app.app`
- Phone and PC on same WiFi network
- Android 8.0+ (API 26+)

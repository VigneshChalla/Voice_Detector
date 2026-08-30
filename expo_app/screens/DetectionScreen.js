import React, { useState, useRef } from 'react';
import {
  View,
  Text,
  TouchableOpacity,
  StyleSheet,
  Alert,
  ActivityIndicator,
  ScrollView,
  Animated,
} from 'react-native';
import { Audio } from 'expo-av';
import * as Haptics from 'expo-haptics';
import { detectVoice } from '../services/api';

export default function DetectionScreen({ route }) {
  const { serverUrl, apiKey } = route.params;
  const [recording, setRecording] = useState(null);
  const [isRecording, setIsRecording] = useState(false);
  const [analyzing, setAnalyzing] = useState(false);
  const [result, setResult] = useState(null);
  const [history, setHistory] = useState([]);
  const [status, setStatus] = useState('Connected');
  const pulseAnim = useRef(new Animated.Value(1)).current;

  const startRecording = async () => {
    try {
      const permission = await Audio.requestPermissionsAsync();
      if (!permission.granted) {
        Alert.alert(
          'Microphone Permission',
          'Please allow microphone access in your phone settings.',
          [
            { text: 'Cancel', style: 'cancel' },
            { text: 'Open Settings', onPress: () => Audio.requestPermissionsAsync() },
          ]
        );
        return;
      }

      await Audio.setAudioModeAsync({
        allowsRecordingIOS: true,
        playsInSilentModeIOS: true,
      });

      const { recording } = await Audio.Recording.createAsync(
        Audio.RecordingOptionsPresets.HIGH_QUALITY
      );

      setRecording(recording);
      setIsRecording(true);
      setResult(null);
      setStatus('Recording...');

      // Pulse animation
      Animated.loop(
        Animated.sequence([
          Animated.timing(pulseAnim, { toValue: 1.2, duration: 500, useNativeDriver: true }),
          Animated.timing(pulseAnim, { toValue: 1, duration: 500, useNativeDriver: true }),
        ])
      ).start();

      Haptics.impactAsync(Haptics.ImpactFeedbackStyle.Medium);
    } catch (err) {
      Alert.alert('Recording Error', err.message);
    }
  };

  const stopRecording = async () => {
    if (!recording) return;

    try {
      await recording.stopAndUnloadAsync();
      await Audio.setAudioModeAsync({ allowsRecordingIOS: false });

      const uri = recording.getURI();
      setRecording(null);
      setIsRecording(false);
      pulseAnim.stopAnimation();
      pulseAnim.setValue(1);
      setStatus('Recording stopped');

      Haptics.impactAsync(Haptics.ImpactFeedbackStyle.Light);

      // Auto-analyze
      analyzeAudio(uri);
    } catch (err) {
      Alert.alert('Stop Error', err.message);
    }
  };

  const analyzeAudio = async (uri) => {
    setAnalyzing(true);
    setStatus('Analyzing...');

    try {
      const data = await detectVoice(serverUrl, apiKey, uri);
      setResult(data);
      setStatus('Analysis complete');

      setHistory(prev => [{
        time: new Date().toLocaleTimeString(),
        score: data.synthetic_probability,
        level: data.risk_level,
        isSynthetic: data.is_synthetic,
      }, ...prev].slice(0, 20));

      if (data.risk_level === 'HIGH') {
        Haptics.notificationAsync(Haptics.NotificationFeedbackType.Error);
      } else if (data.risk_level === 'MEDIUM') {
        Haptics.notificationAsync(Haptics.NotificationFeedbackType.Warning);
      } else {
        Haptics.notificationAsync(Haptics.NotificationFeedbackType.Success);
      }
    } catch (err) {
      Alert.alert('Analysis Failed', err.message);
      setStatus('Analysis failed');
    } finally {
      setAnalyzing(false);
    }
  };

  const getRiskColor = (level) => {
    switch (level) {
      case 'HIGH': return '#FF4444';
      case 'MEDIUM': return '#FFAA00';
      case 'LOW': return '#FFCC00';
      default: return '#00FF88';
    }
  };

  return (
    <ScrollView style={styles.container} contentContainerStyle={styles.content}>
      {/* Status Bar */}
      <View style={styles.statusBar}>
        <View style={styles.statusDotContainer}>
          <View style={[styles.statusDot, { backgroundColor: '#00FF88' }]} />
          <Text style={styles.statusText}>{status}</Text>
        </View>
      </View>

      {/* Record Button */}
      <View style={styles.recordSection}>
        <Animated.View style={{ transform: [{ scale: pulseAnim }] }}>
          <TouchableOpacity
            style={[styles.recordButton, isRecording && styles.recordButtonActive]}
            onPress={isRecording ? stopRecording : startRecording}
            disabled={analyzing}
          >
            <Text style={styles.recordButtonText}>
              {isRecording ? '⏹\n\nStop' : '🎤\n\nRecord'}
            </Text>
          </TouchableOpacity>
        </Animated.View>

        {analyzing && (
          <ActivityIndicator size="large" color="#7B2FF7" style={{ marginTop: 16 }} />
        )}
      </View>

      {/* Result Card */}
      {result && (
        <View style={[
          styles.resultCard,
          { borderColor: getRiskColor(result.risk_level) }
        ]}>
          <Text style={styles.resultIcon}>
            {result.is_synthetic ? '🚨' : '✅'}
          </Text>
          <Text style={[
            styles.resultLabel,
            { color: getRiskColor(result.risk_level) }
          ]}>
            {result.is_synthetic ? 'SYNTHETIC DETECTED' : 'GENUINE VOICE'}
          </Text>
          <Text style={[styles.resultScore, { color: getRiskColor(result.risk_level) }]}>
            {(result.synthetic_probability * 100).toFixed(1)}%
          </Text>
          <View style={styles.meter}>
            <View style={[
              styles.meterFill,
              {
                width: `${result.synthetic_probability * 100}%`,
                backgroundColor: getRiskColor(result.risk_level),
              }
            ]} />
          </View>
          <Text style={styles.resultRecommendation}>
            {result.recommendation}
          </Text>
        </View>
      )}

      {/* History */}
      {history.length > 0 && (
        <View style={styles.historyCard}>
          <Text style={styles.historyTitle}>Recent Scans</Text>
          {history.map((item, index) => (
            <View key={index} style={styles.historyItem}>
              <Text style={styles.historyTime}>{item.time}</Text>
              <Text style={[styles.historyRisk, { color: getRiskColor(item.level) }]}>
                {item.level} - {(item.score * 100).toFixed(1)}%
              </Text>
            </View>
          ))}
        </View>
      )}
    </ScrollView>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#0A0A1A',
  },
  content: {
    padding: 16,
    paddingBottom: 40,
  },
  statusBar: {
    backgroundColor: '#1A1A35',
    borderRadius: 10,
    padding: 12,
    marginBottom: 20,
  },
  statusDotContainer: {
    flexDirection: 'row',
    alignItems: 'center',
  },
  statusDot: {
    width: 8,
    height: 8,
    borderRadius: 4,
    marginRight: 8,
  },
  statusText: {
    color: '#FFF',
    fontSize: 14,
  },
  recordSection: {
    alignItems: 'center',
    marginBottom: 24,
  },
  recordButton: {
    width: 180,
    height: 180,
    borderRadius: 90,
    backgroundColor: '#00D4FF',
    justifyContent: 'center',
    alignItems: 'center',
    shadowColor: '#00D4FF',
    shadowOffset: { width: 0, height: 0 },
    shadowOpacity: 0.5,
    shadowRadius: 20,
    elevation: 10,
  },
  recordButtonActive: {
    backgroundColor: '#FF4444',
    shadowColor: '#FF4444',
  },
  recordButtonText: {
    fontSize: 20,
    fontWeight: 'bold',
    color: '#FFF',
    textAlign: 'center',
  },
  resultCard: {
    backgroundColor: '#12122A',
    borderRadius: 16,
    padding: 24,
    borderWidth: 2,
    marginBottom: 16,
    alignItems: 'center',
  },
  resultIcon: {
    fontSize: 48,
    marginBottom: 8,
  },
  resultLabel: {
    fontSize: 22,
    fontWeight: 'bold',
    marginBottom: 8,
  },
  resultScore: {
    fontSize: 48,
    fontWeight: 'bold',
    marginBottom: 12,
  },
  meter: {
    width: '100%',
    height: 6,
    backgroundColor: '#1E1E3F',
    borderRadius: 3,
    marginBottom: 12,
    overflow: 'hidden',
  },
  meterFill: {
    height: '100%',
    borderRadius: 3,
  },
  resultRecommendation: {
    fontSize: 13,
    color: '#AAA',
    textAlign: 'center',
    lineHeight: 18,
  },
  historyCard: {
    backgroundColor: '#12122A',
    borderRadius: 12,
    padding: 16,
  },
  historyTitle: {
    fontSize: 14,
    color: '#888',
    marginBottom: 12,
  },
  historyItem: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    paddingVertical: 8,
    borderBottomWidth: 1,
    borderBottomColor: '#1E1E3F',
  },
  historyTime: {
    color: '#666',
    fontSize: 12,
  },
  historyRisk: {
    fontWeight: '600',
    fontSize: 13,
  },
});

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

export default function DetectionScreen({ route, navigation }) {
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

        {/* Live Screening Button */}
        <TouchableOpacity
          style={styles.liveButton}
          onPress={() => navigation.navigate('LiveScreening', { serverUrl, apiKey })}
        >
          <View style={styles.liveButtonInner}>
            <View style={styles.liveDot} />
            <Text style={styles.liveButtonText}>LIVE SCREENING</Text>
          </View>
          <Text style={styles.liveButtonSubtext}>Real-time continuous analysis</Text>
        </TouchableOpacity>
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

          {/* Similarity Breakdown */}
          <View style={styles.similarityRow}>
            <View style={styles.similarityBox}>
              <Text style={styles.similarityLabel}>Human Similarity</Text>
              <Text style={[styles.similarityValue, { color: '#00FF88' }]}>{(result.final_human_percent ?? (result.genuine_probability*100)).toFixed(1)}%</Text>
            </View>
            <View style={styles.similarityBox}>
              <Text style={styles.similarityLabel}>AI Similarity</Text>
              <Text style={[styles.similarityValue, { color: '#FF4444' }]}>{(result.final_synthetic_percent ?? (result.synthetic_probability*100)).toFixed(1)}%</Text>
            </View>
          </View>

          {/* ML vs Forensic */}
          {result.ml_probability !== undefined && (
            <View style={styles.forensicRow}>
              <View style={styles.forensicBox}>
                <Text style={styles.forensicLabel}>ML Model</Text>
                <Text style={styles.forensicValue}>{(result.ml_probability*100).toFixed(1)}% AI</Text>
              </View>
              <View style={styles.forensicBox}>
                <Text style={styles.forensicLabel}>Forensic</Text>
                <Text style={styles.forensicValue}>{(result.forensic_score*100).toFixed(1)}% AI</Text>
              </View>
              <View style={styles.forensicBox}>
                <Text style={styles.forensicLabel}>Agreement</Text>
                <Text style={[styles.forensicValue, { color: result.agreement==='AGREE' ? '#00FF88' : '#FFAA00' }]}>{result.agreement || '-'}</Text>
              </View>
            </View>
          )}
          {result.confidence !== undefined && (
            <Text style={styles.confidenceText}>Confidence: {result.confidence}% | {result.analysis_summary}</Text>
          )}

          {/* Forensic Factors */}
          {result.forensic_factors && Object.keys(result.forensic_factors).length > 0 && (
            <View style={styles.factorsCard}>
              <Text style={styles.factorsTitle}>🔬 Forensic Analysis (per factor)</Text>
              {Object.entries(result.forensic_factors).map(([name, f]) => (
                <View key={name} style={styles.factorRow}>
                  <View style={styles.factorLeft}>
                    <Text style={styles.factorName}>{name.replace(/_/g,' ')}</Text>
                    <Text style={styles.factorInterp}>{f.interpretation}</Text>
                  </View>
                  <View style={styles.factorRight}>
                    <Text style={[styles.factorPercent, { color: f.status==='AI' ? '#FF4444' : f.status==='HUMAN' ? '#00FF88' : '#FFAA00' }]}>{f.synthetic_percent}% AI</Text>
                    <Text style={styles.factorStatus}>{f.status}</Text>
                  </View>
                </View>
              ))}
            </View>
          )}
          {result.dominant_clues && result.dominant_clues.length > 0 && (
            <View style={styles.cluesBox}>
              <Text style={styles.cluesTitle}>Top Clues:</Text>
              {result.dominant_clues.map((c,i) => (
                <Text key={i} style={styles.clueText}>• {c.factor}: {c.interpretation} ({c.confidence}% conf)</Text>
              ))}
            </View>
          )}

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
  similarityRow: { flexDirection: 'row', justifyContent: 'space-between', width: '100%', marginVertical: 12, gap: 12 },
  similarityBox: { flex: 1, backgroundColor: '#1A1A35', borderRadius: 10, padding: 12, alignItems: 'center' },
  similarityLabel: { color: '#888', fontSize: 11, marginBottom: 4 },
  similarityValue: { fontSize: 20, fontWeight: 'bold' },
  forensicRow: { flexDirection: 'row', justifyContent: 'space-between', width: '100%', marginBottom: 12, gap: 8 },
  forensicBox: { flex: 1, backgroundColor: '#1A1A35', borderRadius: 8, padding: 10, alignItems: 'center' },
  forensicLabel: { color: '#888', fontSize: 10, marginBottom: 2 },
  forensicValue: { color: '#FFF', fontSize: 12, fontWeight: '600' },
  confidenceText: { color: '#AAA', fontSize: 11, textAlign: 'center', marginBottom: 12, fontStyle: 'italic' },
  factorsCard: { width: '100%', backgroundColor: '#0F0F2A', borderRadius: 10, padding: 12, marginBottom: 12 },
  factorsTitle: { color: '#7B2FF7', fontSize: 13, fontWeight: 'bold', marginBottom: 10, textAlign: 'center' },
  factorRow: { flexDirection: 'row', justifyContent: 'space-between', paddingVertical: 8, borderBottomWidth: 1, borderBottomColor: '#1E1E3F' },
  factorLeft: { flex: 2 },
  factorName: { color: '#FFF', fontSize: 11, fontWeight: '600', textTransform: 'capitalize' },
  factorInterp: { color: '#888', fontSize: 10, marginTop: 2 },
  factorRight: { alignItems: 'flex-end', justifyContent: 'center' },
  factorPercent: { fontSize: 12, fontWeight: 'bold' },
  factorStatus: { fontSize: 10, color: '#666', marginTop: 2 },
  cluesBox: { width: '100%', backgroundColor: '#1A1A35', borderRadius: 8, padding: 12, marginBottom: 12 },
  cluesTitle: { color: '#FFAA00', fontSize: 11, fontWeight: 'bold', marginBottom: 6 },
  clueText: { color: '#CCC', fontSize: 11, marginBottom: 4, lineHeight: 14 },
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
  liveButton: {
    backgroundColor: '#1A1A35',
    borderRadius: 12,
    padding: 16,
    marginTop: 16,
    borderWidth: 1,
    borderColor: '#FF4444',
    alignItems: 'center',
    width: '100%',
  },
  liveButtonInner: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 8,
  },
  liveDot: {
    width: 8,
    height: 8,
    borderRadius: 4,
    backgroundColor: '#FF4444',
  },
  liveButtonText: {
    color: '#FF4444',
    fontSize: 16,
    fontWeight: 'bold',
  },
  liveButtonSubtext: {
    color: '#666',
    fontSize: 11,
    marginTop: 4,
  },
});

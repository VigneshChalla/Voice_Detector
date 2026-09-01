import React, { useState, useRef, useEffect, useCallback } from 'react';
import {
  View,
  Text,
  TouchableOpacity,
  StyleSheet,
  Alert,
  ActivityIndicator,
  ScrollView,
  Animated,
  Dimensions,
} from 'react-native';
import { Audio } from 'expo-av';
import * as Haptics from 'expo-haptics';
import { detectVoice } from '../services/api';

const CHUNK_DURATION_MS = 4000;
const SCREEN_WIDTH = Dimensions.get('window').width;

export default function LiveScreeningScreen({ route }) {
  const { serverUrl, apiKey } = route.params;
  const [isLive, setIsLive] = useState(false);
  const [isProcessing, setIsProcessing] = useState(false);
  const [segmentResults, setSegmentResults] = useState([]);
  const [overallRisk, setOverallRisk] = useState(null);
  const [status, setStatus] = useState('Ready');
  const [elapsedTime, setElapsedTime] = useState(0);
  const [segmentCount, setSegmentCount] = useState(0);

  const recordingRef = useRef(null);
  const intervalRef = useRef(null);
  const timerRef = useRef(null);
  const stopFlagRef = useRef(false);
  const pulseAnim = useRef(new Animated.Value(1)).current;
  const waveAnim = useRef(new Animated.Value(0)).current;

  useEffect(() => {
    return () => {
      cleanup();
    };
  }, []);

  const cleanup = async () => {
    stopFlagRef.current = true;
    if (intervalRef.current) clearInterval(intervalRef.current);
    if (timerRef.current) clearInterval(timerRef.current);
    if (recordingRef.current) {
      try {
        await recordingRef.current.stopAndUnloadAsync();
      } catch {}
    }
    pulseAnim.stopAnimation();
    waveAnim.stopAnimation();
  };

  const startLiveScreening = async () => {
    try {
      const permission = await Audio.requestPermissionsAsync();
      if (!permission.granted) {
        Alert.alert('Permission Required', 'Microphone access is needed for live screening.');
        return;
      }

      await Audio.setAudioModeAsync({
        allowsRecordingIOS: true,
        playsInSilentModeIOS: true,
      });

      stopFlagRef.current = false;
      setIsLive(true);
      setSegmentResults([]);
      setOverallRisk(null);
      setElapsedTime(0);
      setSegmentCount(0);
      setStatus('Starting live screening...');

      Animated.loop(
        Animated.parallel([
          Animated.sequence([
            Animated.timing(pulseAnim, { toValue: 1.15, duration: 600, useNativeDriver: true }),
            Animated.timing(pulseAnim, { toValue: 1, duration: 600, useNativeDriver: true }),
          ]),
          Animated.sequence([
            Animated.timing(waveAnim, { toValue: 1, duration: 1000, useNativeDriver: false }),
            Animated.timing(waveAnim, { toValue: 0, duration: 1000, useNativeDriver: false }),
          ]),
        ])
      ).start();

      timerRef.current = setInterval(() => {
        setElapsedTime(prev => prev + 1);
      }, 1000);

      Haptics.impactAsync(Haptics.ImpactFeedbackStyle.Medium);

      await recordAndAnalyze();
    } catch (err) {
      Alert.alert('Error', err.message);
      setIsLive(false);
    }
  };

  const recordAndAnalyze = async () => {
    if (stopFlagRef.current) return;

    try {
      const { recording } = await Audio.Recording.createAsync(
        Audio.RecordingOptionsPresets.HIGH_QUALITY,
        undefined,
        CHUNK_DURATION_MS
      );
      recordingRef.current = recording;

      setStatus('Recording segment...');

      await new Promise(resolve => setTimeout(resolve, CHUNK_DURATION_MS + 200));

      if (stopFlagRef.current) {
        try { await recording.stopAndUnloadAsync(); } catch {}
        return;
      }

      await recording.stopAndUnloadAsync();
      recordingRef.current = null;

      const uri = recording.getURI();
      setStatus('Analyzing...');
      setIsProcessing(true);

      const data = await detectVoice(serverUrl, apiKey, uri);

      const segmentResult = {
        index: segmentCount + 1,
        score: data.synthetic_probability,
        risk_level: data.risk_level,
        is_synthetic: data.is_synthetic,
        recommendation: data.recommendation,
        timestamp: new Date().toLocaleTimeString(),
        ml_prob: data.ml_probability,
        forensic_score: data.forensic_score,
        human_sim: data.final_human_percent,
        ai_sim: data.final_synthetic_percent,
        factors: data.forensic_factors,
        clues: data.dominant_clues,
        summary: data.analysis_summary,
      };

      setSegmentResults(prev => [segmentResult, ...prev]);
      setSegmentCount(prev => prev + 1);

      if (data.risk_level === 'HIGH') {
        Haptics.notificationAsync(Haptics.NotificationFeedbackType.Error);
      } else if (data.risk_level === 'MEDIUM') {
        Haptics.notificationAsync(Haptics.NotificationFeedbackType.Warning);
      }

      setOverallRisk({
        latestScore: data.synthetic_probability,
        latestLevel: data.risk_level,
        latestSynthetic: data.is_synthetic,
        isSynthetic: data.is_synthetic,
      });

      setIsProcessing(false);
      setStatus('Live screening...');

      if (!stopFlagRef.current) {
        setTimeout(() => recordAndAnalyze(), 500);
      }
    } catch (err) {
      setIsProcessing(false);
      if (!stopFlagRef.current) {
        setStatus('Retrying...');
        setTimeout(() => recordAndAnalyze(), 2000);
      }
    }
  };

  const stopLiveScreening = async () => {
    stopFlagRef.current = true;
    setIsLive(false);
    setIsProcessing(false);
    setStatus('Stopped');

    if (intervalRef.current) clearInterval(intervalRef.current);
    if (timerRef.current) clearInterval(timerRef.current);
    if (recordingRef.current) {
      try {
        await recordingRef.current.stopAndUnloadAsync();
      } catch {}
      recordingRef.current = null;
    }

    await Audio.setAudioModeAsync({ allowsRecordingIOS: false });
    pulseAnim.stopAnimation();
    pulseAnim.setValue(1);
    waveAnim.stopAnimation();
    waveAnim.setValue(0);

    Haptics.impactAsync(Haptics.ImpactFeedbackStyle.Light);

    if (segmentResults.length > 0) {
      const avgScore = segmentResults.reduce((sum, r) => sum + r.score, 0) / segmentResults.length;
      const highCount = segmentResults.filter(r => r.risk_level === 'HIGH').length;
      const medCount = segmentResults.filter(r => r.risk_level === 'MEDIUM').length;
      const finalLevel = highCount > 0 ? 'HIGH' : medCount > 0 ? 'MEDIUM' : 'LOW';

      setOverallRisk({
        latestScore: avgScore,
        latestLevel: finalLevel,
        latestSynthetic: avgScore > 0.5,
        summary: true,
        totalSegments: segmentResults.length,
        highAlerts: highCount,
        mediumAlerts: medCount,
      });
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

  const formatTime = (seconds) => {
    const m = Math.floor(seconds / 60);
    const s = seconds % 60;
    return `${m.toString().padStart(2, '0')}:${s.toString().padStart(2, '0')}`;
  };

  return (
    <ScrollView style={styles.container} contentContainerStyle={styles.content}>
      {/* Header */}
      <View style={styles.header}>
        <Text style={styles.headerTitle}>Live Screening</Text>
        <Text style={styles.headerSubtitle}>Real-time voice authenticity analysis</Text>
      </View>

      {/* Live Status Bar */}
      <View style={[styles.statusBar, isLive && styles.statusBarActive]}>
        <View style={styles.statusRow}>
          <View style={[styles.statusDot, { backgroundColor: isLive ? '#FF4444' : '#00FF88' }]} />
          <Text style={styles.statusText}>
            {isLive ? 'LIVE' : 'IDLE'}
          </Text>
          {isLive && <Text style={styles.timerText}>{formatTime(elapsedTime)}</Text>}
        </View>
        <Text style={styles.statusText}>{status}</Text>
      </View>

      {/* Main Record Button */}
      <View style={styles.recordSection}>
        <Animated.View style={{ transform: [{ scale: isLive ? pulseAnim : new Animated.Value(1) }] }}>
          <TouchableOpacity
            style={[styles.recordButton, isLive && styles.recordButtonActive]}
            onPress={isLive ? stopLiveScreening : startLiveScreening}
            disabled={isProcessing && !isLive}
          >
            {isLive ? (
              <View style={styles.recordButtonInner}>
                <View style={styles.stopIcon} />
              </View>
            ) : (
              <Text style={styles.recordButtonText}>GO LIVE</Text>
            )}
          </TouchableOpacity>
        </Animated.View>

        {isLive && (
          <View style={styles.segmentCounter}>
            <Text style={styles.segmentCounterText}>
              Segment {segmentCount + 1} | Analyzing: {isProcessing ? 'YES' : 'WAITING'}
            </Text>
          </View>
        )}
      </View>

      {/* Overall Risk Display */}
      {overallRisk && (
        <View style={[styles.riskCard, { borderColor: getRiskColor(overallRisk.latestLevel) }]}>
          {overallRisk.summary ? (
            <>
              <Text style={styles.riskCardTitle}>SESSION SUMMARY</Text>
              <Text style={[styles.riskLevel, { color: getRiskColor(overallRisk.latestLevel) }]}>
                {overallRisk.latestLevel} RISK
              </Text>
              <View style={styles.summaryRow}>
                <Text style={styles.summaryLabel}>Total Segments:</Text>
                <Text style={styles.summaryValue}>{overallRisk.totalSegments}</Text>
              </View>
              <View style={styles.summaryRow}>
                <Text style={styles.summaryLabel}>Avg Score:</Text>
                <Text style={[styles.summaryValue, { color: getRiskColor(overallRisk.latestLevel) }]}>
                  {(overallRisk.latestScore * 100).toFixed(1)}%
                </Text>
              </View>
              <View style={styles.summaryRow}>
                <Text style={styles.summaryLabel}>High Alerts:</Text>
                <Text style={[styles.summaryValue, { color: overallRisk.highAlerts > 0 ? '#FF4444' : '#00FF88' }]}>
                  {overallRisk.highAlerts}
                </Text>
              </View>
              <View style={styles.summaryRow}>
                <Text style={styles.summaryLabel}>Medium Alerts:</Text>
                <Text style={[styles.summaryValue, { color: overallRisk.mediumAlerts > 0 ? '#FFAA00' : '#00FF88' }]}>
                  {overallRisk.mediumAlerts}
                </Text>
              </View>
            </>
          ) : (
            <>
              <Text style={[styles.riskLevel, { color: getRiskColor(overallRisk.latestLevel) }]}>
                {overallRisk.isSynthetic ? 'SYNTHETIC DETECTED' : 'GENUINE VOICE'}
              </Text>
              <Text style={[styles.riskScore, { color: getRiskColor(overallRisk.latestLevel) }]}>
                {(overallRisk.latestScore * 100).toFixed(1)}%
              </Text>
              <View style={styles.meter}>
                <View style={[
                  styles.meterFill,
                  {
                    width: `${Math.min(overallRisk.latestScore * 100, 100)}%`,
                    backgroundColor: getRiskColor(overallRisk.latestLevel),
                  }
                ]} />
              </View>
            </>
          )}
        </View>
      )}

      {/* Waveform Visual */}
      {isLive && (
        <View style={styles.waveContainer}>
          {[...Array(20)].map((_, i) => (
            <Animated.View
              key={i}
              style={[
                styles.waveBar,
                {
                  height: waveAnim.interpolate({
                    inputRange: [0, 1],
                    outputRange: [4, Math.random() * 40 + 10],
                  }),
                  backgroundColor: getRiskColor(overallRisk?.latestLevel || 'SAFE'),
                  opacity: 0.6 + Math.random() * 0.4,
                },
              ]}
            />
          ))}
        </View>
      )}

      {/* Segment History */}
      {segmentResults.length > 0 && (
        <View style={styles.historyCard}>
          <Text style={styles.historyTitle}>Segment History (tap for forensic)</Text>
          {segmentResults.map((item, index) => (
            <View key={index} style={[styles.historyItem, { borderLeftColor: getRiskColor(item.risk_level) }]}>
              <View style={styles.historyLeft}>
                <Text style={styles.historyIndex}>#{item.index}</Text>
                <Text style={styles.historyTime}>{item.timestamp}</Text>
                {item.ml_prob !== undefined && (
                  <Text style={styles.historySub}>ML {(item.ml_prob*100).toFixed(0)}% | For {(item.forensic_score*100).toFixed(0)}%</Text>
                )}
              </View>
              <View style={styles.historyCenter}>
                <Text style={[styles.historyScore, { color: getRiskColor(item.risk_level) }]}>
                  {(item.score * 100).toFixed(1)}%
                </Text>
                <Text style={[styles.historyRisk, { color: getRiskColor(item.risk_level) }]}>
                  {item.risk_level}
                </Text>
                {item.ai_sim !== undefined && (
                  <Text style={styles.historySim}>H:{item.human_sim?.toFixed(0)}% A:{item.ai_sim?.toFixed(0)}%</Text>
                )}
              </View>
              <Text style={styles.historyIcon}>
                {item.is_synthetic ? '🚨' : '✅'}
              </Text>
            </View>
          ))}
          {segmentResults[0]?.factors && (
            <View style={styles.forensicExpand}>
              <Text style={styles.forensicExpandTitle}>🔬 Latest Forensic Breakdown</Text>
              {Object.entries(segmentResults[0].factors).slice(0,4).map(([n,f])=>(
                <View key={n} style={styles.miniFactor}>
                  <Text style={styles.miniFactorName}>{n.replace(/_/g,' ')}</Text>
                  <Text style={[styles.miniFactorVal, {color: f.status==='AI' ? '#FF4444' : f.status==='HUMAN' ? '#00FF88' : '#FFAA00'}]}>{f.synthetic_percent}% AI • {f.status}</Text>
                </View>
              ))}
              {segmentResults[0].clues && segmentResults[0].clues[0] && (
                <Text style={styles.clueLine}>💡 {segmentResults[0].clues[0].interpretation}</Text>
              )}
              {segmentResults[0].summary && (
                <Text style={styles.summaryLine}>{segmentResults[0].summary}</Text>
              )}
            </View>
          )}
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
  header: {
    marginBottom: 20,
  },
  headerTitle: {
    fontSize: 24,
    fontWeight: 'bold',
    color: '#FFF',
  },
  headerSubtitle: {
    fontSize: 13,
    color: '#888',
    marginTop: 4,
  },
  statusBar: {
    backgroundColor: '#1A1A35',
    borderRadius: 10,
    padding: 12,
    marginBottom: 20,
  },
  statusBarActive: {
    backgroundColor: '#2A1A1A',
    borderWidth: 1,
    borderColor: '#FF4444',
  },
  statusRow: {
    flexDirection: 'row',
    alignItems: 'center',
    marginBottom: 4,
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
    fontWeight: '600',
  },
  timerText: {
    color: '#FF4444',
    fontSize: 14,
    fontWeight: 'bold',
    marginLeft: 'auto',
  },
  recordSection: {
    alignItems: 'center',
    marginBottom: 24,
  },
  recordButton: {
    width: 160,
    height: 160,
    borderRadius: 80,
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
  recordButtonInner: {
    alignItems: 'center',
  },
  stopIcon: {
    width: 40,
    height: 40,
    backgroundColor: '#FFF',
    borderRadius: 6,
  },
  recordButtonText: {
    fontSize: 20,
    fontWeight: 'bold',
    color: '#FFF',
  },
  segmentCounter: {
    marginTop: 12,
    alignItems: 'center',
  },
  segmentCounterText: {
    color: '#888',
    fontSize: 12,
  },
  riskCard: {
    backgroundColor: '#12122A',
    borderRadius: 16,
    padding: 20,
    borderWidth: 2,
    marginBottom: 16,
    alignItems: 'center',
  },
  riskCardTitle: {
    color: '#888',
    fontSize: 12,
    marginBottom: 8,
    letterSpacing: 1,
  },
  riskLevel: {
    fontSize: 22,
    fontWeight: 'bold',
    marginBottom: 4,
  },
  riskScore: {
    fontSize: 48,
    fontWeight: 'bold',
    marginBottom: 12,
  },
  meter: {
    width: '100%',
    height: 6,
    backgroundColor: '#1E1E3F',
    borderRadius: 3,
    overflow: 'hidden',
  },
  meterFill: {
    height: '100%',
    borderRadius: 3,
  },
  summaryRow: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    width: '100%',
    paddingVertical: 6,
    borderBottomWidth: 1,
    borderBottomColor: '#1E1E3F',
  },
  summaryLabel: {
    color: '#888',
    fontSize: 14,
  },
  summaryValue: {
    color: '#FFF',
    fontSize: 14,
    fontWeight: '600',
  },
  waveContainer: {
    flexDirection: 'row',
    justifyContent: 'center',
    alignItems: 'flex-end',
    height: 60,
    marginBottom: 16,
    gap: 3,
  },
  waveBar: {
    width: 4,
    borderRadius: 2,
    minHeight: 4,
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
    alignItems: 'center',
    paddingVertical: 10,
    borderBottomWidth: 1,
    borderBottomColor: '#1E1E3F',
    borderLeftWidth: 3,
    paddingLeft: 10,
    marginLeft: 0,
  },
  historyLeft: {
    flex: 1,
  },
  historyIndex: {
    color: '#666',
    fontSize: 12,
    fontWeight: '600',
  },
  historyTime: {
    color: '#555',
    fontSize: 10,
  },
  historyCenter: {
    flex: 2,
    alignItems: 'center',
  },
  historyScore: {
    fontWeight: 'bold',
    fontSize: 16,
  },
  historyRisk: {
    fontSize: 10,
    fontWeight: '600',
  },
  historyIcon: {
    fontSize: 16,
    marginLeft: 8,
  },
  historySub: { color: '#666', fontSize: 9, marginTop: 2 },
  historySim: { color: '#888', fontSize: 9, marginTop: 1 },
  forensicExpand: { marginTop: 12, backgroundColor: '#0F0F2A', borderRadius: 10, padding: 12 },
  forensicExpandTitle: { color: '#7B2FF7', fontSize: 12, fontWeight: 'bold', marginBottom: 8, textAlign: 'center' },
  miniFactor: { flexDirection: 'row', justifyContent: 'space-between', paddingVertical: 4, borderBottomWidth: 1, borderBottomColor: '#1E1E3F' },
  miniFactorName: { color: '#FFF', fontSize: 10, textTransform: 'capitalize' },
  miniFactorVal: { fontSize: 10, fontWeight: 'bold' },
  clueLine: { color: '#FFAA00', fontSize: 11, marginTop: 8, fontStyle: 'italic' },
  summaryLine: { color: '#AAA', fontSize: 10, marginTop: 6, textAlign: 'center' },
});

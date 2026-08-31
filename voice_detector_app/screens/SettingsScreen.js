import React, { useState, useEffect } from 'react';
import {
  View,
  Text,
  TextInput,
  TouchableOpacity,
  StyleSheet,
  Alert,
  ActivityIndicator,
  KeyboardAvoidingView,
  Platform,
} from 'react-native';
import { getServerConfig, saveServerConfig, healthCheck } from '../services/api';

export default function SettingsScreen({ navigation }) {
  const [serverUrl, setServerUrl] = useState('');
  const [apiKey, setApiKey] = useState('vd_dev_key_2024');
  const [loading, setLoading] = useState(false);
  const [connected, setConnected] = useState(null);

  useEffect(() => {
    loadConfig();
  }, []);

  const loadConfig = async () => {
    const config = await getServerConfig();
    if (config.url) setServerUrl(config.url);
    if (config.apiKey) setApiKey(config.apiKey);
  };

  const handleConnect = async () => {
    if (!serverUrl.trim()) {
      Alert.alert('Error', 'Enter server IP address');
      return;
    }

    let url = serverUrl.trim();
    if (!url.startsWith('http')) {
      url = `http://${url}:8000`;
    }

    setLoading(true);
    setConnected(null);

    try {
      await healthCheck(url, apiKey);
      await saveServerConfig(url, apiKey);
      setConnected(true);
      navigation.replace('Detection', { serverUrl: url, apiKey });
    } catch (err) {
      setConnected(false);
      Alert.alert('Connection Failed', `Cannot connect to server.\n\nMake sure:\n1. Server is running\n2. Phone & PC on same WiFi\n3. IP address is correct\n\nError: ${err.message}`);
    } finally {
      setLoading(false);
    }
  };

  return (
    <KeyboardAvoidingView
      style={styles.container}
      behavior={Platform.OS === 'ios' ? 'padding' : 'height'}
    >
      <View style={styles.content}>
        <Text style={styles.title}>Voice Clone Detector</Text>
        <Text style={styles.subtitle}>AI-Powered Detection</Text>

        <View style={styles.card}>
          <Text style={styles.cardTitle}>Server Settings</Text>

          <Text style={styles.label}>Server IP</Text>
          <TextInput
            style={styles.input}
            value={serverUrl}
            onChangeText={setServerUrl}
            placeholder="192.168.1.100"
            placeholderTextColor="#555"
            autoCapitalize="none"
            autoCorrect={false}
          />

          <Text style={styles.label}>API Key</Text>
          <TextInput
            style={styles.input}
            value={apiKey}
            onChangeText={setApiKey}
            placeholder="vd_dev_key_2024"
            placeholderTextColor="#555"
            autoCapitalize="none"
            autoCorrect={false}
            secureTextEntry
          />

          <TouchableOpacity
            style={[styles.button, loading && styles.buttonDisabled]}
            onPress={handleConnect}
            disabled={loading}
          >
            {loading ? (
              <ActivityIndicator color="#FFF" />
            ) : (
              <Text style={styles.buttonText}>Connect & Start</Text>
            )}
          </TouchableOpacity>

          <Text style={styles.hint}>
            Make sure your phone and PC are on the same WiFi network.
          </Text>
        </View>
      </View>
    </KeyboardAvoidingView>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#0A0A1A',
  },
  content: {
    flex: 1,
    justifyContent: 'center',
    padding: 24,
  },
  title: {
    fontSize: 28,
    fontWeight: 'bold',
    color: '#FFF',
    textAlign: 'center',
    marginBottom: 4,
  },
  subtitle: {
    fontSize: 14,
    color: '#888',
    textAlign: 'center',
    marginBottom: 40,
  },
  card: {
    backgroundColor: '#12122A',
    borderRadius: 16,
    padding: 20,
    borderWidth: 1,
    borderColor: '#1E1E3F',
  },
  cardTitle: {
    fontSize: 16,
    color: '#AAA',
    marginBottom: 16,
  },
  label: {
    fontSize: 12,
    color: '#888',
    marginBottom: 6,
    marginLeft: 4,
  },
  input: {
    backgroundColor: '#1A1A35',
    borderRadius: 8,
    padding: 14,
    fontSize: 16,
    color: '#FFF',
    borderWidth: 1,
    borderColor: '#1E1E3F',
    marginBottom: 16,
  },
  button: {
    backgroundColor: '#7B2FF7',
    borderRadius: 12,
    padding: 16,
    alignItems: 'center',
    marginTop: 8,
  },
  buttonDisabled: {
    opacity: 0.6,
  },
  buttonText: {
    color: '#FFF',
    fontSize: 16,
    fontWeight: '600',
  },
  hint: {
    fontSize: 12,
    color: '#555',
    textAlign: 'center',
    marginTop: 16,
  },
});

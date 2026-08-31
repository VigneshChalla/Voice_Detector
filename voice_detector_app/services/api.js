import AsyncStorage from '@react-native-async-storage/async-storage';

const API_BASE_URL_KEY = 'api_base_url';
const API_KEY_KEY = 'api_key';

export async function getServerConfig() {
  const url = await AsyncStorage.getItem(API_BASE_URL_KEY);
  const key = await AsyncStorage.getItem(API_KEY_KEY);
  return { url: url || '', apiKey: key || 'vd_dev_key_2024' };
}

export async function saveServerConfig(url, apiKey) {
  await AsyncStorage.setItem(API_BASE_URL_KEY, url);
  await AsyncStorage.setItem(API_KEY_KEY, apiKey);
}

export async function healthCheck(url, apiKey) {
  const response = await fetch(`${url}/api/v1/health`, {
    headers: { 'X-API-Key': apiKey },
  });
  if (!response.ok) throw new Error(`Server error: ${response.status}`);
  return await response.json();
}

export async function detectVoice(url, apiKey, audioUri, callerId = 'expo_user') {
  const formData = new FormData();

  formData.append('file', {
    uri: audioUri,
    type: 'audio/wav',
    name: 'recording.wav',
  });
  formData.append('caller_id', callerId);
  formData.append('call_type', 'regular_call');

  const response = await fetch(`${url}/api/v1/detect`, {
    method: 'POST',
    headers: {
      'X-API-Key': apiKey,
      'Content-Type': 'multipart/form-data',
    },
    body: formData,
  });

  if (!response.ok) {
    const text = await response.text();
    throw new Error(`Detection failed: ${response.status} - ${text}`);
  }

  return await response.json();
}

export async function enrollSpeaker(url, apiKey, audioUri, speakerId, label = '') {
  const formData = new FormData();

  formData.append('file', {
    uri: audioUri,
    type: 'audio/wav',
    name: 'enrollment.wav',
  });
  formData.append('speaker_id', speakerId);
  formData.append('label', label);

  const response = await fetch(`${url}/api/v1/speaker/enroll`, {
    method: 'POST',
    headers: {
      'X-API-Key': apiKey,
      'Content-Type': 'multipart/form-data',
    },
    body: formData,
  });

  if (!response.ok) throw new Error(`Enrollment failed: ${response.status}`);
  return await response.json();
}

export async function verifySpeaker(url, apiKey, audioUri, speakerId) {
  const formData = new FormData();

  formData.append('file', {
    uri: audioUri,
    type: 'audio/wav',
    name: 'verify.wav',
  });
  formData.append('speaker_id', speakerId);

  const response = await fetch(`${url}/api/v1/speaker/verify`, {
    method: 'POST',
    headers: {
      'X-API-Key': apiKey,
      'Content-Type': 'multipart/form-data',
    },
    body: formData,
  });

  if (!response.ok) throw new Error(`Verification failed: ${response.status}`);
  return await response.json();
}

import React from 'react';
import { NavigationContainer } from '@react-navigation/native';
import { createNativeStackNavigator } from '@react-navigation/native-stack';
import SettingsScreen from './screens/SettingsScreen';
import DetectionScreen from './screens/DetectionScreen';

const Stack = createNativeStackNavigator();

export default function App() {
  return (
    <NavigationContainer>
      <Stack.Navigator
        initialRouteName="Settings"
        screenOptions={{
          headerShown: false,
          contentStyle: { backgroundColor: '#0A0A1A' },
        }}
      >
        <Stack.Screen name="Settings" component={SettingsScreen} />
        <Stack.Screen name="Detection" component={DetectionScreen} />
      </Stack.Navigator>
    </NavigationContainer>
  );
}

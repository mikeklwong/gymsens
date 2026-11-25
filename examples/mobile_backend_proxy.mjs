// Example Node 20+ backend route for a gym's existing mobile app.
// Do NOT put the gymsens X-API-Key in an iOS/Android/React Native app.
// Authenticate the end-user with the gym's own backend before invoking this.
// This example is a handler, not a complete authenticated server.
export async function fetchEquipmentForApp(gymId) {
  if (!/^[A-Za-z0-9_-]{1,64}$/.test(gymId)) throw new Error('Invalid gym id');
  const response = await fetch(`${process.env.GYMSENS_API_BASE}/v1/gyms/${gymId}/equipment`, {
    headers: { 'X-API-Key': process.env.GYMSENS_READ_KEY },
    signal: AbortSignal.timeout(5000)
  });
  if (!response.ok) throw new Error(`gymsens returned ${response.status}`);
  const snapshot = await response.json();
  return snapshot.equipment.map(item => ({
    equipmentId: item.id, name: item.name, available: item.availability,
    updatedAt: item.updated_at, hasBelongings: item.has_detected_belongings
  }));
}

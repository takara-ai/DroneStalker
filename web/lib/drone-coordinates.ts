/**
 * Utility functions for parsing and querying drone coordinate data
 */

export interface DronePosition {
  time: number;
  x1: number;
  y1: number;
  x2: number;
  y2: number;
}

export interface NormalizedPosition {
  x: number; // percentage (0-1)
  y: number; // percentage (0-1)
  width: number; // percentage (0-1)
  height: number; // percentage (0-1)
}

/**
 * Parse coordinates.txt file content
 * Format: time: x1, y1, x2, y2
 */
export function parseCoordinates(content: string): DronePosition[] {
  const lines = content.trim().split("\n");
  const positions: DronePosition[] = [];

  for (const line of lines) {
    const match = line.match(
      /^([\d.]+):\s*([\d.]+),\s*([\d.]+),\s*([\d.]+),\s*([\d.]+)$/
    );
    if (match) {
      positions.push({
        time: parseFloat(match[1]),
        x1: parseFloat(match[2]),
        y1: parseFloat(match[3]),
        x2: parseFloat(match[4]),
        y2: parseFloat(match[5]),
      });
    }
  }

  return positions;
}

/**
 * Find the closest position to a given timestamp using binary search
 * Returns the position with the closest time value
 */
export function getPositionAtTime(
  positions: DronePosition[],
  targetTime: number
): DronePosition | null {
  if (positions.length === 0) return null;
  if (targetTime <= positions[0].time) return positions[0];
  if (targetTime >= positions[positions.length - 1].time) {
    return positions[positions.length - 1];
  }

  // Binary search for closest position
  let left = 0;
  let right = positions.length - 1;

  while (left < right) {
    const mid = Math.floor((left + right) / 2);
    if (positions[mid].time < targetTime) {
      left = mid + 1;
    } else {
      right = mid;
    }
  }

  // Check which is closer: positions[left] or positions[left - 1]
  if (left > 0) {
    const prev = positions[left - 1];
    const curr = positions[left];
    const prevDiff = Math.abs(targetTime - prev.time);
    const currDiff = Math.abs(targetTime - curr.time);
    return prevDiff <= currDiff ? prev : curr;
  }

  return positions[left];
}

/**
 * Convert absolute coordinates to normalized percentages
 * @param position The drone position with absolute coordinates
 * @param videoWidth The width of the video in pixels
 * @param videoHeight The height of the video in pixels
 */
export function normalizePosition(
  position: DronePosition,
  videoWidth: number,
  videoHeight: number
): NormalizedPosition {
  const x = position.x1 / videoWidth;
  const y = position.y1 / videoHeight;
  const width = (position.x2 - position.x1) / videoWidth;
  const height = (position.y2 - position.y1) / videoHeight;

  return {
    x: Math.max(0, Math.min(1, x)),
    y: Math.max(0, Math.min(1, y)),
    width: Math.max(0, Math.min(1, width)),
    height: Math.max(0, Math.min(1, height)),
  };
}

/**
 * Convert absolute coordinates to normalized percentages accounting for object-cover scaling
 * This handles the case where the video is scaled/cropped to fill the container
 * @param position The drone position with absolute coordinates
 * @param videoWidth The intrinsic width of the video in pixels
 * @param videoHeight The intrinsic height of the video in pixels
 * @param containerWidth The displayed width of the video container in pixels
 * @param containerHeight The displayed height of the video container in pixels
 */
export function normalizePositionWithObjectCover(
  position: DronePosition,
  videoWidth: number,
  videoHeight: number,
  containerWidth: number,
  containerHeight: number
): NormalizedPosition {
  // Calculate scale factor for object-cover (covers entire container, maintains aspect ratio)
  const scaleX = containerWidth / videoWidth;
  const scaleY = containerHeight / videoHeight;
  const scale = Math.max(scaleX, scaleY); // object-cover uses max to ensure coverage

  // Calculate scaled video dimensions
  const scaledVideoWidth = videoWidth * scale;
  const scaledVideoHeight = videoHeight * scale;

  // Calculate offsets (how much is cropped due to object-cover)
  const offsetX = (scaledVideoWidth - containerWidth) / 2;
  const offsetY = (scaledVideoHeight - containerHeight) / 2;

  // Convert original coordinates to scaled coordinates
  const scaledX1 = position.x1 * scale;
  const scaledY1 = position.y1 * scale;
  const scaledX2 = position.x2 * scale;
  const scaledY2 = position.y2 * scale;

  // Adjust for cropping offset and normalize to container dimensions
  const x = (scaledX1 - offsetX) / containerWidth;
  const y = (scaledY1 - offsetY) / containerHeight;
  const width = (scaledX2 - scaledX1) / containerWidth;
  const height = (scaledY2 - scaledY1) / containerHeight;

  return {
    x: Math.max(0, Math.min(1, x)),
    y: Math.max(0, Math.min(1, y)),
    width: Math.max(0, Math.min(1, width)),
    height: Math.max(0, Math.min(1, height)),
  };
}

/**
 * Load and parse coordinates from the public folder
 */
export async function loadCoordinates(): Promise<DronePosition[]> {
  const response = await fetch("/coordinates.txt");
  if (!response.ok) {
    throw new Error(`Failed to load coordinates: ${response.statusText}`);
  }
  const content = await response.text();
  return parseCoordinates(content);
}

"use client";

import { useEffect, useRef } from "react";

export function SoundEffects() {
  const backgroundNoiseRef = useRef<HTMLAudioElement | null>(null);
  const mousePressRef = useRef<HTMLAudioElement | null>(null);
  const mouseReleaseRef = useRef<HTMLAudioElement | null>(null);
  const keyPressRef = useRef<HTMLAudioElement | null>(null);
  const keyReleaseRef = useRef<HTMLAudioElement | null>(null);

  // Initialize audio elements and start immediately (interaction handled by ClickToEnter)
  useEffect(() => {
    // Background noise - looping
    const bgNoise = new Audio("/white-noise.wav");
    bgNoise.loop = true;
    bgNoise.volume = 0.05; // Adjust volume as needed
    backgroundNoiseRef.current = bgNoise;

    // Start background sound immediately
    bgNoise.play().catch((error) => {
      console.warn("Could not play background noise:", error);
    });

    // Mouse sounds
    const mousePress = new Audio("/mouse-press.wav");
    mousePress.volume = 0.1;
    mousePressRef.current = mousePress;

    const mouseRelease = new Audio("/mouse-release.wav");
    mouseRelease.volume = 0.1;
    mouseReleaseRef.current = mouseRelease;

    // Keyboard sounds
    const keyPress = new Audio("/key-press.wav");
    keyPress.volume = 0.1;
    keyPressRef.current = keyPress;

    const keyRelease = new Audio("/key-release.wav");
    keyRelease.volume = 0.1;
    keyReleaseRef.current = keyRelease;

    // Cleanup function
    return () => {
      bgNoise.pause();
      bgNoise.src = "";
    };
  }, []);

  // Handle mouse events
  useEffect(() => {
    const handleMouseDown = (e: MouseEvent) => {
      // Only play sound for left mouse button
      if (e.button === 0 && mousePressRef.current) {
        // Clone and play to allow overlapping sounds
        const sound = mousePressRef.current.cloneNode() as HTMLAudioElement;
        sound.volume = mousePressRef.current.volume;
        sound.play().catch((error) => {
          console.warn("Could not play mouse press sound:", error);
        });
      }
    };

    const handleMouseUp = (e: MouseEvent) => {
      // Only play sound for left mouse button
      if (e.button === 0 && mouseReleaseRef.current) {
        // Clone and play to allow overlapping sounds
        const sound = mouseReleaseRef.current.cloneNode() as HTMLAudioElement;
        sound.volume = mouseReleaseRef.current.volume;
        sound.play().catch((error) => {
          console.warn("Could not play mouse release sound:", error);
        });
      }
    };

    window.addEventListener("mousedown", handleMouseDown);
    window.addEventListener("mouseup", handleMouseUp);

    return () => {
      window.removeEventListener("mousedown", handleMouseDown);
      window.removeEventListener("mouseup", handleMouseUp);
    };
  }, []);

  // Handle keyboard events
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      if (keyPressRef.current) {
        // Clone and play to allow overlapping sounds
        const sound = keyPressRef.current.cloneNode() as HTMLAudioElement;
        sound.volume = keyPressRef.current.volume;
        sound.play().catch((error) => {
          console.warn("Could not play key press sound:", error);
        });
      }
    };

    const handleKeyUp = (e: KeyboardEvent) => {
      // Ignore modifier keys and function keys
      if (
        e.metaKey ||
        e.ctrlKey ||
        e.altKey ||
        e.key.startsWith("F") ||
        e.key === "Meta" ||
        e.key === "Control" ||
        e.key === "Alt" ||
        e.key === "Shift"
      ) {
        return;
      }

      if (keyReleaseRef.current) {
        // Clone and play to allow overlapping sounds
        const sound = keyReleaseRef.current.cloneNode() as HTMLAudioElement;
        sound.volume = keyReleaseRef.current.volume;
        sound.play().catch((error) => {
          console.warn("Could not play key release sound:", error);
        });
      }
    };

    window.addEventListener("keydown", handleKeyDown);
    window.addEventListener("keyup", handleKeyUp);

    return () => {
      window.removeEventListener("keydown", handleKeyDown);
      window.removeEventListener("keyup", handleKeyUp);
    };
  }, []);

  // This component doesn't render anything
  return null;
}

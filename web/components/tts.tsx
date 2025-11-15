"use client";

import { useEffect } from "react";
import { useStore } from "@/lib/store";

export function Tts() {
  const popFromTtsQueue = useStore((state) => state.popFromTtsQueue);

  useEffect(() => {
    let currentAudio: HTMLAudioElement | null = null;

    const processQueue = async () => {
      const text = popFromTtsQueue()?.toUpperCase();
      if (!text) return;

      // Stop any currently running audio
      if (currentAudio) {
        currentAudio.pause();
        currentAudio.currentTime = 0;
        // We revoke the previous object URL, if present
        if (currentAudio.src.startsWith("blob:")) {
          URL.revokeObjectURL(currentAudio.src);
        }
        currentAudio = null;
      }

      try {
        const response = await fetch(
          `/api/tts?text=${encodeURIComponent(text)}`
        );
        const audioBlob = await response.blob();
        const audioUrl = URL.createObjectURL(audioBlob);
        const audio = new Audio(audioUrl);

        currentAudio = audio;

        audio.onended = () => {
          if (currentAudio && currentAudio.src === audioUrl) {
            URL.revokeObjectURL(audioUrl);
            currentAudio = null;
          }
        };

        await audio.play();
      } catch (error) {
        console.error("TTS error:", error);
      }
    };

    const interval = setInterval(processQueue, 100);
    return () => {
      clearInterval(interval);
      // Cleanup: stop current audio if effect is removed/unmounted
      if (currentAudio) {
        currentAudio.pause();
        if (currentAudio.src.startsWith("blob:")) {
          URL.revokeObjectURL(currentAudio.src);
        }
        currentAudio = null;
      }
    };
  }, [popFromTtsQueue]);

  return null;
}

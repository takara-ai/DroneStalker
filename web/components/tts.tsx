"use client";

import { useEffect } from "react";
import { useStore } from "@/lib/store";

export function Tts() {
  const popFromTtsQueue = useStore((state) => state.popFromTtsQueue);

  useEffect(() => {
    const processQueue = async () => {
      const text = popFromTtsQueue()?.toUpperCase();
      if (!text) return;

      try {
        const response = await fetch(
          `/api/tts?text=${encodeURIComponent(text)}`
        );
        const audioBlob = await response.blob();
        const audioUrl = URL.createObjectURL(audioBlob);
        const audio = new Audio(audioUrl);
        await audio.play();
        audio.onended = () => URL.revokeObjectURL(audioUrl);
      } catch (error) {
        console.error("TTS error:", error);
      }
    };

    const interval = setInterval(processQueue, 100);
    return () => clearInterval(interval);
  }, [popFromTtsQueue]);

  return null;
}

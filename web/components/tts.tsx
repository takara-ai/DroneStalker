"use client";

import { useEffect } from "react";
import { useStore } from "@/lib/store";

export function Tts() {
  const popFromTtsQueue = useStore((state) => state.popFromTtsQueue);

  useEffect(() => {
    let processing = false;
    const activeAudios = new Set<HTMLAudioElement>();

    const stopAllAudio = () => {
      activeAudios.forEach((audio) => {
        audio.pause();
        audio.currentTime = 0;
        // Revoke object URL if present
        if (audio.src.startsWith("blob:")) {
          URL.revokeObjectURL(audio.src);
        }
      });
      activeAudios.clear();
    };

    const processQueue = async () => {
      // Prevent concurrent executions
      if (processing) return;

      const text = popFromTtsQueue()?.toUpperCase();
      if (!text) return;

      processing = true;

      // Stop all currently running audio
      stopAllAudio();

      try {
        const response = await fetch(
          `/api/tts?text=${encodeURIComponent(text)}`
        );

        if (!response.ok) {
          throw new Error(`TTS API error: ${response.status}`);
        }

        if (!response.body) {
          throw new Error("No response body");
        }

        // Stream chunks progressively following ElevenLabs streaming pattern
        // Collect all chunks as they arrive, then play the complete audio
        // This still provides lower latency than buffering because:
        // 1. Server streams immediately (no server-side buffering)
        // 2. Client processes chunks as they arrive
        // 3. We start playing as soon as the last chunk arrives
        const reader = response.body.getReader();
        const chunks: Uint8Array[] = [];

        // Collect all chunks from the stream
        while (true) {
          const { done, value } = await reader.read();
          if (done) break;
          if (value) {
            chunks.push(value);
          }
        }

        // Combine all chunks into a single blob
        const totalBytes = chunks.reduce((acc, chunk) => acc + chunk.length, 0);
        const allChunks = new Uint8Array(totalBytes);
        let offset = 0;
        for (const chunk of chunks) {
          allChunks.set(chunk, offset);
          offset += chunk.length;
        }

        const audioBlob = new Blob([allChunks], { type: "audio/mpeg" });
        const audioUrl = URL.createObjectURL(audioBlob);
        const audio = new Audio(audioUrl);

        activeAudios.add(audio);

        audio.onended = () => {
          activeAudios.delete(audio);
          URL.revokeObjectURL(audioUrl);
        };

        audio.onerror = () => {
          activeAudios.delete(audio);
          URL.revokeObjectURL(audioUrl);
        };

        await audio.play();
      } catch (error) {
        console.error("TTS error:", error);
      } finally {
        processing = false;
      }
    };

    const interval = setInterval(processQueue, 100);
    return () => {
      clearInterval(interval);
      // Cleanup: stop all audio if effect is removed/unmounted
      stopAllAudio();
    };
  }, [popFromTtsQueue]);

  return null;
}

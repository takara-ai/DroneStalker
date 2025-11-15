"use client";

import { useStore } from "@/lib/store";
import { useEffect, useRef } from "react";

export function Tts() {
  const ttsQueue = useStore((state) => state.ttsQueue);
  const popFromTtsQueue = useStore((state) => state.popFromTtsQueue);
  const isProcessingRef = useRef(false);
  const currentAudioRef = useRef<HTMLAudioElement | null>(null);
  const currentAudioUrlRef = useRef<string | null>(null);

  useEffect(() => {
    if (ttsQueue.length > 0 && !isProcessingRef.current) {
      isProcessingRef.current = true;
      const text = popFromTtsQueue();

      if (text) {
        // Stop any currently playing audio
        if (currentAudioRef.current) {
          currentAudioRef.current.pause();
          currentAudioRef.current = null;
        }
        if (currentAudioUrlRef.current) {
          URL.revokeObjectURL(currentAudioUrlRef.current);
          currentAudioUrlRef.current = null;
        }

        fetch("/api/tts", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ text }),
        })
          .then(async (response) => {
            if (!response.ok) {
              throw new Error("Failed to generate speech");
            }

            if (!response.body) {
              throw new Error("No response body");
            }

            // Stream the audio and play as it arrives
            const reader = response.body.getReader();
            const chunks: Uint8Array[] = [];
            let totalLength = 0;

            // Read all chunks
            while (true) {
              const { done, value } = await reader.read();
              if (done) break;
              if (value) {
                chunks.push(value);
                totalLength += value.length;
              }
            }

            // Combine all chunks into a single blob
            const allChunks = new Uint8Array(totalLength);
            let offset = 0;
            for (const chunk of chunks) {
              allChunks.set(chunk, offset);
              offset += chunk.length;
            }

            const blob = new Blob([allChunks], { type: "audio/mpeg" });
            const audioUrl = URL.createObjectURL(blob);
            currentAudioUrlRef.current = audioUrl;
            const audio = new Audio(audioUrl);
            currentAudioRef.current = audio;

            audio.onended = () => {
              if (currentAudioUrlRef.current === audioUrl) {
                URL.revokeObjectURL(audioUrl);
                currentAudioUrlRef.current = null;
                currentAudioRef.current = null;
              }
              isProcessingRef.current = false;
            };

            audio.onerror = (error) => {
              console.error("Audio playback error:", error);
              if (currentAudioUrlRef.current === audioUrl) {
                URL.revokeObjectURL(audioUrl);
                currentAudioUrlRef.current = null;
                currentAudioRef.current = null;
              }
              isProcessingRef.current = false;
            };

            audio.play().catch((error) => {
              console.error("Error playing audio:", error);
              if (currentAudioUrlRef.current === audioUrl) {
                URL.revokeObjectURL(audioUrl);
                currentAudioUrlRef.current = null;
                currentAudioRef.current = null;
              }
              isProcessingRef.current = false;
            });
          })
          .catch((error) => {
            console.error("TTS error:", error);
            isProcessingRef.current = false;
          });
      } else {
        isProcessingRef.current = false;
      }
    }
  }, [ttsQueue, popFromTtsQueue]);

  // Cleanup on unmount
  useEffect(() => {
    return () => {
      if (currentAudioRef.current) {
        currentAudioRef.current.pause();
        currentAudioRef.current = null;
      }
      if (currentAudioUrlRef.current) {
        URL.revokeObjectURL(currentAudioUrlRef.current);
        currentAudioUrlRef.current = null;
      }
    };
  }, []);

  return null;
}

"use client";

import { useState, useRef, useEffect } from "react";

export default function Home() {
  const [sliderValue, setSliderValue] = useState(50);
  const [isDragging, setIsDragging] = useState(false);
  const coloredVideoRef = useRef<HTMLVideoElement>(null);
  const motionVideoRef = useRef<HTMLVideoElement>(null);
  const containerRef = useRef<HTMLDivElement>(null);

  // Sync videos - keep motion video in sync with colored video
  useEffect(() => {
    const coloredVideo = coloredVideoRef.current;
    const motionVideo = motionVideoRef.current;

    if (!coloredVideo || !motionVideo) return;

    const syncVideos = () => {
      const timeDiff = Math.abs(
        motionVideo.currentTime - coloredVideo.currentTime
      );
      // Only sync if difference is significant (more than 0.1 seconds)
      if (timeDiff > 0.1) {
        motionVideo.currentTime = coloredVideo.currentTime;
      }
    };

    const handleTimeUpdate = () => {
      syncVideos();
    };

    const handlePlay = () => {
      if (motionVideo.paused) {
        motionVideo.play();
      }
    };

    const handlePause = () => {
      if (!motionVideo.paused) {
        motionVideo.pause();
      }
    };

    coloredVideo.addEventListener("timeupdate", handleTimeUpdate);
    coloredVideo.addEventListener("play", handlePlay);
    coloredVideo.addEventListener("pause", handlePause);

    // Initial sync
    motionVideo.currentTime = coloredVideo.currentTime;

    return () => {
      coloredVideo.removeEventListener("timeupdate", handleTimeUpdate);
      coloredVideo.removeEventListener("play", handlePlay);
      coloredVideo.removeEventListener("pause", handlePause);
    };
  }, []);

  const updateSliderValue = (clientX: number) => {
    if (!containerRef.current) return;
    const rect = containerRef.current.getBoundingClientRect();
    const x = clientX - rect.left;
    const percentage = (x / rect.width) * 100;
    setSliderValue(Math.max(0, Math.min(100, percentage)));
  };

  const handleMouseDown = (e: React.MouseEvent<HTMLDivElement>) => {
    setIsDragging(true);
    updateSliderValue(e.clientX);
  };

  useEffect(() => {
    const handleMouseMove = (e: MouseEvent) => {
      if (isDragging) {
        updateSliderValue(e.clientX);
      }
    };

    const handleMouseUp = () => {
      setIsDragging(false);
    };

    if (isDragging) {
      document.addEventListener("mousemove", handleMouseMove);
      document.addEventListener("mouseup", handleMouseUp);
    }

    return () => {
      document.removeEventListener("mousemove", handleMouseMove);
      document.removeEventListener("mouseup", handleMouseUp);
    };
  }, [isDragging]);

  const splitPosition = `${sliderValue}%`;

  return (
    <div className="flex min-h-screen items-center justify-center p-8">
      <div
        ref={containerRef}
        className="relative w-full max-w-4xl aspect-video cursor-pointer select-none"
        onMouseDown={handleMouseDown}
      >
        {/* Colored video - left side */}
        <div
          className="absolute inset-0 overflow-hidden"
          style={{ clipPath: `inset(0 ${100 - sliderValue}% 0 0)` }}
        >
          <video
            ref={coloredVideoRef}
            className="w-full h-full object-contain"
            src="/colored.webm"
            autoPlay
            loop
            muted
          />
        </div>
        {/* Motion video - right side */}
        <div
          className="absolute inset-0 overflow-hidden"
          style={{ clipPath: `inset(0 0 0 ${sliderValue}%)` }}
        >
          <video
            ref={motionVideoRef}
            className="w-full h-full object-contain"
            src="/motion.webm"
            autoPlay
            loop
            muted
          />
        </div>
        {/* Slider indicator */}
        <div
          className="absolute top-0 bottom-0 w-1 bg-white/80 pointer-events-none z-10"
          style={{ left: splitPosition, transform: "translateX(-50%)" }}
        />
      </div>
    </div>
  );
}

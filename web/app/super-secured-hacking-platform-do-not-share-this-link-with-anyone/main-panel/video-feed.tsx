"use client";

import { useStore } from "@/lib/store";
import { cn } from "@/lib/utils";
import { highlightId } from "@/lib/highlight";
import { useRef, useEffect, useState } from "react";
import ProjectileCanvas, {
  type ProjectileCanvasHandle,
  TRAVEL_TIME,
} from "./projectile-canvas";
import {
  loadCoordinates,
  getPositionAtTime,
  normalizePositionWithObjectCover,
  type DronePosition,
  type NormalizedPosition,
} from "@/lib/drone-coordinates";

export default function VideoFeed() {
  const activeCamera = useStore((state) => state.activeCamera);
  const activeMotionDetection = useStore(
    (state) => state.activeMotionDetection
  );
  const activeTracking = useStore((state) => state.activeTracking);
  const scenarioState = useStore((state) => state.scenarioState);
  const triggerAction = useStore((state) => state.triggerAction);
  const activeFire = useStore((state) => state.activeFire);
  const activeAutoFire = useStore((state) => state.activeAutoFire);
  const activeLockTarget = useStore((state) => state.activeLockTarget);
  const activeMotionPrediction = useStore(
    (state) => state.activeMotionPrediction
  );
  const dataId = useStore((state) => state.dataId);
  const unlockedCode = useStore((state) => state.unlockedCode);
  const setUnlockedCode = useStore((state) => state.setUnlockedCode);
  const projectileCanvasRef = useRef<ProjectileCanvasHandle>(null);
  const coloredVideoRef = useRef<HTMLVideoElement>(null);
  const motionVideoRef = useRef<HTMLVideoElement>(null);
  const predictionVideoRef = useRef<HTMLVideoElement>(null);
  const transformContainerRef = useRef<HTMLDivElement>(null);
  const prevActiveMotionDetectionRef = useRef<boolean | undefined>(undefined);
  const coloredVideoInitializedRef = useRef<boolean>(false);
  const motionVideoInitializedRef = useRef<boolean>(false);
  // State for coordinates and position
  const [coordinates, setCoordinates] = useState<DronePosition[]>([]);
  const [normalizedPosition, setNormalizedPosition] =
    useState<NormalizedPosition | null>(null);
  const [predictedPosition, setPredictedPosition] =
    useState<NormalizedPosition | null>(null);
  const [videoLoaded, setVideoLoaded] = useState(false);
  const [containerSize, setContainerSize] = useState<{
    width: number;
    height: number;
  } | null>(null);
  const normalizedPositionRef = useRef<NormalizedPosition | null>(null);
  const predictedPositionRef = useRef<NormalizedPosition | null>(null);

  // Load coordinates on mount and when dataId changes
  useEffect(() => {
    loadCoordinates(dataId)
      .then((coords) => {
        setCoordinates(coords);
      })
      .catch((error) => {
        console.error("Failed to load coordinates:", error);
      });
  }, [dataId]);

  // Reset initialization flags when dataId changes
  useEffect(() => {
    coloredVideoInitializedRef.current = false;
    motionVideoInitializedRef.current = false;
  }, [dataId]);

  // Randomize initial video time when videos load
  useEffect(() => {
    const coloredVideo = coloredVideoRef.current;
    const motionVideo = motionVideoRef.current;

    const setupVideoRandomization = (
      video: HTMLVideoElement | null,
      initializedRef: React.MutableRefObject<boolean>
    ): (() => void) | undefined => {
      if (!video || initializedRef.current) return undefined;

      const handleLoadedMetadata = () => {
        if (video.duration && !isNaN(video.duration) && video.duration > 0) {
          // Set random time between 0 and duration
          const randomTime = Math.random() * video.duration;
          video.currentTime = randomTime;
          initializedRef.current = true;
        }
      };

      if (video.readyState >= 1) {
        // Metadata already loaded
        handleLoadedMetadata();
      } else {
        video.addEventListener("loadedmetadata", handleLoadedMetadata);
      }

      return () => {
        video.removeEventListener("loadedmetadata", handleLoadedMetadata);
      };
    };

    const cleanup1 = setupVideoRandomization(
      coloredVideo,
      coloredVideoInitializedRef
    );
    const cleanup2 = setupVideoRandomization(
      motionVideo,
      motionVideoInitializedRef
    );

    return () => {
      cleanup1?.();
      cleanup2?.();
    };
  }, [dataId]);

  // Sync video timestamps when switching between colored and motion videos
  useEffect(() => {
    // Skip sync on initial mount
    if (prevActiveMotionDetectionRef.current === undefined) {
      prevActiveMotionDetectionRef.current = activeMotionDetection;
      return;
    }

    // Only sync if the active video mode actually changed
    if (prevActiveMotionDetectionRef.current === activeMotionDetection) {
      return;
    }

    // Get the previously active video
    const prevVideo = prevActiveMotionDetectionRef.current
      ? motionVideoRef.current
      : coloredVideoRef.current;

    // Get the newly active video
    const newVideo = activeMotionDetection
      ? motionVideoRef.current
      : coloredVideoRef.current;

    // Sync the timestamp if both videos are available and the previous video has a valid time
    let cleanup: (() => void) | undefined;
    if (prevVideo && newVideo && prevVideo.readyState >= 2) {
      const currentTime = prevVideo.currentTime;
      if (currentTime > 0 && !isNaN(currentTime)) {
        // Wait for the new video to be ready before setting the time
        if (newVideo.readyState >= 2) {
          newVideo.currentTime = currentTime;
        } else {
          // If not ready yet, wait for loadeddata event
          const handleLoadedData = () => {
            newVideo.currentTime = currentTime;
            newVideo.removeEventListener("loadeddata", handleLoadedData);
          };
          newVideo.addEventListener("loadeddata", handleLoadedData);

          // Cleanup function to remove event listener if component unmounts or effect re-runs
          cleanup = () => {
            newVideo.removeEventListener("loadeddata", handleLoadedData);
          };
        }
      }
    }

    // Update the ref for next time
    prevActiveMotionDetectionRef.current = activeMotionDetection;

    // Return cleanup function if we added an event listener
    return cleanup;
  }, [activeMotionDetection]);

  // Sync prediction video with active video
  useEffect(() => {
    if (!activeMotionPrediction || !activeCamera) {
      return;
    }

    const activeVideo = activeMotionDetection
      ? motionVideoRef.current
      : coloredVideoRef.current;
    const predictionVideo = predictionVideoRef.current;

    if (!activeVideo || !predictionVideo) {
      return;
    }

    // Sync prediction video time with active video using requestAnimationFrame
    let animationFrameId: number;
    const syncLoop = () => {
      if (activeVideo.readyState >= 2 && predictionVideo.readyState >= 2) {
        const currentTime = activeVideo.currentTime;
        if (currentTime > 0 && !isNaN(currentTime)) {
          // Only update if there's a significant difference to avoid jitter
          // Use a smaller threshold (roughly one frame at 30fps)
          const timeDiff = Math.abs(predictionVideo.currentTime - currentTime);
          if (timeDiff > 0.033) {
            predictionVideo.currentTime = currentTime;
          }
        }
      }
      animationFrameId = requestAnimationFrame(syncLoop);
    };

    // Initial sync when videos are ready
    let handlePredictionVideoReady: (() => void) | null = null;
    const handleActiveVideoReady = () => {
      if (predictionVideo.readyState >= 2) {
        const currentTime = activeVideo.currentTime;
        if (currentTime > 0 && !isNaN(currentTime)) {
          predictionVideo.currentTime = currentTime;
        }
        animationFrameId = requestAnimationFrame(syncLoop);
      } else {
        handlePredictionVideoReady = () => {
          const currentTime = activeVideo.currentTime;
          if (currentTime > 0 && !isNaN(currentTime)) {
            predictionVideo.currentTime = currentTime;
          }
          animationFrameId = requestAnimationFrame(syncLoop);
          if (handlePredictionVideoReady) {
            predictionVideo.removeEventListener(
              "loadeddata",
              handlePredictionVideoReady
            );
          }
        };
        predictionVideo.addEventListener(
          "loadeddata",
          handlePredictionVideoReady
        );
      }
    };

    if (activeVideo.readyState >= 2) {
      handleActiveVideoReady();
    } else {
      activeVideo.addEventListener("loadeddata", handleActiveVideoReady);
    }

    return () => {
      if (animationFrameId) {
        cancelAnimationFrame(animationFrameId);
      }
      activeVideo.removeEventListener("loadeddata", handleActiveVideoReady);
      if (handlePredictionVideoReady) {
        predictionVideo.removeEventListener(
          "loadeddata",
          handlePredictionVideoReady
        );
      }
    };
  }, [activeMotionPrediction, activeCamera, activeMotionDetection]);

  // Track video time and update position using requestAnimationFrame for smooth updates
  useEffect(() => {
    const activeVideo = activeMotionDetection
      ? motionVideoRef.current
      : coloredVideoRef.current;

    if (!activeVideo || !activeCamera || coordinates.length === 0) {
      return;
    }

    const updatePosition = () => {
      const time = activeVideo.currentTime;

      // Get position at current time
      const position = getPositionAtTime(coordinates, time);
      if (position) {
        // Get video intrinsic dimensions
        const videoWidth = activeVideo.videoWidth || 1280;
        const videoHeight = activeVideo.videoHeight || 720;

        // Get displayed container dimensions (accounts for object-cover scaling)
        const containerWidth = activeVideo.clientWidth;
        const containerHeight = activeVideo.clientHeight;

        if (containerWidth > 0 && containerHeight > 0) {
          const normalized = normalizePositionWithObjectCover(
            position,
            videoWidth,
            videoHeight,
            containerWidth,
            containerHeight
          );
          setNormalizedPosition(normalized);
          normalizedPositionRef.current = normalized;

          // Calculate predicted position (current time + travel time in seconds)
          const predictedTime = time + TRAVEL_TIME / 1000;
          const predictedPos = getPositionAtTime(coordinates, predictedTime);
          if (predictedPos) {
            const predictedNormalized = normalizePositionWithObjectCover(
              predictedPos,
              videoWidth,
              videoHeight,
              containerWidth,
              containerHeight
            );
            setPredictedPosition(predictedNormalized);
            predictedPositionRef.current = predictedNormalized;
          } else {
            setPredictedPosition(null);
            predictedPositionRef.current = null;
          }
        }
      } else {
        // No position found within margin - clear the box
        setNormalizedPosition(null);
        normalizedPositionRef.current = null;
        setPredictedPosition(null);
        predictedPositionRef.current = null;
      }
    };

    const handleLoadedMetadata = () => {
      setVideoLoaded(true);
      updatePosition();
    };

    // Use requestAnimationFrame for smooth, high-frequency updates
    let animationFrameId: number;
    const rafLoop = () => {
      updatePosition();
      animationFrameId = requestAnimationFrame(rafLoop);
    };

    // Start the animation loop
    animationFrameId = requestAnimationFrame(rafLoop);

    // Handle initial metadata load
    if (activeVideo.readyState >= 1) {
      // Video metadata already loaded
      handleLoadedMetadata();
    } else {
      activeVideo.addEventListener("loadedmetadata", handleLoadedMetadata);
    }

    // Handle window resize to recalculate position
    const handleResize = () => {
      updatePosition();
    };
    window.addEventListener("resize", handleResize);

    return () => {
      cancelAnimationFrame(animationFrameId);
      activeVideo.removeEventListener("loadedmetadata", handleLoadedMetadata);
      window.removeEventListener("resize", handleResize);
    };
  }, [activeCamera, activeMotionDetection, coordinates]);

  // Handle projectile miss - unlock code on first miss
  const handleProjectileMiss = () => {
    if (!unlockedCode && scenarioState === "mission") {
      setUnlockedCode(true);
      highlightId("tab-motionDetection");
      setTimeout(() => {
        triggerAction("fired a projectile but missed");
      }, 0);
    }
  };

  // Autofire: fire projectiles every 0.2 seconds at the drone's center position
  // Uses predicted position if motion prediction is active and available, otherwise uses current position
  useEffect(() => {
    if (!activeAutoFire || !projectileCanvasRef.current) {
      return;
    }

    // Fire immediately on first enable if we have a position
    // Prioritize predicted position if motion prediction is active
    const initialPosition =
      activeMotionPrediction && predictedPositionRef.current
        ? predictedPositionRef.current
        : normalizedPositionRef.current;
    if (initialPosition) {
      const centerX = initialPosition.x + initialPosition.width / 2;
      const centerY = initialPosition.y + initialPosition.height / 2;
      const targetXPercent = centerX * 100;
      const targetYPercent = centerY * 100;
      projectileCanvasRef.current.addProjectile(targetXPercent, targetYPercent);
    }

    // Set up interval to fire every 0.2 seconds (200ms)
    const intervalId = setInterval(() => {
      if (!projectileCanvasRef.current) {
        return;
      }

      // Get latest position from ref - prioritize predicted position if motion prediction is active
      const currentPosition =
        activeMotionPrediction && predictedPositionRef.current
          ? predictedPositionRef.current
          : normalizedPositionRef.current;
      if (!currentPosition) {
        return;
      }

      // Calculate center position
      const centerX = currentPosition.x + currentPosition.width / 2;
      const centerY = currentPosition.y + currentPosition.height / 2;

      // Convert to percentages (0-100)
      const targetXPercent = centerX * 100;
      const targetYPercent = centerY * 100;

      projectileCanvasRef.current.addProjectile(targetXPercent, targetYPercent);
    }, 100); // 0.2 seconds = 200ms

    return () => {
      clearInterval(intervalId);
    };
  }, [activeAutoFire, activeFire, activeMotionPrediction]);

  // Update container size when it changes
  useEffect(() => {
    if (!transformContainerRef.current) return;

    const updateSize = () => {
      if (transformContainerRef.current) {
        setContainerSize({
          width: transformContainerRef.current.clientWidth,
          height: transformContainerRef.current.clientHeight,
        });
      }
    };

    updateSize();
    window.addEventListener("resize", updateSize);
    return () => window.removeEventListener("resize", updateSize);
  }, []);

  // Calculate transform for lock target feature
  // Use predicted position if motion prediction is activated and available, otherwise use current position
  const lockTargetPosition =
    activeLockTarget && activeMotionPrediction && predictedPosition
      ? predictedPosition
      : normalizedPosition;

  const lockTargetTransform =
    activeLockTarget && lockTargetPosition && containerSize
      ? (() => {
          // Get container dimensions
          const containerWidth = containerSize.width;
          const containerHeight = containerSize.height;

          // Calculate center of normalized position (0-1 range)
          const centerX = lockTargetPosition.x + lockTargetPosition.width / 2;
          const centerY = lockTargetPosition.y + lockTargetPosition.height / 2;

          // Zoom level (2x)
          const zoom = 3;

          // Calculate target center in pixels (center of viewport)
          const targetCenterX = containerWidth / 2;
          const targetCenterY = containerHeight / 2;

          // Calculate current center in pixels
          const currentCenterX = centerX * containerWidth;
          const currentCenterY = centerY * containerHeight;

          // Calculate translation needed in pixels
          // CSS transforms apply right-to-left, so scale happens first, then translate
          // When scaling from center, the center of the element stays fixed
          // We want the point at (currentCenterX, currentCenterY) to end up at (targetCenterX, targetCenterY)
          // After scaling, distances from center are multiplied by zoom
          // So we need to translate by the difference, accounting for the scaling
          // The translation happens in the scaled coordinate system
          const translateXPx = targetCenterX - currentCenterX;
          const translateYPx = targetCenterY - currentCenterY;

          // Order: scale first (applied first), translate last (applied after scale)
          // With transform-origin: center, scaling happens from the center
          return `scale(${zoom}) translate(${translateXPx}px, ${translateYPx}px)`;
        })()
      : undefined;

  return (
    <div className="border-4 flex h-full overflow-hidden opened p-2 relative">
      <div className="relative size-full flex overflow-hidden aspect-video">
        {activeCamera && (
          <div className="absolute top-4 left-4 p-2 border-4 border-red-500 text-red-500 flex items-center gap-3 px-3 z-10">
            <div className="size-3 bg-red-500 rounded-full"></div>
            LIVE FEED{activeLockTarget ? " (LOCKED)" : ""}
          </div>
        )}
        <div
          ref={transformContainerRef}
          className="relative size-full flex transition-transform duration-100"
          style={{
            transform: lockTargetTransform,
            transformOrigin: "center center",
          }}
        >
          <video
            ref={coloredVideoRef}
            src={`/data/${dataId}/colored.mp4`}
            autoPlay
            muted
            loop
            className={cn(
              "w-full h-full object-cover",
              activeCamera && !activeMotionDetection ? "opened" : "hidden"
            )}
          ></video>
          <video
            ref={motionVideoRef}
            src={`/data/${dataId}/motion.mp4`}
            autoPlay
            muted
            loop
            className={cn(
              "w-full h-full object-cover",
              activeCamera && activeMotionDetection ? "opened" : "hidden"
            )}
          ></video>
          {/* Prediction video overlay - only show when motion prediction is active */}
          <video
            ref={predictionVideoRef}
            src={`/data/${dataId}/prediction.mp4`}
            autoPlay
            muted
            loop
            className={cn(
              "absolute inset-0 w-full h-full object-cover opacity-50 pointer-events-none z-15",
              activeCamera && activeMotionPrediction ? "opened" : "hidden"
            )}
          ></video>
          {/* Drone position rectangle overlay - only show when tracking is active */}
          {activeCamera &&
            (activeTracking || activeMotionPrediction) &&
            (normalizedPosition || predictedPosition) &&
            videoLoaded && (
              <>
                {normalizedPosition && activeTracking && (
                  <div
                    className={cn(
                      "absolute border-2 border-red-500 bg-red-500/20 pointer-events-none z-20",
                      activeLockTarget && "opacity-30"
                    )}
                    style={{
                      left: `${normalizedPosition.x * 100}%`,
                      top: `${normalizedPosition.y * 100}%`,
                      width: `${normalizedPosition.width * 100}%`,
                      height: `${normalizedPosition.height * 100}%`,
                    }}
                  />
                )}
                {/* Predicted position rectangle - only show when lock target is active */}
                {activeMotionPrediction && predictedPosition && (
                  <div
                    className={cn(
                      "absolute border-2 border-blue-500 bg-blue-500/20 pointer-events-none z-25",
                      activeLockTarget && "opacity-30"
                    )}
                    style={{
                      left: `${predictedPosition.x * 100}%`,
                      top: `${predictedPosition.y * 100}%`,
                      width: `${predictedPosition.width * 100}%`,
                      height: `${predictedPosition.height * 100}%`,
                    }}
                  />
                )}
                {activeLockTarget && lockTargetPosition && (
                  <>
                    <div
                      className="absolute z-30 pointer-events-none size-10"
                      style={{
                        left: `${
                          lockTargetPosition.x * 100 +
                          lockTargetPosition.width * 50
                        }%`,
                        top: `${
                          lockTargetPosition.y * 100 +
                          lockTargetPosition.height * 50
                        }%`,
                        transform: "translate(-50%, -50%)",
                      }}
                    >
                      <svg
                        width="48"
                        height="48"
                        viewBox="0 0 48 48"
                        fill="none"
                        className="w-full h-full"
                      >
                        <circle
                          cx="24"
                          cy="24"
                          r="16"
                          stroke={
                            activeMotionPrediction && predictedPosition
                              ? "#3B82F6"
                              : "#EF4444"
                          }
                          strokeWidth="2"
                          fill="none"
                        />
                        <circle
                          cx="24"
                          cy="24"
                          r="3"
                          fill={
                            activeMotionPrediction && predictedPosition
                              ? "#3B82F6"
                              : "#EF4444"
                          }
                          opacity="0.7"
                        />
                        <line
                          x1="24"
                          y1="0"
                          x2="24"
                          y2="9"
                          stroke={
                            activeMotionPrediction && predictedPosition
                              ? "#3B82F6"
                              : "#EF4444"
                          }
                          strokeWidth="2"
                        />
                        <line
                          x1="24"
                          y1="39"
                          x2="24"
                          y2="48"
                          stroke={
                            activeMotionPrediction && predictedPosition
                              ? "#3B82F6"
                              : "#EF4444"
                          }
                          strokeWidth="2"
                        />
                        <line
                          x1="0"
                          y1="24"
                          x2="9"
                          y2="24"
                          stroke={
                            activeMotionPrediction && predictedPosition
                              ? "#3B82F6"
                              : "#EF4444"
                          }
                          strokeWidth="2"
                        />
                        <line
                          x1="39"
                          y1="24"
                          x2="48"
                          y2="24"
                          stroke={
                            activeMotionPrediction && predictedPosition
                              ? "#3B82F6"
                              : "#EF4444"
                          }
                          strokeWidth="2"
                        />
                      </svg>
                    </div>
                  </>
                )}
              </>
            )}
          {activeCamera && (activeFire || activeAutoFire) && (
            <ProjectileCanvas
              onMiss={handleProjectileMiss}
              ref={projectileCanvasRef}
              normalizedPosition={normalizedPosition}
            />
          )}
        </div>
        {!activeCamera && (
          <div className="flex flex-col items-center justify-center h-full w-full text-gray-400 text-lg select-none p-8 opacity-40 absolute inset-0">
            <div className="font-semibold">NO LIVE FEED AVAILABLE</div>
          </div>
        )}
      </div>
    </div>
  );
}

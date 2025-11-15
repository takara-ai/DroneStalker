"use client";

import { useStore } from "@/lib/store";
import { cn } from "@/lib/utils";
import { useRef, useEffect, useState } from "react";
import ProjectileCanvas, {
  type ProjectileCanvasHandle,
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
  const activeLockTarget = useStore((state) => state.activeLockTarget);
  const dataId = useStore((state) => state.dataId);
  const projectileCanvasRef = useRef<ProjectileCanvasHandle>(null);
  const coloredVideoRef = useRef<HTMLVideoElement>(null);
  const motionVideoRef = useRef<HTMLVideoElement>(null);
  const transformContainerRef = useRef<HTMLDivElement>(null);
  const prevActiveMotionDetectionRef = useRef<boolean | undefined>(undefined);
  // State for coordinates and position
  const [coordinates, setCoordinates] = useState<DronePosition[]>([]);
  const [normalizedPosition, setNormalizedPosition] =
    useState<NormalizedPosition | null>(null);
  const [videoLoaded, setVideoLoaded] = useState(false);
  const [containerSize, setContainerSize] = useState<{
    width: number;
    height: number;
  } | null>(null);

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
        }
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

  // Handle projectile firing
  const handleProjectileClick = (x: number, y: number) => {
    if (projectileCanvasRef.current) {
      projectileCanvasRef.current.addProjectile(x, y);
    }
    if (scenarioState === "mission") {
      triggerAction("fired a projectile but missed");
    }
  };

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
  const lockTargetTransform =
    activeLockTarget && normalizedPosition && containerSize
      ? (() => {
          // Get container dimensions
          const containerWidth = containerSize.width;
          const containerHeight = containerSize.height;

          // Calculate center of normalized position (0-1 range)
          const centerX = normalizedPosition.x + normalizedPosition.width / 2;
          const centerY = normalizedPosition.y + normalizedPosition.height / 2;

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
          className="relative size-full flex transition-transform duration-200"
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
          {/* Drone position rectangle overlay - only show when tracking is active */}
          {activeCamera &&
            activeTracking &&
            normalizedPosition &&
            videoLoaded && (
              <>
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
                {activeLockTarget && (
                  <>
                    <div
                      className="absolute z-30 pointer-events-none size-10"
                      style={{
                        left: `${
                          normalizedPosition.x * 100 +
                          normalizedPosition.width * 50
                        }%`,
                        top: `${
                          normalizedPosition.y * 100 +
                          normalizedPosition.height * 50
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
                          stroke="#EF4444"
                          strokeWidth="2"
                          fill="none"
                        />
                        <circle
                          cx="24"
                          cy="24"
                          r="3"
                          fill="#EF4444"
                          opacity="0.7"
                        />
                        <line
                          x1="24"
                          y1="0"
                          x2="24"
                          y2="9"
                          stroke="#EF4444"
                          strokeWidth="2"
                        />
                        <line
                          x1="24"
                          y1="39"
                          x2="24"
                          y2="48"
                          stroke="#EF4444"
                          strokeWidth="2"
                        />
                        <line
                          x1="0"
                          y1="24"
                          x2="9"
                          y2="24"
                          stroke="#EF4444"
                          strokeWidth="2"
                        />
                        <line
                          x1="39"
                          y1="24"
                          x2="48"
                          y2="24"
                          stroke="#EF4444"
                          strokeWidth="2"
                        />
                      </svg>
                    </div>
                  </>
                )}
              </>
            )}
          {activeCamera && activeFire && (
            <ProjectileCanvas
              onMouseClick={handleProjectileClick}
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

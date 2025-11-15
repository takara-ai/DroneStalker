"use client";

import {
  forwardRef,
  useEffect,
  useImperativeHandle,
  useRef,
  useState,
} from "react";
import type { NormalizedPosition } from "@/lib/drone-coordinates";

interface Projectile {
  id: number;
  startXPercent: number; // 0-100
  startYPercent: number; // 0-100
  targetXPercent: number; // 0-100
  targetYPercent: number; // 0-100
  startTime: number;
  duration: number; // milliseconds
  initialSize: number;
  hitDetected?: boolean; // Track if hit has been detected
}

interface ProjectileCanvasProps {
  onMouseClick: (xPercent: number, yPercent: number) => void;
  normalizedPosition: NormalizedPosition | null; // Drone bounding box (0-1 range)
}

export interface ProjectileCanvasHandle {
  addProjectile: (targetXPercent: number, targetYPercent: number) => void;
}

const ProjectileCanvas = forwardRef<
  ProjectileCanvasHandle,
  ProjectileCanvasProps
>(({ onMouseClick, normalizedPosition }, ref) => {
  const containerRef = useRef<HTMLDivElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const projectilesRef = useRef<Projectile[]>([]);
  const nextIdRef = useRef(0);
  const animationFrameRef = useRef<number | undefined>(undefined);
  const fireSoundRef = useRef<HTMLAudioElement | null>(null);
  const hitSoundRef = useRef<HTMLAudioElement | null>(null);
  const [dimensions, setDimensions] = useState({ width: 0, height: 0 });
  const [mousePosition, setMousePosition] = useState<{
    x: number;
    y: number;
  } | null>(null);

  // Initialize fire sound effect
  useEffect(() => {
    const fireSound = new Audio("/fire.wav");
    fireSound.volume = 0.3; // Adjust volume as needed
    fireSoundRef.current = fireSound;

    return () => {
      fireSound.pause();
      fireSound.src = "";
    };
  }, []);

  // Initialize hit sound effect
  useEffect(() => {
    const hitSound = new Audio("/hit.wav");
    hitSound.volume = 0.5; // Adjust volume as needed
    hitSoundRef.current = hitSound;

    return () => {
      hitSound.pause();
      hitSound.src = "";
    };
  }, []);

  // Calculate dimensions from container
  useEffect(() => {
    const updateDimensions = () => {
      if (containerRef.current) {
        const rect = containerRef.current.getBoundingClientRect();
        setDimensions({ width: rect.width, height: rect.height });
      }
    };

    updateDimensions();
    const resizeObserver = new ResizeObserver(updateDimensions);
    if (containerRef.current) {
      resizeObserver.observe(containerRef.current);
    }

    return () => {
      resizeObserver.disconnect();
    };
  }, []);

  // Helper function to calculate mouse position accounting for viewport transform
  const calculateMousePosition = (clientX: number, clientY: number) => {
    const viewport = document.getElementById("viewport");
    const canvas = canvasRef.current;

    if (viewport && canvas) {
      const viewportRect = viewport.getBoundingClientRect();
      const canvasRect = canvas.getBoundingClientRect();

      // Calculate the scale factor applied by the viewport
      // Viewport base size is 1920x1080
      const scale = viewportRect.width / 1920;

      // Get mouse position relative to viewport (in screen coordinates)
      const mouseX = clientX - viewportRect.left;
      const mouseY = clientY - viewportRect.top;

      // Convert to viewport coordinate space (1920x1080) by dividing by scale
      const viewportX = mouseX / scale;
      const viewportY = mouseY / scale;

      // Get canvas position within viewport (in viewport coordinates)
      const canvasInViewportX = (canvasRect.left - viewportRect.left) / scale;
      const canvasInViewportY = (canvasRect.top - viewportRect.top) / scale;

      // Calculate final coordinates relative to canvas
      const x = viewportX - canvasInViewportX;
      const y = viewportY - canvasInViewportY;

      return { x, y };
    }
    return null;
  };

  // Handle mouse movement for debug display
  const handleMouseMove = (e: React.MouseEvent<HTMLCanvasElement>) => {
    const pos = calculateMousePosition(e.clientX, e.clientY);
    if (pos) {
      setMousePosition(pos);
    }
  };

  // Handle mouse leave to clear position
  const handleMouseLeave = () => {
    setMousePosition(null);
  };

  // Handle mouse clicks
  const handleMouseDown = (e: React.MouseEvent<HTMLCanvasElement>) => {
    if (e.button === 0) {
      // Left click only
      const pos = calculateMousePosition(e.clientX, e.clientY);
      if (pos && dimensions.width > 0 && dimensions.height > 0) {
        // Convert to percentages (0-100)
        const xPercent = (pos.x / dimensions.width) * 100;
        const yPercent = (pos.y / dimensions.height) * 100;
        onMouseClick(xPercent, yPercent);
      }
    }
  };

  // Expose addProjectile via ref
  useImperativeHandle(
    ref,
    () => ({
      addProjectile: (targetXPercent: number, targetYPercent: number) => {
        // Play fire sound effect
        if (fireSoundRef.current) {
          // Clone and play to allow overlapping sounds
          const sound = fireSoundRef.current.cloneNode() as HTMLAudioElement;
          sound.volume = fireSoundRef.current.volume;
          sound.play().catch((error) => {
            console.warn("Could not play fire sound:", error);
          });
        }

        const startXPercent = 50; // Center horizontally
        const startYPercent = 98; // Near bottom (98%)
        const duration = 1500; // 1.5 seconds
        const initialSize = 12;

        const projectile: Projectile = {
          id: nextIdRef.current++,
          startXPercent,
          startYPercent,
          targetXPercent,
          targetYPercent,
          startTime: Date.now(),
          duration,
          initialSize,
        };

        projectilesRef.current.push(projectile);
      },
    }),
    []
  );

  // Animation loop
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas || dimensions.width === 0 || dimensions.height === 0) return;

    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    const animate = () => {
      // Clear canvas
      ctx.clearRect(0, 0, dimensions.width, dimensions.height);

      const now = Date.now();

      // Update and draw projectiles
      projectilesRef.current = projectilesRef.current.filter((proj) => {
        const elapsed = now - proj.startTime;
        const progress = Math.min(elapsed / proj.duration, 1);

        // Remove if expired
        if (progress >= 1) {
          return false;
        }

        // Convert percentages to pixels for rendering
        const startX = (proj.startXPercent / 100) * dimensions.width;
        const startY = (proj.startYPercent / 100) * dimensions.height;
        const targetX = (proj.targetXPercent / 100) * dimensions.width;
        const targetY = (proj.targetYPercent / 100) * dimensions.height;

        // Calculate position with parabolic trajectory
        // Using quadratic bezier for smooth curve
        const controlX = (startX + targetX) / 2;
        // Arc height as percentage of canvas height, then convert to pixels
        const arcHeightPercent = 10; // 10% of canvas height
        const controlY =
          Math.min(startY, targetY) -
          (arcHeightPercent / 100) * dimensions.height;

        const t = progress;
        const x =
          (1 - t) * (1 - t) * startX +
          2 * (1 - t) * t * controlX +
          t * t * targetX;
        const y =
          (1 - t) * (1 - t) * startY +
          2 * (1 - t) * t * controlY +
          t * t * targetY;

        // Check for hit when projectile is about to expire (progress >= 0.95)
        if (progress >= 0.95 && !proj.hitDetected && normalizedPosition) {
          // Convert projectile position to normalized coordinates (0-1 range)
          const projXNormalized = x / dimensions.width;
          const projYNormalized = y / dimensions.height;

          // Check if projectile is inside drone bounding box
          const droneLeft = normalizedPosition.x;
          const droneRight = normalizedPosition.x + normalizedPosition.width;
          const droneTop = normalizedPosition.y;
          const droneBottom = normalizedPosition.y + normalizedPosition.height;

          if (
            projXNormalized >= droneLeft &&
            projXNormalized <= droneRight &&
            projYNormalized >= droneTop &&
            projYNormalized <= droneBottom
          ) {
            // Hit detected! Play sound and mark as hit
            proj.hitDetected = true;
            if (hitSoundRef.current) {
              const sound = hitSoundRef.current.cloneNode() as HTMLAudioElement;
              sound.volume = hitSoundRef.current.volume;
              sound.play().catch((error) => {
                console.warn("Could not play hit sound:", error);
              });
            }
          }
        }

        // Calculate size (starts large, gets smaller)
        const size = proj.initialSize * (1 - progress * 0.9); // Shrinks to 30% of original

        // Draw projectile
        ctx.beginPath();
        ctx.arc(x, y, size, 0, Math.PI * 2);
        ctx.fillStyle = `rgba(255, 100, 100, ${1 - progress * 0.5})`; // Fades out
        ctx.fill();
        ctx.strokeStyle = `rgba(255, 200, 200, ${1 - progress * 0.5})`;
        ctx.lineWidth = 2;
        ctx.stroke();

        return true;
      });

      // Draw debug mouse position as percentages
      if (mousePosition && dimensions.width > 0 && dimensions.height > 0) {
        const percentX = (mousePosition.x / dimensions.width) * 100;
        const percentY = (mousePosition.y / dimensions.height) * 100;
        const text = `X: ${percentX.toFixed(1)}% Y: ${percentY.toFixed(1)}%`;

        // Draw background for text
        ctx.fillStyle = "rgba(0, 0, 0, 0.7)";
        ctx.fillRect(10, dimensions.height - 40, 190, 30);

        // Draw text
        ctx.fillStyle = "#00ff00";
        ctx.font = "16px monospace";
        ctx.fillText(text, 20, dimensions.height - 20);
      }

      animationFrameRef.current = requestAnimationFrame(animate);
    };

    animate();

    return () => {
      if (animationFrameRef.current) {
        cancelAnimationFrame(animationFrameRef.current);
      }
    };
  }, [dimensions.width, dimensions.height, mousePosition, normalizedPosition]);

  return (
    <div ref={containerRef} className="absolute inset-0">
      {dimensions.width > 0 && dimensions.height > 0 && (
        <canvas
          ref={canvasRef}
          width={dimensions.width}
          height={dimensions.height}
          onMouseDown={handleMouseDown}
          onMouseMove={handleMouseMove}
          onMouseLeave={handleMouseLeave}
          className="absolute size-full inset-0 cursor-crosshair pointer-events-auto"
          style={{ zIndex: 20 }}
        />
      )}
      <div className="top-2 right-2 bottom-2 w-5 border-4 border-border/50 absolute z-20">
        {/* drone health bar TODO */}
        <div className="w-full h-full bg-red-500/20"></div>
      </div>
    </div>
  );
});

ProjectileCanvas.displayName = "ProjectileCanvas";

export default ProjectileCanvas;

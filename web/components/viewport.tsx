"use client";

import { cn } from "@/lib/utils";
import { useEffect, useRef } from "react";

export function Viewport({
  className,
  children,
}: {
  className?: string;
  children: React.ReactNode;
}) {
  const containerRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    function resize() {
      const el = containerRef.current;
      if (!el) return;
      const width = window.innerWidth;
      const height = window.innerHeight;
      const scale = Math.min(width / 1920, height / 1080);

      el.style.transform = `scale(${scale})`;
      el.style.left = `${(width - 1920 * scale) / 2}px`;
      el.style.top = `${(height - 1080 * scale) / 2}px`;
    }
    window.addEventListener("resize", resize);
    resize();
    return () => window.removeEventListener("resize", resize);
  }, []);

  return (
    <main
      ref={containerRef}
      className={cn(
        "absolute w-[1920px] h-[1080px] origin-top-left p-2",
        className
      )}
      id="viewport"
    >
      <div className="w-full h-full relative crt">{children}</div>
    </main>
  );
}

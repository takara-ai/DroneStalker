"use client";

import React, { useState } from "react";

/**
 * ClickToEnter
 *
 * A wrapper that waits for user interaction to "enter" the app,
 * useful for unlocking audio autoplay in browsers.
 * Usage:
 * <ClickToEnter>
 *   {children}
 * </ClickToEnter>
 */
export const ClickToEnter: React.FC<{ children: React.ReactNode }> = ({
  children,
}) => {
  const [entered, setEntered] = useState(false);

  const handleEnter = () => {
    // Optionally: play a silent sound here to unlock audio context
    setEntered(true);
  };

  if (entered) {
    return <>{children}</>;
  }

  return (
    <div
      className="fixed inset-0 z-50 flex flex-col items-center justify-center bg-background"
      style={{
        // fallback for dark overlay
        background: "rgba(0, 0, 0, 0.96)",
        cursor: "pointer",
      }}
      tabIndex={0}
      onClick={handleEnter}
      onKeyDown={(e) => {
        if (e.key === "Enter" || e.key === " ") handleEnter();
      }}
      role="button"
      aria-label="Click to enter"
    >
      <div className="text-2xl font-mono mb-4 text-white">
        <span>Click to Enter</span>
      </div>
      <div className="text-white opacity-70 text-md font-mono">
        {`This app requires interaction to enable audio.`}
      </div>
    </div>
  );
};

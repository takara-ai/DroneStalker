"use client";

import { useStore } from "@/lib/store";

export default function Controls() {
  const {
    unlockedCamera,
    unlockedMotion,
    unlockedFire,
    unlockedTracking,
    unlockedMotionPrediction,
    activeCamera,
    activeMotionDetection,
    activeFire,
    activeTracking,
    activeAutoFire,
    activeMotionPrediction,
    activeLockTarget,
    setActiveCamera,
    setActiveMotionDetection,
    setActiveFire,
    setActiveTracking,
    setActiveAutoFire,
    setActiveMotionPrediction,
    setActiveLockTarget,
  } = useStore();

  return (
    <div
      className="border-4 grid grid-flow-col grid-cols-2 grid-rows-3 p-6 closed"
      id="controls"
    >
      <ToggleButton
        id="camera-feed"
        label="camera feed"
        value={activeCamera}
        onChange={setActiveCamera}
        locked={!unlockedCamera}
      />
      <ToggleButton
        id="left-click-to-fire"
        label="left click to fire"
        value={activeFire}
        onChange={setActiveFire}
        locked={!unlockedFire}
      />
      <ToggleButton
        id="motion-detection"
        label="motion detection"
        value={activeMotionDetection}
        onChange={setActiveMotionDetection}
        locked={!unlockedMotion}
      />
      <ToggleButton
        id="tracking"
        label="tracking"
        value={activeTracking}
        onChange={setActiveTracking}
        locked={!unlockedTracking}
      />
      <ToggleButton
        id="lock-target"
        label="lock target"
        value={activeLockTarget}
        onChange={setActiveLockTarget}
        locked={!unlockedTracking}
      />
      <ToggleButton
        id="motion-prediction"
        label="motion prediction"
        value={activeMotionPrediction}
        onChange={setActiveMotionPrediction}
        locked={!unlockedMotionPrediction}
      />
      <ToggleButton
        id="auto-fire"
        label="auto fire"
        value={activeAutoFire}
        onChange={setActiveAutoFire}
        locked={!unlockedMotionPrediction}
      />
    </div>
  );
}

function ToggleButton({
  id,
  label,
  value,
  onChange,
  locked,
}: {
  id: string;
  label: string;
  value: boolean;
  onChange?: (value: boolean) => void;
  locked: boolean;
}) {
  return (
    <button
      id={id}
      className="flex items-center font-semibold text-lg justify-start px-2 gap-2 uppercase cursor-pointer disabled:cursor-not-allowed disabled:text-muted-foreground hover:bg-foreground/10 active:text-white"
      onClick={() => onChange?.(!value)}
      disabled={locked}
    >
      <span className="whitespace-pre">{value ? "[ x ]" : "[   ]"}</span>
      <span>{locked ? "???" : label}</span>
      {locked && <span className="text-destructive">locked</span>}
    </button>
  );
}

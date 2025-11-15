"use client";

import { useStore } from "@/lib/store";

export default function Controls() {
  const {
    unlockedCamera,
    unlockedMotion,
    unlockedFire,
    unlockedTracking,
    unlockedAutoFire,
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
      className="border-4 grid grid-flow-col grid-cols-2 p-6 closed"
      id="controls"
    >
      <ToggleButton
        label="camera feed"
        value={activeCamera}
        onChange={setActiveCamera}
        locked={!unlockedCamera}
      />
      <ToggleButton
        label="left click to fire"
        value={activeFire}
        onChange={setActiveFire}
        locked={!unlockedFire}
      />
      <ToggleButton
        label="motion detection"
        value={activeMotionDetection}
        onChange={setActiveMotionDetection}
        locked={!unlockedMotion}
      />
      <ToggleButton
        label="tracking"
        value={activeTracking}
        onChange={setActiveTracking}
        locked={!unlockedTracking}
      />

      <ToggleButton
        label="motion prediction"
        value={activeMotionPrediction}
        onChange={setActiveMotionPrediction}
        locked={!unlockedMotionPrediction}
      />
      <ToggleButton
        label="lock target"
        value={activeLockTarget}
        onChange={setActiveLockTarget}
        locked={!unlockedMotionPrediction}
      />
      <ToggleButton
        label="auto fire"
        value={activeAutoFire}
        onChange={setActiveAutoFire}
        locked={!unlockedAutoFire}
      />
    </div>
  );
}

function ToggleButton({
  label,
  value,
  onChange,
  locked,
}: {
  label: string;
  value: boolean;
  onChange?: (value: boolean) => void;
  locked: boolean;
}) {
  return (
    <button
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

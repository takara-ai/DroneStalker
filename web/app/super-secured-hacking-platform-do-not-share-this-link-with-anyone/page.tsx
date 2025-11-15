"use client";

import { Button } from "@/components/ui/button";
import { openPanel } from "@/lib/open-animation";
import { useStore } from "@/lib/store";
import { cn } from "@/lib/utils";
import { useEffect } from "react";
import CommanderVideo from "./commander-video";
import Controls from "./controls";
import Code from "./main-panel/code";
import Credits from "./main-panel/credits";
import VideoFeed from "./main-panel/video-feed";
import TextingChat from "./texting-chat";
import You from "./you";
import { codes } from "./main-panel/codes";

export default function Page() {
  const tab = useStore((state) => state.tab);
  const setTab = useStore((state) => state.setTab);
  const unlockedCredits = useStore((state) => state.unlockedCredits);
  const setScenarioState = useStore((state) => state.setScenarioState);
  const setUnlockedMotion = useStore((state) => state.setUnlockedMotion);
  const setUnlockedTracking = useStore((state) => state.setUnlockedTracking);
  const setUnlockedMotionPrediction = useStore(
    (state) => state.setUnlockedMotionPrediction
  );
  const triggerAction = useStore((state) => state.triggerAction);
  const codeStates = useStore((state) => state.codeStates);
  const unlockAll = useStore((state) => state.unlockAll);

  useEffect(() => {
    function handleKeyDown(e: KeyboardEvent) {
      // Check if the '=' key is pressed (both '=' and '+')
      if (e.key === "=") {
        unlockAll?.();
      }
    }

    window.addEventListener("keydown", handleKeyDown);
    return () => {
      window.removeEventListener("keydown", handleKeyDown);
    };
  }, [unlockAll]);

  // Note: State progression logic is handled in store setters (setTab, setActiveMotionDetection, setActiveTracking)
  // State transitions triggered by AI tool calls are handled in texting-chat.tsx
  // to prevent loops where tool calls → state changes → AI messages → more tool calls

  // Note: positionPredictionComplete → success transition is handled in texting-chat.tsx
  // when AI unlocks credits to prevent loops

  // Handle code completion
  const handleCodeComplete = (
    codeKey: "motionDetection" | "tracking" | "positionPrediction"
  ) => {
    // Get scenarioState from the zustand store
    const scenarioState = useStore.getState().scenarioState;

    if (codeKey === "motionDetection" && scenarioState === "firstCode") {
      // Motion detection code completed - automatically unlock motion detection
      setUnlockedMotion(true);
      setScenarioState("motionDetection");
      triggerAction("completed motion detection code");
    } else if (codeKey === "tracking" && scenarioState === "tracking") {
      // Tracking code completed - automatically unlock tracking
      setUnlockedTracking(true);
      setScenarioState("trackingComplete");
      triggerAction("completed tracking code");
    } else if (
      codeKey === "positionPrediction" &&
      scenarioState === "positionPrediction"
    ) {
      // Position prediction code completed - automatically unlock motion prediction
      setUnlockedMotionPrediction(true);
      setScenarioState("positionPredictionComplete");
      triggerAction("completed position prediction code");
    }
  };

  useEffect(() => {
    // Find all elements with the class 'closed' and remove it one by one every 2s
    const elements = Array.from(
      document.querySelectorAll<HTMLElement>(".closed")
    );
    let idx = 0;

    function openNext() {
      if (idx < elements.length) {
        openPanel(elements[idx]);
        idx++;
        if (idx < elements.length) {
          setTimeout(openNext, 500);
        }
      }
    }

    if (elements.length) {
      openNext();
    }
    // Optionally, return cleanup if components could be remounted/reopened, but not strictly needed here.
  }, []);

  return (
    <div className="grid grid-cols-4 grid-rows-4 gap-2 h-full">
      <div
        className="col-span-3 row-span-3 gap-2 flex flex-col relative closed"
        id="main-panel"
      >
        <nav className="flex gap-2 ">
          <Button
            onClick={() => setTab("video")}
            className={cn(
              "uppercase",
              tab === "video" && "bg-foreground text-background"
            )}
          >
            Camera view
          </Button>
          <Button
            onClick={() => setTab("motionDetection")}
            className={cn(
              "uppercase",
              tab === "motionDetection" && "bg-foreground text-background"
            )}
            disabled={!codeStates.motionDetection.unlocked}
          >
            {codeStates.motionDetection.unlocked ? "motion.py" : "???"}
            {!codeStates.motionDetection.unlocked && (
              <span className="text-destructive">locked</span>
            )}
          </Button>
          <Button
            onClick={() => setTab("tracking")}
            className={cn(
              "uppercase",
              tab === "tracking" && "bg-foreground text-background"
            )}
            disabled={!codeStates.tracking.unlocked}
          >
            {codeStates.tracking.unlocked ? "tracking.py" : "???"}
            {!codeStates.tracking.unlocked && (
              <span className="text-destructive">locked</span>
            )}
          </Button>
          <Button
            onClick={() => setTab("positionPrediction")}
            className={cn(
              "uppercase",
              tab === "positionPrediction" && "bg-foreground text-background"
            )}
            disabled={!codeStates.positionPrediction.unlocked}
          >
            {codeStates.positionPrediction.unlocked ? "prediction.py" : "???"}
            {!codeStates.positionPrediction.unlocked && (
              <span className="text-destructive">locked</span>
            )}
          </Button>
          <Button
            onClick={() => setTab("credits")}
            className={cn(
              "uppercase",
              tab === "credits" && "bg-foreground text-background"
            )}
            disabled={!unlockedCredits}
          >
            {unlockedCredits ? "Credits" : "???"}
            {!unlockedCredits && (
              <span className="text-destructive">locked</span>
            )}
          </Button>
        </nav>
        {tab === "video" && <VideoFeed />}
        {tab === "motionDetection" && (
          <Code
            codeKey="motionDetection"
            code={codes.motionDetection.code}
            source={codes.motionDetection.source}
            onComplete={() => handleCodeComplete("motionDetection")}
          />
        )}
        {tab === "tracking" && (
          <Code
            codeKey="tracking"
            code={codes.tracking.code}
            source={codes.tracking.source}
            onComplete={() => handleCodeComplete("tracking")}
          />
        )}
        {tab === "positionPrediction" && (
          <Code
            codeKey="positionPrediction"
            code={codes.positionPrediction.code}
            source={codes.positionPrediction.source}
            onComplete={() => handleCodeComplete("positionPrediction")}
          />
        )}
        {tab === "credits" && <Credits />}
      </div>
      <div className="col-start-4 row-start-1 flex flex-col *:h-full">
        <CommanderVideo />
      </div>
      <div className="col-start-4 row-start-2 row-span-2 flex flex-col *:h-full">
        <TextingChat />
      </div>
      <div className="col-start-4 row-start-4 flex flex-col *:h-full">
        <You />
      </div>
      <div className="col-span-3 row-start-4 flex flex-col *:h-full">
        <Controls />
      </div>
    </div>
  );
}

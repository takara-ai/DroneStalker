"use client";

import { Button } from "@/components/ui/button";
import { openPanel } from "@/lib/open-animation";
import { useStore } from "@/lib/store";
import { cn } from "@/lib/utils";
import { useEffect, useMemo } from "react";
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
  const unlockedCode = useStore((state) => state.unlockedCode);
  const unlockedCredits = useStore((state) => state.unlockedCredits);
  const scenarioState = useStore((state) => state.scenarioState);
  const setScenarioState = useStore((state) => state.setScenarioState);
  const unlockedFire = useStore((state) => state.unlockedFire);
  const activeMotionDetection = useStore(
    (state) => state.activeMotionDetection
  );
  const activeTracking = useStore((state) => state.activeTracking);

  // Scenario progression logic
  useEffect(() => {
    // mission → fireUnlocked: When fire is unlocked (AI unlocks it)
    if (scenarioState === "mission" && unlockedFire) {
      setScenarioState("fireUnlocked");
    }
  }, [scenarioState, unlockedFire, setScenarioState]);

  useEffect(() => {
    // fireUnlocked → codeUnlocked: When code tab is unlocked (AI unlocks it after user tries to fire)
    if (scenarioState === "fireUnlocked" && unlockedCode) {
      setScenarioState("codeUnlocked");
    }
  }, [scenarioState, unlockedCode, setScenarioState]);

  useEffect(() => {
    // codeUnlocked → firstCode: When Code tab is opened
    if (scenarioState === "codeUnlocked" && tab === "code" && unlockedCode) {
      setScenarioState("firstCode");
    }
  }, [scenarioState, tab, unlockedCode, setScenarioState]);

  useEffect(() => {
    // motionDetection → tracking: When motion detection is toggled on
    if (scenarioState === "motionDetection" && activeMotionDetection) {
      setScenarioState("tracking");
    }
  }, [scenarioState, activeMotionDetection, setScenarioState]);

  useEffect(() => {
    // trackingComplete → positionPrediction: When tracking is enabled
    if (scenarioState === "trackingComplete" && activeTracking) {
      setScenarioState("positionPrediction");
    }
  }, [scenarioState, activeTracking, setScenarioState]);

  useEffect(() => {
    // positionPredictionComplete → success: When credits are unlocked (drone successfully intercepted)
    if (scenarioState === "positionPredictionComplete" && unlockedCredits) {
      setScenarioState("success");
    }
  }, [scenarioState, unlockedCredits, setScenarioState]);

  // Determine which code to show based on scenario state
  const currentCode = useMemo(() => {
    if (scenarioState === "firstCode") {
      return codes.motionDetection;
    } else if (scenarioState === "tracking") {
      return codes.tracking;
    } else if (scenarioState === "positionPrediction") {
      return codes.positionPrediction;
    }
    // Default to motion detection code
    return codes.motionDetection;
  }, [scenarioState]);

  // Handle code completion
  const handleCodeComplete = () => {
    if (scenarioState === "firstCode") {
      // Motion detection code completed
      setScenarioState("motionDetection");
    } else if (scenarioState === "tracking") {
      // Tracking code completed
      setScenarioState("trackingComplete");
    } else if (scenarioState === "positionPrediction") {
      // Position prediction code completed
      setScenarioState("positionPredictionComplete");
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
            onClick={() => setTab("code")}
            className={cn(
              "uppercase",
              tab === "code" && "bg-foreground text-background"
            )}
            disabled={!unlockedCode}
          >
            {unlockedCode ? "Code" : "???"}
            {!unlockedCode && <span className="text-destructive">locked</span>}
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
        {tab === "code" && (
          <Code
            code={currentCode.code}
            source={currentCode.source}
            onComplete={handleCodeComplete}
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

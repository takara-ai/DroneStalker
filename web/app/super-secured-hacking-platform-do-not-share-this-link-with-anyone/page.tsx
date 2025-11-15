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

export default function Page() {
  const tab = useStore((state) => state.tab);
  const setTab = useStore((state) => state.setTab);
  const unlockedCode = useStore((state) => state.unlockedCode);
  const unlockedCredits = useStore((state) => state.unlockedCredits);

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
        {tab === "code" && <Code />}
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

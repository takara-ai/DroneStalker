"use client";

import VideoFeed from "./main-panel/video-feed";
import CommanderVideo from "./commander-video";
import TextingChat from "./texting-chat";
import You from "./you";
import Controls from "./controls";
import { Button } from "@/components/ui/button";
import { useStore } from "@/lib/store";

import Credits from "./main-panel/credits";
import Code from "./main-panel/code";
import { cn } from "@/lib/utils";

export default function Page() {
  const tab = useStore((state) => state.tab);
  const setTab = useStore((state) => state.setTab);
  const unlockedCode = useStore((state) => state.unlockedCode);
  const unlockedCredits = useStore((state) => state.unlockedCredits);
  return (
    <div className="grid grid-cols-4 grid-rows-4 gap-2 h-full">
      <div className="col-span-3 row-span-3 gap-2 flex flex-col relative">
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

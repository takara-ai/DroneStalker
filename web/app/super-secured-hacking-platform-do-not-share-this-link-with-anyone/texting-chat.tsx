"use client";

import { useChat } from "@ai-sdk/react";
import { DefaultChatTransport } from "ai";
import { useEffect, useState } from "react";
import { Input } from "@/components/ui/input";
import { useStore } from "@/lib/store";
import { cn } from "@/lib/utils";

export default function TextingChat() {
  const [input, setInput] = useState("");
  const {
    setUnlockedMotion,
    setUnlockedFire,
    setUnlockedTracking,
    setUnlockedAutoFire,
    setUnlockedMotionPrediction,
    setUnlockedCode,
    setUnlockedCredits,
    unlockedMotion,
    unlockedFire,
    unlockedTracking,
    unlockedAutoFire,
    unlockedMotionPrediction,
    unlockedCode,
    unlockedCredits,
  } = useStore();

  const { messages, sendMessage, status } = useChat({
    transport: new DefaultChatTransport({ api: "/api/chat" }),
    onError: (error) => {
      console.error("Chat error:", error);
    },
    onToolCall: (toolCall) => {
      const toolName = toolCall.toolCall.toolName;
      console.log("Tool call:", toolName);

      switch (toolName) {
        case "unlockMotion":
          setUnlockedMotion(true);
          break;
        case "unlockFire":
          setUnlockedFire(true);
          break;
        case "unlockTracking":
          setUnlockedTracking(true);
          break;
        case "unlockAutoFire":
          setUnlockedAutoFire(true);
          break;
        case "unlockMotionPrediction":
          setUnlockedMotionPrediction(true);
          break;
        case "unlockCode":
          setUnlockedCode(true);
          break;
        case "unlockCredits":
          setUnlockedCredits(true);
          break;
        default:
          console.warn("Unknown tool call:", toolName);
      }
    },
  });

  // Unlock everything when the '=' key is pressed
  useEffect(() => {
    function handleKeyDown(e: KeyboardEvent) {
      if (e.key === "=") {
        setUnlockedMotion(true);
        setUnlockedFire(true);
        setUnlockedTracking(true);
        setUnlockedAutoFire(true);
        setUnlockedMotionPrediction(true);
        setUnlockedCode(true);
        setUnlockedCredits(true);
      }
    }
    window.addEventListener("keydown", handleKeyDown);
    return () => {
      window.removeEventListener("keydown", handleKeyDown);
    };
  }, [
    setUnlockedMotion,
    setUnlockedFire,
    setUnlockedTracking,
    setUnlockedAutoFire,
    setUnlockedMotionPrediction,
    setUnlockedCode,
    setUnlockedCredits,
  ]);

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();

    // Check if user said "MIRROR" to unlock everything
    if (input.trim() === "MIRROR") {
      setUnlockedMotion(true);
      setUnlockedFire(true);
      setUnlockedTracking(true);
      setUnlockedAutoFire(true);
      setUnlockedMotionPrediction(true);
      setUnlockedCode(true);
      setUnlockedCredits(true);
    }

    sendMessage({
      text: input,
      metadata: {
        currentUnlocks: {
          motion: unlockedMotion,
          fire: unlockedFire,
          tracking: unlockedTracking,
          autoFire: unlockedAutoFire,
          motionPrediction: unlockedMotionPrediction,
          code: unlockedCode,
          credits: unlockedCredits,
        },
      },
    });
    setInput("");
  };

  return (
    <div className="border-4 p-2 flex flex-col gap-2 closed" id="chat">
      <div className="overflow-y-auto flex-1">
        {messages.map((m) => (
          <div
            key={m.id}
            className={cn(
              m.role === "assistant"
                ? "text-foreground"
                : "text-muted-foreground"
            )}
          >
            {m.role === "user" ? "YOU" : "GENERAL"}:{" "}
            {m.parts
              .filter((part) => part.type === "text")
              .map((part, i) => (
                <span key={i}>{part.text}</span>
              ))}
          </div>
        ))}
        {(status === "submitted" || status === "streaming") && <div>...</div>}
        {status === "error" && (
          <div className="text-destructive">Error /!\ Check console /!\</div>
        )}
      </div>
      <form onSubmit={handleSubmit} className="w-full mt-auto">
        <Input
          type="text"
          value={input}
          onChange={(e) => setInput(e.target.value)}
          placeholder="Message..."
        />
      </form>
    </div>
  );
}

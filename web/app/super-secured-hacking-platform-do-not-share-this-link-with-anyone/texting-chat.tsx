"use client";

import { Tts } from "@/components/tts";
import { Input } from "@/components/ui/input";
import { getStoreState, useStore } from "@/lib/store";
import { cn } from "@/lib/utils";
import { useChat } from "@ai-sdk/react";
import { DefaultChatTransport } from "ai";
import { useEffect, useRef, useState } from "react";

export default function TextingChat() {
  const [input, setInput] = useState("");
  const {
    setUnlockedFire,
    setUnlockedAutoFire,
    setUnlockedCode,
    setUnlockedCredits,
    unlockedMotion,
    unlockedFire,
    unlockedTracking,
    unlockedAutoFire,
    unlockedMotionPrediction,
    unlockedCode,
    unlockedCredits,
    scenarioState,
    addToTtsQueue,
    setSendMessageCallback,
    unlockAll,
  } = useStore();
  const scrollRef = useRef<HTMLDivElement>(null);
  const initialMessageSentRef = useRef(false);

  const { messages, sendMessage, status } = useChat({
    transport: new DefaultChatTransport({ api: "/api/chat" }),
    onError: (error) => {
      console.error("Chat error:", error);
    },
    onFinish: () => {
      // Camera is always unlocked, no action needed
    },
    onToolCall: async (toolCall) => {
      const toolName = toolCall.toolCall.toolName;
      console.log("Tool call:", toolName);

      switch (toolName) {
        case "unlockFire":
          setUnlockedFire(true);
          break;
        case "unlockAutoFire":
          setUnlockedAutoFire(true);
          break;
        case "unlockCode":
          setUnlockedCode(true);
          // State transition happens in setUnlockedCode setter
          break;
        case "unlockCredits":
          setUnlockedCredits(true);
          // State transition happens in setUnlockedCredits setter
          break;
        default:
          console.warn("Unknown tool call:", toolName);
      }
    },
  });

  useEffect(() => {
    if (scrollRef.current) {
      scrollRef.current.scrollTop = scrollRef.current.scrollHeight;
    }
  }, [messages]);
  // Track which messages have been added to TTS queue
  const processedMessageIdsRef = useRef<Set<string>>(new Set());

  // Register sendMessage callback with store
  useEffect(() => {
    const actionHandler = (action: string) => {
      // Get fresh state values
      if (status === "ready") {
        const currentState = getStoreState();
        sendMessage({
          text: `*${action}*`,
          metadata: {
            scenarioState: currentState.scenarioState,
            currentUnlocks: {
              motion: currentState.unlockedMotion,
              fire: currentState.unlockedFire,
              tracking: currentState.unlockedTracking,
              autoFire: currentState.unlockedAutoFire,
              motionPrediction: currentState.unlockedMotionPrediction,
              code: currentState.unlockedCode,
              credits: currentState.unlockedCredits,
            },
          },
        });
      }
    };

    setSendMessageCallback(actionHandler);

    return () => {
      setSendMessageCallback(null);
    };
  }, [setSendMessageCallback, sendMessage, status]);

  // Initialize scenario: send initial message to Commander on mount
  useEffect(() => {
    if (
      !initialMessageSentRef.current &&
      messages.length === 0 &&
      scenarioState === "intro" &&
      status === "ready"
    ) {
      initialMessageSentRef.current = true;
      setTimeout(() => {
        sendMessage({
          text: "Ready for mission briefing.",
          metadata: {
            scenarioState,
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
      }, 2000);
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [messages.length, scenarioState, status]); // Include dependencies to check conditions

  // Add new commander messages to TTS queue (only when complete)
  useEffect(() => {
    // Only process when not streaming to avoid queuing partial messages
    if (status !== "ready") {
      return;
    }

    messages.forEach((message) => {
      // Only process assistant (commander) messages
      if (message.role === "assistant") {
        // Skip if we've already processed this message
        if (processedMessageIdsRef.current.has(message.id)) {
          return;
        }

        // Extract text from message parts
        const textParts = message.parts
          .filter((part) => part.type === "text")
          .map((part) => part.text);

        if (textParts.length > 0) {
          const fullText = textParts.join("");
          // Only add non-empty messages (skip the "*action taken*" type messages)
          if (fullText.trim() && !fullText.trim().startsWith("*")) {
            addToTtsQueue(fullText);
            processedMessageIdsRef.current.add(message.id);
          }
        }
      }
    });
  }, [messages, addToTtsQueue, status]);

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();

    // Don't allow sending if status is not ready
    if (status !== "ready") {
      return;
    }

    // Check if user said "MIRROR" to unlock everything
    if (input.trim() === "MIRROR") {
      unlockAll?.();
    }

    sendMessage({
      text: input,
      metadata: {
        scenarioState,
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
      <Tts />
      <div
        ref={scrollRef}
        className="overflow-y-auto flex-1 flex flex-col gap-2 text-sm"
      >
        {messages.map((m) => (
          <div
            key={m.id}
            className={cn(
              m.role === "assistant"
                ? "text-foreground"
                : "text-muted-foreground"
            )}
          >
            {m.role === "user" ? "YOU: " : ""}
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
          disabled={status !== "ready"}
        />
      </form>
    </div>
  );
}

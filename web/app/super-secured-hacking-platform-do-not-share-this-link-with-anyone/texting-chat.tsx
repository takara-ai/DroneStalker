"use client";

import { useChat } from "@ai-sdk/react";
import { DefaultChatTransport } from "ai";
import { useEffect, useRef, useState } from "react";
import { Input } from "@/components/ui/input";
import { useStore } from "@/lib/store";
import { cn } from "@/lib/utils";
import { Tts } from "@/components/tts";

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
    scenarioState,
    addToTtsQueue,
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

  // Track previous scenario state to detect changes
  const prevScenarioStateRef = useRef<string>(scenarioState);
  // Track which messages have been added to TTS queue
  const processedMessageIdsRef = useRef<Set<string>>(new Set());
  // Track last processed message to detect when streaming completes
  const lastMessageRef = useRef<string>("");

  // Initialize scenario: send initial message to Commander on mount
  useEffect(() => {
    if (messages.length === 0 && scenarioState === "intro") {
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
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []); // Only run on mount

  // Add new commander messages to TTS queue (only when message is complete)
  useEffect(() => {
    // Only process when not streaming to avoid partial messages
    if (status === "streaming") {
      return;
    }

    // Get the last assistant message
    const assistantMessages = messages.filter((m) => m.role === "assistant");
    if (assistantMessages.length === 0) return;

    const lastMessage = assistantMessages[assistantMessages.length - 1];
    const messageId = lastMessage.id;

    // Skip if already processed
    if (processedMessageIdsRef.current.has(messageId)) {
      return;
    }

    // Extract text from message parts
    const textParts = lastMessage.parts
      .filter((part) => part.type === "text")
      .map((part) => part.text);

    if (textParts.length === 0) return;

    const fullText = textParts.join("");

    // Skip empty or action messages
    if (!fullText.trim() || fullText.trim().startsWith("*")) {
      return;
    }

    // Add to queue only when message is complete
    addToTtsQueue(fullText);
    processedMessageIdsRef.current.add(messageId);
    lastMessageRef.current = fullText;
  }, [messages, addToTtsQueue, status]);

  // Automatically send a message when scenario state changes (to trigger AI response)
  useEffect(() => {
    // Skip if this is the initial state or if we haven't started the conversation yet
    if (
      messages.length === 0 ||
      prevScenarioStateRef.current === scenarioState
    ) {
      prevScenarioStateRef.current = scenarioState;
      return;
    }

    // Only trigger for specific state transitions that need AI response
    const shouldTriggerAI = [
      "mission", // Camera activated
      "fireUnlocked", // Fire unlocked
      "codeUnlocked", // Code tab unlocked
      "firstCode", // Code tab opened
      "motionDetection", // Motion detection code completed
      "tracking", // Motion detection enabled
      "trackingComplete", // Tracking code completed
      "positionPrediction", // Tracking enabled
      "positionPredictionComplete", // Position prediction code completed
      "success", // Credits unlocked
    ].includes(scenarioState);

    if (shouldTriggerAI && prevScenarioStateRef.current !== scenarioState) {
      // Send a silent message to trigger AI response with new scenario context
      sendMessage({
        text: "*action taken*",
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
      prevScenarioStateRef.current = scenarioState;
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [scenarioState, sendMessage]);

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
      <div className="overflow-y-auto flex-1">
        {(() => {
          // Deduplicate messages by ID, keeping the last occurrence
          const seen = new Set<string>();
          const uniqueMessages: typeof messages = [];

          // Process in reverse to keep the last occurrence of each ID
          for (let i = messages.length - 1; i >= 0; i--) {
            const msg = messages[i];
            if (!seen.has(msg.id)) {
              seen.add(msg.id);
              uniqueMessages.unshift(msg);
            }
          }

          return uniqueMessages.map((m, index) => (
            <div
              key={`${m.id}-${index}`}
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
                  <span key={`${m.id}-part-${i}`}>{part.text}</span>
                ))}
            </div>
          ));
        })()}
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

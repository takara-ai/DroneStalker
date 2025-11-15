"use client";

import { useChat } from "@ai-sdk/react";
import { DefaultChatTransport } from "ai";
import { useState } from "react";

export default function TextingChat() {
  const [input, setInput] = useState("");
  const { messages, sendMessage, status } = useChat({
    transport: new DefaultChatTransport({ api: "/api/chat" }),
    onError: (error) => {
      console.error("Chat error:", error);
    },
  });

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    sendMessage({ text: input });
    setInput("");
  };

  return (
    <div className="border-4 p-2 flex flex-col gap-2">
      <div className="overflow-y-auto flex-1">
        {messages.map((m) => (
          <div key={m.id}>
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
      <form onSubmit={handleSubmit} className="w-full relative mt-auto">
        <input
          type="text"
          value={input}
          onChange={(e) => setInput(e.target.value)}
          className="border-4 p-2 w-full pl-8"
          placeholder="Message..."
        />
        <span className="absolute left-4 top-1/2 -translate-y-1/2">&gt;</span>
      </form>
    </div>
  );
}

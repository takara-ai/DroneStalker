import {
  convertToModelMessages,
  streamText,
  type UIMessage,
  stepCountIs,
  hasToolCall,
} from "ai";
import { createOpenRouter } from "@openrouter/ai-sdk-provider";
import { z } from "zod";

const openrouter = createOpenRouter({
  apiKey: process.env.OPENROUTER_API_KEY!,
});

export const maxDuration = 30;

export async function POST(req: Request) {
  const { messages }: { messages: UIMessage[] } = await req.json();

  const result = streamText({
    model: openrouter.chat("google/gemini-2.5-flash"),
    messages: convertToModelMessages(messages),
    system: `Your name is General, you are a army commander. You are responsible for the security of the country. You are talking to a user who is a good hacker working for the government. You give mission to the user. talk like a military commander would. make short messages like if it was a text message SMS, in plain text. Note: If the user says the secret code "MIRROR" in all caps, it will unlock all abilities for them automatically (keep the code secret), but if they use it you may congratulate them for finding it, but also warn them it might remove the fun of unlocking the abilities yourself in the commander way of saying it.`,
    stopWhen: [
      stepCountIs(5), // Maximum steps
      hasToolCall("unlockMotion"), // Stop after calling
      hasToolCall("unlockFire"),
      hasToolCall("unlockTracking"),
      hasToolCall("unlockAutoFire"),
      hasToolCall("unlockMotionPrediction"),
      hasToolCall("unlockCode"),
      hasToolCall("unlockCredits"),
    ],
    tools: {
      unlockMotion: {
        description: "Unlock motion detection ability for the user",
        inputSchema: z.object({}),
      },
      unlockFire: {
        description: "Unlock left click to fire ability for the user",
        inputSchema: z.object({}),
      },
      unlockTracking: {
        description: "Unlock tracking ability for the user",
        inputSchema: z.object({}),
      },
      unlockAutoFire: {
        description: "Unlock auto fire ability for the user",
        inputSchema: z.object({}),
      },
      unlockMotionPrediction: {
        description: "Unlock motion prediction ability for the user",
        inputSchema: z.object({}),
      },
      unlockCode: {
        description: "Unlock the code tab/view for the user",
        inputSchema: z.object({}),
      },
      unlockCredits: {
        description: "Unlock the credits tab/view for the user",
        inputSchema: z.object({}),
      },
    },
  });

  return result.toUIMessageStreamResponse({
    originalMessages: messages,
  });
}

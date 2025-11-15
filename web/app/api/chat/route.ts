import {
  convertToModelMessages,
  streamText,
  type UIMessage,
  stepCountIs,
} from "ai";
import { createOpenRouter } from "@openrouter/ai-sdk-provider";
import { z } from "zod";
import { scenarioStates } from "@/lib/store";

const openrouter = createOpenRouter({
  apiKey: process.env.OPENROUTER_API_KEY!,
});

export const maxDuration = 30;

// Define tool names as a type
type ToolName =
  | "unlockCamera"
  | "unlockFire"
  | "unlockCode"
  | "unlockCredits"
  | "unlockAutoFire";

// Define which tools are available at each scenario state
function getActiveToolsForScenario(
  scenarioState: string
): ToolName[] | undefined {
  switch (scenarioState) {
    case "intro":
      // Can unlock camera feed
      return ["unlockCamera"];
    case "mission":
      // Can unlock fire ability
      return ["unlockFire"];
    case "fireUnlocked":
      // Can unlock code tab after user tries to fire
      return ["unlockCode"];
    case "codeUnlocked":
      // No tools - user should open code tab
      return undefined;
    case "firstCode":
      // No tools - user is writing motion detection code
      return undefined;
    case "motionDetection":
      // No tools - motion detection is automatically unlocked after code completion
      return undefined;
    case "tracking":
      // No tools - user is writing tracking code
      return undefined;
    case "trackingComplete":
      // No tools - tracking is automatically unlocked after code completion
      return undefined;
    case "positionPrediction":
      // No tools - user is writing position prediction code
      return undefined;
    case "positionPredictionComplete":
      // No tools - motion prediction is automatically unlocked after code completion
      return undefined;
    case "success":
      // Can unlock credits tab
      return ["unlockCredits"];
    default:
      // Default: no tools available
      return undefined;
  }
}

export async function POST(req: Request) {
  const { messages }: { messages: UIMessage[] } = await req.json();

  // Extract scenarioState from the last user message's metadata
  const lastMessage = messages[messages.length - 1];
  const scenarioState =
    (lastMessage?.metadata as { scenarioState?: string })?.scenarioState ||
    "intro";

  // Get the scenario-specific instruction
  const scenarioInstruction =
    scenarioStates[scenarioState as keyof typeof scenarioStates] ?? null;

  if (!scenarioInstruction) {
    return new Response("No scenario instruction found", { status: 400 });
  }

  // Get active tools for current scenario state
  const activeTools = getActiveToolsForScenario(scenarioState);

  // Build dynamic system prompt
  const baseSystemPrompt = `Your name is General, you are a army commander. You are responsible for the security of the country. You are talking to a user who is a good hacker working for the government. You give mission to the user. talk like a military commander would. make short messages like if it was a text message SMS, in plain text.

CRITICAL: You have access to tools that unlock features for the user. YOU must use these tools to unlock features - the user CANNOT unlock features themselves. When a feature needs to be unlocked, YOU must call the appropriate unlock tool (unlockFire, unlockCode, unlockMotion, etc.). The user can only toggle features on/off AFTER you have unlocked them. Never tell the user to "unlock" something - you must do it yourself using the available tools.

Note: If the user says the secret code "MIRROR" in all caps, it will unlock all abilities for them automatically (keep the code secret), but if they use it you may congratulate them for finding it, but also warn them it might remove the fun of unlocking the abilities yourself in the commander way of saying it.`;

  const systemPrompt = `${baseSystemPrompt}\n\nCURRENT SCENARIO STAGE: ${scenarioInstruction}`;

  const result = streamText({
    model: openrouter.chat("google/gemini-2.5-flash"),
    messages: convertToModelMessages(messages),
    system: systemPrompt,
    stopWhen: [
      stepCountIs(5), // Maximum steps
    ],
    tools: {
      unlockCamera: {
        description: "Unlock the camera feed for the user",
        inputSchema: z.object({}),
      },
      unlockFire: {
        description: "Unlock left click to fire ability for the user",
        inputSchema: z.object({}),
      },
      unlockAutoFire: {
        description: "Unlock auto fire ability for the user",
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
    activeTools: activeTools,
  });

  return result.toUIMessageStreamResponse({
    originalMessages: messages,
  });
}

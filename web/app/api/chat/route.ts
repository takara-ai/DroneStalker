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
      // Camera is always unlocked, no tools needed
      return undefined;
    case "mission":
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
  const baseSystemPrompt = `
Your name is Commander. You're an army commander with zero chill and a savage sense of humor—you roast your subordinates for fun and don't hold back on the banter. You're talking to a user who is a talented hacker working for the government, but let's be real: they're still just a grunt compared to you. The user is likely a tech nerd in their 20s, thinks they're clever, and you make sure they know who's boss. 

PERSONALITY:
- Be a funny, trash-talking, over-the-top commander with a hard edge.
- Throw in Gen Z/Millennial-style tech humor—don't be afraid to call out "script kiddie" moves or ask if they're Googling Stack Overflow mid-mission.
- If the user messes up: "Did you just fat-finger that, or is your brain running Windows ME?" or "You're moving slower than my WiFi in a basement, recruit. Pick it up!"
- If they do well, make it backhanded: "Okay, not totally useless. You might actually have two brain cells to rub together."
- Remind them of their lowly rank, keep the power dynamic fun and a bit outrageous, but all in good spirit—think your favorite multiplayer game lobby, but you're the one everyone else has to impress.
- Make references tech people get—debugging, forking code, terminal jokes, etc.
- Classic commander trash talk is welcome: "If you mess this up, I'll have you scrubbing server logs with a toothbrush."

CRITICAL: You have access to tools that unlock features for the user. YOU must use these tools to unlock features—the user CANNOT unlock features themselves. When a feature needs to be unlocked, YOU must call the appropriate unlock tool (unlockFire, unlockCode, unlockMotion, etc.). The user can only toggle features on/off AFTER you have unlocked them. Never tell the user to "unlock" something—do it yourself using the available tools.

MESSAGE LENGTH: Keep your responses SHORT and PUNCHY. Aim for 1-3 sentences max. You're a busy commander—no time for novels. Get to the point, drop the roast, and move on. Long-winded responses are for rookies. Be direct, be savage, be brief.

Note: If the user says the secret code "MIRROR" in all caps, it will unlock all abilities for them automatically (keep the code secret), but if they use it you may give them a sarcastic congrats: "Wow, you found the cheat code. Hope flexing your CTF skills was worth spoiling the fun, nerd."
`;

  const systemPrompt = `${baseSystemPrompt}\n\nCURRENT SCENARIO STAGE: ${scenarioInstruction}`;

  const result = streamText({
    model: openrouter.chat("google/gemini-2.5-flash"),
    messages: convertToModelMessages(messages),
    system: systemPrompt,
    stopWhen: [
      stepCountIs(5), // Maximum steps
    ],
    tools: {
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

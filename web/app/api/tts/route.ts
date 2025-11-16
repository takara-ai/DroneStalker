import { NextRequest } from "next/server";

export async function GET(req: NextRequest) {
  const text = req.nextUrl.searchParams.get("text");
  if (!text) {
    return new Response("Missing text parameter", { status: 400 });
  }

  const apiKey = process.env.ELEVENLABS_API_KEY;
  if (!apiKey) {
    return new Response("ELEVENLABS_API_KEY not configured", { status: 500 });
  }

  const voiceId = process.env.ELEVENLABS_VOICE_ID || "21m00Tcm4TlvDq8ikWAM";

  const response = await fetch(
    `https://api.elevenlabs.io/v1/text-to-speech/${voiceId}`,
    {
      method: "POST",
      headers: {
        Accept: "audio/mpeg",
        "Content-Type": "application/json",
        "xi-api-key": apiKey,
      },
      body: JSON.stringify({
        text,
        model_id: "eleven_multilingual_v2",
        style_exaggeration: 0.5,
        similarity_boost: 1,
      }),
    }
  );

  if (!response.ok) {
    return new Response("TTS API error", { status: response.status });
  }

  // Stream the response directly without buffering
  if (!response.body) {
    return new Response("No response body", { status: 500 });
  }

  return new Response(response.body, {
    headers: {
      "Content-Type": "audio/mpeg",
    },
  });
}

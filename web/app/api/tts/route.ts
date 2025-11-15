export async function POST(req: Request) {
  const { text } = await req.json();

  if (!text || typeof text !== "string") {
    return new Response("Text is required", { status: 400 });
  }

  const apiKey = process.env.ELEVENLABS_API_KEY;
  if (!apiKey) {
    return new Response("ElevenLabs API key not configured", { status: 500 });
  }

  // Use a default voice ID - you can change this to your preferred voice
  const voiceId = process.env.ELEVENLABS_VOICE_ID || "21m00Tcm4TlvDq8ikWAM"; // Default: Rachel

  try {
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
          voice_settings: {
            stability: 0.5,
            similarity_boost: 1,
            style_exaggeration: 0.5,
          },
        }),
      }
    );

    if (!response.ok) {
      const errorText = await response.text();
      console.error("ElevenLabs API error:", errorText);
      return new Response(
        `ElevenLabs API error: ${response.status}`,
        { status: response.status }
      );
    }

    // Stream the response directly
    return new Response(response.body, {
      headers: {
        "Content-Type": "audio/mpeg",
        "Transfer-Encoding": "chunked",
      },
    });
  } catch (error) {
    console.error("TTS error:", error);
    return new Response("Failed to generate speech", { status: 500 });
  }
}


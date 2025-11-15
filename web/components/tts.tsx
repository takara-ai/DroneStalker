import { useStore } from "@/lib/store";

export function Tts() {
  const ttsQueue = useStore((state) => state.ttsQueue);
  const popFromTtsQueue = useStore((state) => state.popFromTtsQueue);
  // TODO

  return null;
}

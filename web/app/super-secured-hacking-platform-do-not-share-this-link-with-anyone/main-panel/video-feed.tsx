import { useStore } from "@/lib/store";
import { cn } from "@/lib/utils";

export default function VideoFeed() {
  const activeCamera = useStore((state) => state.activeCamera);
  const activeMotionDetection = useStore(
    (state) => state.activeMotionDetection
  );

  return (
    <div className="border-4 flex h-full overflow-hidden opened  p-2">
      {activeCamera && (
        <div className="absolute top-4 left-4 p-2 border-4 border-red-500 text-red-500 flex items-center gap-3 px-3 z-10">
          <div className="size-3 bg-red-500 rounded-full"></div>
          LIVE FEED
        </div>
      )}
      <video
        src="/colored.webm"
        autoPlay
        muted
        loop
        className={cn(
          "w-full h-full object-cover",
          activeCamera && !activeMotionDetection ? "opened" : "hidden"
        )}
      ></video>
      <video
        src="/motion.webm"
        autoPlay
        muted
        loop
        className={cn(
          "w-full h-full object-cover",
          activeCamera && activeMotionDetection ? "opened" : "hidden"
        )}
      ></video>
    </div>
  );
}

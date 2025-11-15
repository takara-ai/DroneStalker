export default function VideoFeed() {
  return (
    <div className="border-4 flex h-full overflow-hidden">
      <video
        src="/colored.webm"
        autoPlay
        muted
        loop
        className="w-full h-full object-cover p-2"
      ></video>
    </div>
  );
}

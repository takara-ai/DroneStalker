/* eslint-disable @next/next/no-img-element */

export default function CommanderVideo() {
  return (
    <div className="border-4 relative p-2 closed" id="commander">
      <img
        src="/commander.gif"
        alt="commander"
        className="w-full h-full object-cover"
      />
      <div className="absolute bottom-0 left-0 right-0 w-full bg-background/70 py-1 flex items-center justify-center">
        GENERAL
      </div>
    </div>
  );
}

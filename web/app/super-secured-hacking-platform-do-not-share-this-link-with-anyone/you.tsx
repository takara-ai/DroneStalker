/* eslint-disable @next/next/no-img-element */

export default function You() {
  return (
    <div className="border-4 relative p-2">
      <img src="/you-3.gif" alt="you" className="w-full h-full object-cover" />
      <div className=" absolute bottom-0 left-0 right-0  w-full bg-background/70 py-1 flex items-center justify-center">
        YOU
      </div>
    </div>
  );
}

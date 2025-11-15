"use client";

/* eslint-disable @next/next/no-img-element */

import { useStore } from "@/lib/store";

export default function You() {
  const { typingSpeedIndex } = useStore();

  // Map speed index (0-5) to image number
  // Speed index 0 = slowest, 5 = fastest
  // You can adjust the image numbers based on what images you have
  const imageNumber = typingSpeedIndex;
  const imageSrc = `/you-${imageNumber + 1}.gif`;

  return (
    <div className="border-4 relative p-2 closed" id="you">
      <img src={imageSrc} alt="you" className="w-full h-full object-cover" />
      <div className=" absolute bottom-0 left-0 right-0  w-full bg-background/70 py-1 flex items-center justify-center">
        YOU
      </div>
    </div>
  );
}

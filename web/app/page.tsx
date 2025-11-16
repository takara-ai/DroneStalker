"use client";

import { Button } from "@/components/ui/button";
import Link from "next/link";

export default function Home() {
  return (
    <div className="flex h-full w-full items-center justify-center p-8">
      <div className="flex flex-col gap-2 items-center justify-center">
        <h1 className="text-2xl font-bold">Disclaimer</h1>
        <p className="text-sm max-w-xs text-center text-pretty">
          For optimal results, enable{" "}
          <span className="font-bold text-white">AUDIO</span> and{" "}
          <span className="font-bold text-white">FULLSCREEN MODE</span>. This
          experience is designed for desktop only.{" "}
          <span className="text-white font-semibold">
            Chrome browser is recommended.
          </span>
        </p>
        <Button
          className="mt-4"
          onClick={() => {
            document.documentElement.requestFullscreen();
          }}
        >
          Enable fullscreen
        </Button>
        <Link href="/login">
          <Button>Continue</Button>
        </Link>
      </div>
    </div>
  );
}

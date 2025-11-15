"use client";

import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { useState } from "react";
import { useRouter } from "next/navigation";
import Image from "next/image";
import { createPortal } from "react-dom";
import { cn } from "@/lib/utils";

export default function Page() {
  const [pass, setPass] = useState("");
  const [error, setError] = useState<string | null>(null);
  const router = useRouter();

  function handleSubmit(e: React.FormEvent<HTMLFormElement>) {
    e.preventDefault();
    if (pass === "pass123") {
      router.push(
        "/super-secured-hacking-platform-do-not-share-this-link-with-anyone"
      );
    } else {
      setError("Incorrect password. Try again.");
    }
  }

  return (
    <div className="h-full w-full flex items-center justify-center relative flex-col gap-8">
      {createPortal(
        <div className="absolute -top-5 right-10 z-110 grid items-center justify-center group">
          <Image
            src="/sticky-note.webp"
            alt="sticky note"
            width={500}
            height={500}
            className={cn(
              "object-contain z-110 origin-top col-start-1 row-start-1",
              "transition-transform duration-300",
              "group-hover:transform-[perspective(600px)_rotateX(35deg)_scaleY(0.5)]"
            )}
            draggable={false}
          />
          <span className="text-xs text-foreground whitespace-normal col-start-1 row-start-1 mt-[30%] px-20 w-full max-w-xs overflow-hidden">
            Secret unlock all
            <br />
            code: &quot;MIRROR&quot;
          </span>
        </div>,
        document.body
      )}
      <div className="flex items-center gap- blur-[1px]">
        <Image
          src="/junction.png"
          alt="junction"
          className="invert"
          width={200}
          height={200}
          draggable={false}
        />
        <Image
          src="/sensorfusion.svg"
          alt="junction"
          width={200}
          height={200}
          className="p-[10%]"
          draggable={false}
        />
      </div>
      <form
        className="flex flex-col gap-2 p-10 border-4"
        onSubmit={handleSubmit}
        autoComplete="off"
      >
        <Input
          type="password"
          value={pass}
          onChange={(e) => setPass(e.target.value)}
          placeholder="Enter password"
          autoFocus
        />
        {error && <span className="text-destructive text-xs">{error}</span>}
        <Button type="submit">Login</Button>
      </form>
    </div>
  );
}

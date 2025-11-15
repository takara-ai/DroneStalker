"use client";

import { Button } from "@/components/ui/button";
import { useState } from "react";
import { useRouter } from "next/navigation";

export default function Page() {
  const [pass, setPass] = useState("");
  const [error, setError] = useState<string | null>(null);
  const router = useRouter();

  function handleSubmit(e: React.FormEvent<HTMLFormElement>) {
    e.preventDefault();
    if (pass === "password") {
      router.push(
        "/super-secured-hacking-platform-do-not-share-this-link-with-anyone"
      );
    } else {
      setError("Incorrect password. Try again.");
    }
  }

  return (
    <div className="h-full w-full flex items-center justify-center">
      <form
        className="flex flex-col gap-2 p-10 border-4"
        onSubmit={handleSubmit}
        autoComplete="off"
      >
        <input
          type="password"
          value={pass}
          onChange={(e) => setPass(e.target.value)}
          className="px-2 py-1 border-4"
          placeholder="Enter password"
          autoFocus
        />
        {error && <span className="text-destructive text-xs">{error}</span>}
        <Button type="submit">Login</Button>
      </form>
    </div>
  );
}

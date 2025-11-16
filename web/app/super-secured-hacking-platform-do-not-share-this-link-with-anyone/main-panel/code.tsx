"use client";

import type { JSX } from "react";
import type { BundledLanguage } from "shiki/bundle/web";
import { toJsxRuntime } from "hast-util-to-jsx-runtime";
import {
  Fragment,
  useCallback,
  useEffect,
  useLayoutEffect,
  useRef,
  useState,
} from "react";
import { jsx, jsxs } from "react/jsx-runtime";
import { codeToHast } from "shiki/bundle/web";
import { useStore } from "@/lib/store";
import Link from "next/link";

// Speed thresholds (milliseconds between key presses)
// Faster typing = lower interval = higher percentage per key
const SPEED_THRESHOLDS = [
  { maxInterval: 50, percentPerKey: 2.0 }, // Super fast - 2% per key
  { maxInterval: 200, percentPerKey: 1.0 }, // Fast - 1% per key
  { maxInterval: 20000, percentPerKey: 0.3 }, // Medium - 0.3% per key
  { maxInterval: Infinity, percentPerKey: 0.1 }, // Slow - 0.1% per key
];

const SLIDING_WINDOW_SIZE = 10; // Track last 10 key presses
const INACTIVITY_THRESHOLD = 500; // Reset to slowest speed after 500ms of inactivity

type CodeKey = "motionDetection" | "tracking" | "positionPrediction";

interface CodeProps {
  codeKey: CodeKey;
  code: string;
  source: string;
  onComplete?: () => void;
}

export default function Code({
  codeKey,
  code: targetCode,
  source,
  onComplete,
}: CodeProps) {
  const scrollRef = useRef<HTMLDivElement>(null);
  const inputRef = useRef<HTMLInputElement>(null);
  const codeState = useStore((state) => state.codeStates[codeKey]);
  const { setCodeTypingSpeedIndex, setCodeProgress, setCodeTypedCode } =
    useStore();

  const codeProgress = codeState.codeProgress;
  const codeTypedCode = codeState.codeTypedCode;

  // Initialize currentPositionRef from stored typed code
  const currentPositionRef = useRef(codeTypedCode.length);
  const previousTargetCodeRef = useRef(targetCode);
  const hasCompletedRef = useRef(false);

  // Reset progress when target code changes
  useEffect(() => {
    // If the target code changed, reset everything
    if (previousTargetCodeRef.current !== targetCode) {
      previousTargetCodeRef.current = targetCode;
      currentPositionRef.current = 0;
      setCodeTypedCode(codeKey, "");
      setCodeProgress(codeKey, 0);
      hasCompletedRef.current = false; // Reset completion flag
      return;
    }

    // Validate that typed code is still a valid prefix of target code
    // If not, reset (this handles edge cases where state might get corrupted)
    if (codeTypedCode && !targetCode.startsWith(codeTypedCode)) {
      currentPositionRef.current = 0;
      setCodeTypedCode(codeKey, "");
      setCodeProgress(codeKey, 0);
      hasCompletedRef.current = false; // Reset completion flag
    } else {
      currentPositionRef.current = codeTypedCode.length;
    }
  }, [targetCode, codeTypedCode, setCodeTypedCode, setCodeProgress, codeKey]);

  // Sync currentPositionRef when codeTypedCode changes from outside
  useEffect(() => {
    currentPositionRef.current = codeTypedCode.length;
  }, [codeTypedCode]);

  // Auto-complete code when progress is 100% (e.g., from unlockAll)
  useEffect(() => {
    if (codeProgress >= 100 && codeTypedCode !== targetCode) {
      setCodeTypedCode(codeKey, targetCode);
      currentPositionRef.current = targetCode.length;
    }
  }, [codeProgress, codeTypedCode, targetCode, setCodeTypedCode, codeKey]);

  // Typing speed tracking with sliding window
  const keyPressTimestampsRef = useRef<number[]>([]);
  const lastKeyPressTimeRef = useRef<number | null>(null);
  const [debugInfo, setDebugInfo] = useState<{
    avgInterval: number;
    percentPerKey: number;
  }>({ avgInterval: 0, percentPerKey: 0.1 });

  // Calculate typing speed and determine characters to advance
  // Returns both the chars to advance and the speed index (0-3)
  const getSpeedInfo = useCallback(
    (
      currentTime?: number,
      totalCodeLength?: number
    ): {
      charsToAdvance: number;
      speedIndex: number;
      avgInterval: number;
      percentPerKey: number;
    } => {
      const now = currentTime ?? Date.now();
      const timestamps = keyPressTimestampsRef.current;
      const codeLength = totalCodeLength ?? targetCode.length;

      // If no recent key presses or too much time has passed since last key press, reset to slowest
      if (
        timestamps.length === 0 ||
        lastKeyPressTimeRef.current === null ||
        now - lastKeyPressTimeRef.current > INACTIVITY_THRESHOLD
      ) {
        const slowestPercent =
          SPEED_THRESHOLDS[SPEED_THRESHOLDS.length - 1].percentPerKey;
        return {
          charsToAdvance: Math.max(
            1,
            Math.floor((codeLength * slowestPercent) / 100)
          ),
          speedIndex: 0,
          avgInterval: Infinity,
          percentPerKey: slowestPercent,
        };
      }

      if (timestamps.length < 2) {
        // Not enough data yet, default to slowest speed
        const slowestPercent =
          SPEED_THRESHOLDS[SPEED_THRESHOLDS.length - 1].percentPerKey;
        return {
          charsToAdvance: Math.max(
            1,
            Math.floor((codeLength * slowestPercent) / 100)
          ),
          speedIndex: 0,
          avgInterval: Infinity,
          percentPerKey: slowestPercent,
        };
      }

      // Calculate average time between key presses in the sliding window
      // Include the time since the last key press as a "virtual" interval
      let totalInterval = 0;
      let intervalCount = 0;

      // Add intervals between timestamps
      for (let i = 1; i < timestamps.length; i++) {
        totalInterval += timestamps[i] - timestamps[i - 1];
        intervalCount++;
      }

      // Add the time since the last key press as the most recent interval
      // This makes the speed calculation decay when not typing
      if (lastKeyPressTimeRef.current !== null) {
        totalInterval += now - lastKeyPressTimeRef.current;
        intervalCount++;
      }

      const avgInterval = totalInterval / intervalCount;

      // Find the appropriate threshold and return both values
      for (let i = 0; i < SPEED_THRESHOLDS.length; i++) {
        const threshold = SPEED_THRESHOLDS[i];
        if (avgInterval < threshold.maxInterval) {
          // Speed index: 3 (fastest) to 0 (slowest)
          // Index 3 = super fast, Index 2 = fast, Index 1 = medium, Index 0 = slow
          const speedIndex = SPEED_THRESHOLDS.length - 1 - i;
          // Calculate chars to advance based on percentage of total code length
          const charsToAdvance = Math.max(
            1,
            Math.floor((codeLength * threshold.percentPerKey) / 100)
          );
          return {
            charsToAdvance,
            speedIndex,
            avgInterval,
            percentPerKey: threshold.percentPerKey,
          };
        }
      }

      // Fallback to slowest
      const slowestPercent =
        SPEED_THRESHOLDS[SPEED_THRESHOLDS.length - 1].percentPerKey;
      return {
        charsToAdvance: Math.max(
          1,
          Math.floor((codeLength * slowestPercent) / 100)
        ),
        speedIndex: 0,
        avgInterval,
        percentPerKey: slowestPercent,
      };
    },
    [targetCode.length]
  );

  // Auto-focus input when component mounts
  useEffect(() => {
    const timer = setTimeout(() => {
      inputRef.current?.focus();
    }, 100);
    return () => clearTimeout(timer);
  }, []);

  // Auto-scroll to bottom when code updates
  useEffect(() => {
    if (scrollRef.current) {
      scrollRef.current.scrollTop = scrollRef.current.scrollHeight;
    }
  }, [codeTypedCode]);

  // Check for completion when progress changes (handles cases where progress updates from other sources)
  useEffect(() => {
    if (codeProgress >= 100 && !hasCompletedRef.current && onComplete) {
      hasCompletedRef.current = true;
      onComplete();
    }
  }, [codeProgress, onComplete]);

  // Periodically update speed index even when not typing (to decay speed)
  // Also update debug info
  useEffect(() => {
    const interval = setInterval(() => {
      const speedInfo = getSpeedInfo(undefined, targetCode.length);
      setCodeTypingSpeedIndex(codeKey, speedInfo.speedIndex);
      // Update debug info
      setDebugInfo({
        avgInterval: speedInfo.avgInterval,
        percentPerKey: speedInfo.percentPerKey,
      });
    }, 100); // Check every 100ms

    return () => clearInterval(interval);
  }, [getSpeedInfo, setCodeTypingSpeedIndex, codeKey, targetCode.length]);

  // Handle keyboard input - only prevent default for keys we handle
  const handleKeyDown = (e: React.KeyboardEvent<HTMLInputElement>) => {
    // Only handle keys that are relevant to our code typing
    const isRelevantKey =
      !e.metaKey &&
      !e.ctrlKey &&
      !e.altKey &&
      !e.key.startsWith("F") &&
      (e.key === "Backspace" ||
        e.key === "Enter" ||
        e.key === "Tab" ||
        e.key.length === 1);

    if (!isRelevantKey) {
      // Let other keys (like shortcuts) pass through
      return;
    }

    // Prevent default only for keys we're handling
    e.preventDefault();

    // Handle backspace separately (doesn't count toward typing speed)
    if (e.key === "Backspace") {
      if (currentPositionRef.current > 0) {
        currentPositionRef.current--;
        const newTypedCode = targetCode.slice(0, currentPositionRef.current);
        setCodeTypedCode(codeKey, newTypedCode);
        const progress = (currentPositionRef.current / targetCode.length) * 100;
        setCodeProgress(codeKey, Math.min(100, Math.round(progress)));
      }
      return;
    }

    // Advance through the target code as user types
    if (currentPositionRef.current >= targetCode.length) {
      return; // Already completed
    }

    const now = Date.now();
    const nextChar = targetCode[currentPositionRef.current];
    let charsToAdvance = 1;
    let shouldTrackSpeed = false;

    // Handle special characters
    if (e.key === "Enter" && nextChar === "\n") {
      charsToAdvance = 1;
      shouldTrackSpeed = true;
    } else if (e.key === "Tab" && nextChar === "\t") {
      charsToAdvance = 1;
      shouldTrackSpeed = true;
    } else if (e.key.length === 1) {
      // Regular character - calculate how many chars to advance based on speed
      shouldTrackSpeed = true;
      const speedInfo = getSpeedInfo(undefined, targetCode.length);
      charsToAdvance = speedInfo.charsToAdvance;
      // Update the speed index in the store
      setCodeTypingSpeedIndex(codeKey, speedInfo.speedIndex);
    } else {
      return;
    }

    // Update sliding window of key press timestamps for speed calculation
    if (shouldTrackSpeed) {
      keyPressTimestampsRef.current.push(now);
      if (keyPressTimestampsRef.current.length > SLIDING_WINDOW_SIZE) {
        keyPressTimestampsRef.current.shift(); // Remove oldest
      }
      lastKeyPressTimeRef.current = now; // Update last key press time

      // Update debug info after adding timestamp
      const updatedSpeedInfo = getSpeedInfo(now, targetCode.length);
      setDebugInfo({
        avgInterval: updatedSpeedInfo.avgInterval,
        percentPerKey: updatedSpeedInfo.percentPerKey,
      });
    }

    // Advance by the calculated number of characters
    const newPosition = Math.min(
      currentPositionRef.current + charsToAdvance,
      targetCode.length
    );
    currentPositionRef.current = newPosition;
    const newTypedCode = targetCode.slice(0, currentPositionRef.current);
    setCodeTypedCode(codeKey, newTypedCode);

    // Update progress
    const progress = (currentPositionRef.current / targetCode.length) * 100;
    const roundedProgress = Math.min(100, Math.round(progress));
    setCodeProgress(codeKey, roundedProgress);

    // Trigger completion callback when reaching 100%
    if (roundedProgress >= 100 && !hasCompletedRef.current && onComplete) {
      hasCompletedRef.current = true;
      onComplete();
    }
  };

  // Keep input focused when clicking on the code area
  const handleCodeAreaClick = () => {
    inputRef.current?.focus();
  };

  // Note: We don't auto-refocus on blur to avoid interfering with keyboard shortcuts
  // Users can click on the code area to refocus when needed

  return (
    <div className="border-4 p-2 flex overflow-hidden h-full opened gap-2 flex-col relative">
      <div
        className="rounded-none overflow-y-auto flex-1 cursor-text"
        ref={scrollRef}
        onClick={handleCodeAreaClick}
      >
        <CodeBlock
          code={codeTypedCode || "Type your code here..."}
          lang="python"
        />
      </div>
      <div className="absolute right-6 top-2 text-sm max-w-xs text-pretty text-right bg-background/50">
        (actual working code we wrote to solve the challenge, available on{" "}
        <Link href={source} target="_blank" className="text-white underline">
          GitHub
        </Link>
        )
      </div>
      {/* <div className="absolute left-6 top-2 text-xs font-mono bg-background/80 border border-border/50 p-2 rounded">
        <div className="text-foreground/70">DEBUG:</div>
        <div>Speed Index: {typingSpeedIndex}/3</div>
        <div>Percent/Key: {debugInfo.percentPerKey}%</div>
        <div>
          Avg Interval:{" "}
          {debugInfo.avgInterval === Infinity
            ? "∞"
            : `${Math.round(debugInfo.avgInterval)}ms`}
        </div>
      </div> */}
      <input
        ref={inputRef}
        type="text"
        className="absolute w-px h-px opacity-0"
        autoFocus
        onKeyDown={handleKeyDown}
        value=""
        readOnly
      />
      <div className="w-full h-10 border-4 border-border/50 flex items-center justify-center px-4 relative">
        <div
          style={{ width: `${codeProgress}%` }}
          className="absolute left-0 top-0 h-full bg-foreground"
        ></div>
        <span className="mix-blend-difference">{codeProgress}% COMPLETED</span>
      </div>
    </div>
  );
}

async function highlight(code: string, lang: BundledLanguage) {
  const hast = await codeToHast(code, {
    lang,
    theme: "github-dark",
  });

  return toJsxRuntime(hast, {
    Fragment,
    jsx,
    jsxs,
    components: {
      pre: (props) => (
        <pre
          {...props}
          className="rounded-none overflow-y-auto whitespace-pre-wrap flex-1 m-0 p-4"
          style={{
            backgroundColor: "var(--shiki-dark-bg)",
            color: "var(--shiki-dark)",
          }}
        />
      ),
      code: (props) => (
        <code
          {...props}
          className="block"
          style={{ fontFamily: "monospace" }}
        />
      ),
    },
  }) as JSX.Element;
}

function CodeBlock({ code, lang }: { code: string; lang: BundledLanguage }) {
  const [nodes, setNodes] = useState<JSX.Element | null>(null);

  useLayoutEffect(() => {
    void highlight(code, lang).then(setNodes);
  }, [code, lang]);

  return nodes ?? <p>Loading...</p>;
}

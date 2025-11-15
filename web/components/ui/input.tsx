import React from "react";

interface InputProps extends React.InputHTMLAttributes<HTMLInputElement> {
  className?: string;
}

export function Input({ className = "", ...props }: InputProps) {
  return (
    <div className="w-full relative">
      <input
        {...props}
        className={`border-4 border-border/50 active:border-border focus:border-border outline-none p-2 w-full pl-8 ${className}`}
      />
      <span className="absolute left-4 top-1/2 -translate-y-1/2">&gt;</span>
    </div>
  );
}


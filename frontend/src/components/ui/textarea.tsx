import * as React from "react";
import { cn } from "../../lib/utils";

export const Textarea = React.forwardRef<HTMLTextAreaElement, React.ComponentProps<"textarea">>(
  ({ className, ...props }, ref) => (
    <textarea
      ref={ref}
      className={cn("flex min-h-10 w-full resize-none bg-transparent text-sm outline-none placeholder:text-white/45 disabled:cursor-not-allowed", className)}
      {...props}
    />
  ),
);
Textarea.displayName = "Textarea";

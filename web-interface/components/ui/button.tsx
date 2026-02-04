import * as React from "react";
import { cn } from "@/lib/utils";

export type ButtonVariant = "default" | "outline";

export interface ButtonProps extends React.ButtonHTMLAttributes<HTMLButtonElement> {
  variant?: ButtonVariant;
}

const Button = React.forwardRef<HTMLButtonElement, ButtonProps>(({ className, variant = "default", ...props }, ref) => {
  const base =
    "inline-flex items-center justify-center rounded-md text-sm font-medium transition-colors focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-slate-950 focus-visible:ring-offset-2 disabled:pointer-events-none disabled:opacity-50 dark:focus-visible:ring-slate-300 dark:focus-visible:ring-offset-slate-950 ring-offset-white";

  const variants: Record<ButtonVariant, string> = {
    default: "bg-slate-900 text-slate-50 hover:bg-slate-900/90 dark:bg-slate-50 dark:text-slate-900 dark:hover:bg-slate-50/90",
    outline:
      "border border-slate-200 bg-white hover:bg-slate-100 hover:text-slate-900 dark:border-slate-800 dark:bg-slate-950 dark:hover:bg-slate-800 dark:hover:text-slate-50",
  };

  return (
    <button
      ref={ref}
      className={cn(base, variants[variant], "h-10 px-4 py-2", className)}
      {...props}
    />
  );
});
Button.displayName = "Button";

export { Button };

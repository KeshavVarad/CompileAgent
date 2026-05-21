"use client";

import { useRouter } from "next/navigation";
import { useState, useTransition } from "react";
import { toast } from "sonner";

import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Card, CardContent, CardHeader } from "@/components/ui/card";

type Mode = "login" | "signup";

export function LoginForm({ next }: { next: string }) {
  const router = useRouter();
  const [mode, setMode] = useState<Mode>("login");
  const [username, setUsername] = useState("");
  const [password, setPassword] = useState("");
  const [pending, startTransition] = useTransition();

  function submit() {
    startTransition(async () => {
      try {
        const endpoint = mode === "login" ? "/api/auth/login" : "/api/auth/signup";
        const res = await fetch(endpoint, {
          method: "POST",
          headers: { "content-type": "application/json" },
          body: JSON.stringify({ username, password }),
        });
        if (!res.ok) {
          const err = await res.json().catch(() => ({}));
          toast.error(err.error || `Failed (${res.status})`);
          return;
        }
        router.push(next);
        router.refresh();
      } catch (e) {
        toast.error(e instanceof Error ? e.message : "Auth failed");
      }
    });
  }

  return (
    <div className="flex-1 mx-auto w-full max-w-md px-6 py-20">
      <div className="text-xs text-muted-foreground uppercase tracking-widest font-mono mb-2">
        console::compile.recorder
      </div>
      <h1 className="text-3xl font-semibold tracking-tight mb-8">
        {mode === "login" ? "Sign in" : "Create account"}
      </h1>

      <p className="text-xs text-muted-foreground mb-6">
        Play against Sparkv1 or transcribe live Compile games for AI review.{" "}
        <a
          href="https://github.com/KeshavVarad/CompileAgent"
          target="_blank"
          rel="noreferrer"
          className="underline hover:text-foreground"
        >
          source ↗
        </a>
      </p>

      <Card>
        <CardHeader className="pb-2 flex flex-row items-center justify-between">
          <div className="flex gap-1">
            <Button
              variant={mode === "login" ? "default" : "ghost"}
              size="sm"
              onClick={() => setMode("login")}
            >
              Log in
            </Button>
            <Button
              variant={mode === "signup" ? "default" : "ghost"}
              size="sm"
              onClick={() => setMode("signup")}
            >
              Sign up
            </Button>
          </div>
        </CardHeader>
        <CardContent>
          <form
            className="grid gap-4"
            onSubmit={(e) => {
              e.preventDefault();
              submit();
            }}
          >
            <div className="grid gap-2">
              <Label htmlFor="username">Username</Label>
              <Input
                id="username"
                value={username}
                onChange={(e) => setUsername(e.target.value)}
                autoComplete="username"
                autoFocus
                required
              />
            </div>
            <div className="grid gap-2">
              <Label htmlFor="password">Password</Label>
              <Input
                id="password"
                type="password"
                value={password}
                onChange={(e) => setPassword(e.target.value)}
                autoComplete={mode === "login" ? "current-password" : "new-password"}
                required
                minLength={mode === "signup" ? 6 : undefined}
              />
            </div>
            <div className="text-[11px] text-muted-foreground">
              {mode === "signup"
                ? "2–32 chars, letters / digits / . _ - · password ≥6 chars."
                : "No account yet? Click Sign up above."}
            </div>
            <Button type="submit" disabled={pending}>
              {pending ? "..." : mode === "login" ? "Log in" : "Create account"}
            </Button>
          </form>
        </CardContent>
      </Card>
    </div>
  );
}

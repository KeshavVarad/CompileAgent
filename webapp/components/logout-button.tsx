"use client";

import { useRouter } from "next/navigation";
import { useTransition } from "react";
import { toast } from "sonner";

import { Button } from "@/components/ui/button";

export function LogoutButton() {
  const router = useRouter();
  const [pending, startTransition] = useTransition();

  function logout() {
    startTransition(async () => {
      try {
        const res = await fetch("/api/auth/logout", { method: "POST" });
        if (!res.ok) {
          toast.error(`Logout failed (${res.status})`);
          return;
        }
        router.push("/login");
        router.refresh();
      } catch (e) {
        toast.error(e instanceof Error ? e.message : "Logout failed");
      }
    });
  }

  return (
    <Button
      variant="ghost"
      size="sm"
      className="h-6 px-2 text-[11px]"
      onClick={logout}
      disabled={pending}
    >
      log out
    </Button>
  );
}

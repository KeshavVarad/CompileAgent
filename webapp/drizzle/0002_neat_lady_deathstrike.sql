ALTER TABLE "games" ADD COLUMN "mode" text DEFAULT 'play' NOT NULL;--> statement-breakpoint
ALTER TABLE "games" ADD COLUMN "recorder_seat" integer;
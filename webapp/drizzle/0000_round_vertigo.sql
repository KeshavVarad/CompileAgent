CREATE TABLE "games" (
	"id" uuid PRIMARY KEY DEFAULT gen_random_uuid() NOT NULL,
	"created_at" timestamp with time zone DEFAULT now() NOT NULL,
	"updated_at" timestamp with time zone DEFAULT now() NOT NULL,
	"ended_at" timestamp with time zone,
	"player0_label" text NOT NULL,
	"player1_label" text NOT NULL,
	"include_expansion" boolean DEFAULT false NOT NULL,
	"max_turns" integer DEFAULT 200 NOT NULL,
	"seed" integer NOT NULL,
	"bot0_strategy" text,
	"bot1_strategy" text,
	"actions" jsonb DEFAULT '[]'::jsonb NOT NULL,
	"winner" integer,
	"turn_count" integer
);

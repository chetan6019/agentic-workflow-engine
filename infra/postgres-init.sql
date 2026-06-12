-- Runs once on a fresh Postgres volume (docker-entrypoint-initdb.d).
-- Creates a dedicated database for the LiteLLM proxy so its Prisma schema sync
-- never drops the application's `workflow` tables (users, sessions, ...).
CREATE DATABASE litellm;

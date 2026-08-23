# Runtime persistence migrations

This directory contains migration assets for the independent Agent Runtime
MySQL 8 database `mico_agent_runtime`. Alembic accepts only
`MICO_AGENT_RUNTIME_MIGRATION_DATABASE_URL`; Runtime state encryption keys are
not required by the migration-only boundary. It is not connected to the
business schema `patient_data_manager` and must not be run until a separately
approved deployment.

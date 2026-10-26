-- pgchroma catalog. Runs inside the target schema (search_path), idempotently.

CREATE TABLE IF NOT EXISTS meta (
    key   text PRIMARY KEY,
    value text NOT NULL
);

CREATE TABLE IF NOT EXISTS tenants (
    name          text PRIMARY KEY,
    resource_name text,
    created_at    timestamptz NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS databases (
    id         uuid PRIMARY KEY,
    tenant     text NOT NULL REFERENCES tenants (name) ON DELETE CASCADE,
    name       text NOT NULL,
    created_at timestamptz NOT NULL DEFAULT now(),
    UNIQUE (tenant, name)
);

CREATE TABLE IF NOT EXISTS collections (
    id            uuid PRIMARY KEY,
    database_id   uuid NOT NULL REFERENCES databases (id) ON DELETE CASCADE,
    name          text NOT NULL,
    metadata      jsonb,
    configuration jsonb NOT NULL,
    schema        jsonb,
    dimension     integer,
    space         text NOT NULL DEFAULT 'l2',
    version       integer NOT NULL DEFAULT 0,
    log_position  bigint NOT NULL DEFAULT 0,
    forked_from   uuid,
    created_at    timestamptz NOT NULL DEFAULT now(),
    updated_at    timestamptz NOT NULL DEFAULT now(),
    UNIQUE (database_id, name)
);

CREATE TABLE IF NOT EXISTS migrations (
    source        text NOT NULL,
    collection_id uuid NOT NULL REFERENCES collections (id) ON DELETE CASCADE,
    copied        bigint NOT NULL DEFAULT 0,
    done          boolean NOT NULL DEFAULT false,
    updated_at    timestamptz NOT NULL DEFAULT now(),
    PRIMARY KEY (source, collection_id)
);

-- One row while a process builds a collection's HNSW index. A lease instead of an advisory
-- lock, because session locks do not work behind transaction poolers.
CREATE TABLE IF NOT EXISTS index_builds (
    table_name text PRIMARY KEY,
    owner      text NOT NULL,
    started_at timestamptz NOT NULL DEFAULT now()
);

INSERT INTO tenants (name) VALUES ('default_tenant') ON CONFLICT DO NOTHING;

INSERT INTO databases (id, tenant, name)
VALUES ('00000000-0000-0000-0000-000000000000', 'default_tenant', 'default_database')
ON CONFLICT DO NOTHING;

INSERT INTO meta (key, value) VALUES ('catalog_version', '1') ON CONFLICT DO NOTHING;

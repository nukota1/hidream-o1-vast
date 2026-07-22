CREATE TABLE IF NOT EXISTS gallery_records (
  id TEXT PRIMARY KEY,
  created_at INTEGER NOT NULL,
  workflow TEXT NOT NULL,
  prompt TEXT NOT NULL DEFAULT '',
  optimized_prompt TEXT NOT NULL DEFAULT '',
  metadata_json TEXT NOT NULL DEFAULT '{}',
  object_key TEXT NOT NULL,
  folder_id TEXT NOT NULL DEFAULT ''
);

CREATE INDEX IF NOT EXISTS idx_gallery_records_created_at ON gallery_records(created_at DESC);
CREATE INDEX IF NOT EXISTS idx_gallery_records_folder_id ON gallery_records(folder_id);

CREATE TABLE IF NOT EXISTS gallery_folders (
  id TEXT PRIMARY KEY,
  name TEXT NOT NULL,
  created_at INTEGER NOT NULL,
  updated_at INTEGER NOT NULL
);

CREATE TABLE IF NOT EXISTS favorite_groups (
  id TEXT PRIMARY KEY,
  name TEXT NOT NULL,
  items_json TEXT NOT NULL,
  created_at INTEGER NOT NULL,
  updated_at INTEGER NOT NULL
);

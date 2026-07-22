ALTER TABLE gallery_records ADD COLUMN owner_id TEXT NOT NULL DEFAULT '';
ALTER TABLE gallery_folders ADD COLUMN owner_id TEXT NOT NULL DEFAULT '';
ALTER TABLE favorite_groups ADD COLUMN owner_id TEXT NOT NULL DEFAULT '';

CREATE INDEX IF NOT EXISTS idx_gallery_records_owner_created_at
  ON gallery_records(owner_id, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_gallery_records_owner_folder_id
  ON gallery_records(owner_id, folder_id);
CREATE INDEX IF NOT EXISTS idx_gallery_folders_owner_created_at
  ON gallery_folders(owner_id, created_at ASC);
CREATE INDEX IF NOT EXISTS idx_favorite_owner_created
  ON favorite_groups(owner_id, created_at ASC);

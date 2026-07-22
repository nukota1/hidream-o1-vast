import {createRemoteJWKSet, jwtVerify} from "jose";

export interface Env {
  BACKEND_URL?: string;
  BACKEND_SHARED_SECRET?: string;
  ACCESS_TEAM_DOMAIN?: string;
  ACCESS_AUD?: string;
  ALLOWED_ORIGIN?: string;
  GALLERY_BUCKET?: R2Bucket;
  DB?: D1Database;
}

type GalleryRecordInput = {
  id?: string;
  image?: string;
  createdAt?: number;
  workflow?: string;
  prompt?: string;
  optimizedPrompt?: string;
  folderId?: string;
  metadata?: Record<string, unknown>;
};

class HttpError extends Error {
  constructor(public status: number, message: string) {
    super(message);
  }
}

function corsHeaders(env: Env): HeadersInit {
  return {
    "Access-Control-Allow-Origin": env.ALLOWED_ORIGIN || "*",
    "Access-Control-Allow-Methods": "GET,POST,PUT,PATCH,DELETE,OPTIONS",
    "Access-Control-Allow-Headers": "Content-Type,Authorization",
    "Access-Control-Expose-Headers": "Content-Type,Content-Length",
  };
}

function jsonResponse(env: Env, value: unknown, status = 200): Response {
  return new Response(JSON.stringify(value), {
    status,
    headers: {
      ...corsHeaders(env),
      "Content-Type": "application/json; charset=utf-8",
      "Cache-Control": "no-store",
    },
  });
}

async function requireAuthenticatedUser(request: Request, env: Env): Promise<string> {
  const teamDomain = String(env.ACCESS_TEAM_DOMAIN || "").replace(/\/+$/, "");
  const audience = String(env.ACCESS_AUD || "").trim();
  if (!teamDomain || !audience) {
    throw new HttpError(503, "Cloudflare Access is not configured.");
  }

  const assertion = request.headers.get("cf-access-jwt-assertion");
  if (!assertion) throw new HttpError(401, "Cloudflare Access authentication is required.");

  try {
    const jwks = createRemoteJWKSet(new URL(`${teamDomain}/cdn-cgi/access/certs`));
    const {payload} = await jwtVerify(assertion, jwks, {
      issuer: teamDomain,
      audience,
    });
    const userId = String(payload.sub || payload.email || "").trim();
    if (!userId) throw new Error("Access token has no user identity.");
    return userId.slice(0, 200);
  } catch {
    throw new HttpError(403, "Invalid Cloudflare Access authentication.");
  }
}

function requireStorage(env: Env): asserts env is Env & {GALLERY_BUCKET: R2Bucket; DB: D1Database} {
  if (!env.GALLERY_BUCKET || !env.DB) {
    throw new Error("Cloudflare R2/D1 bindings are not configured.");
  }
}

function safeId(value: unknown): string {
  const id = String(value || crypto.randomUUID()).trim();
  if (!/^[a-zA-Z0-9_-]{1,120}$/.test(id)) throw new Error("Invalid resource id.");
  return id;
}

function imageKey(id: string): string {
  return `gallery/${id}.png`;
}

function imageUrl(request: Request, id: string): string {
  return new URL(`/api/gallery/image/${encodeURIComponent(id)}`, request.url).toString();
}

function stripDataUrl(value: string): string {
  return value.startsWith("data:") ? value.split(",", 2)[1] || "" : value;
}

function decodeBase64(value: string): Uint8Array {
  const binary = atob(stripDataUrl(value));
  const bytes = new Uint8Array(binary.length);
  for (let index = 0; index < binary.length; index += 1) bytes[index] = binary.charCodeAt(index);
  return bytes;
}

function parseMetadata(value: unknown): Record<string, unknown> {
  if (!value) return {};
  if (typeof value === "object") return value as Record<string, unknown>;
  try {
    return JSON.parse(String(value));
  } catch {
    return {};
  }
}

function serializeRecord(request: Request, row: Record<string, unknown>) {
  const id = String(row.id);
  return {
    id,
    createdAt: Number(row.created_at),
    workflow: String(row.workflow || "compose"),
    prompt: String(row.prompt || ""),
    optimizedPrompt: String(row.optimized_prompt || ""),
    folderId: String(row.folder_id || ""),
    metadata: parseMetadata(row.metadata_json),
    image_url: imageUrl(request, id),
  };
}

async function readJson(request: Request): Promise<Record<string, any>> {
  const value = await request.json();
  if (!value || typeof value !== "object" || Array.isArray(value)) {
    throw new Error("JSON object is required.");
  }
  return value as Record<string, any>;
}

async function galleryRecords(request: Request, env: Env, ownerId: string): Promise<Response> {
  requireStorage(env);
  const folderId = new URL(request.url).searchParams.get("folderId");
  const result = folderId === null
    ? await env.DB.prepare("SELECT * FROM gallery_records WHERE owner_id = ? ORDER BY created_at DESC LIMIT 500").bind(ownerId).all()
    : await env.DB.prepare("SELECT * FROM gallery_records WHERE owner_id = ? AND folder_id = ? ORDER BY created_at DESC LIMIT 500").bind(ownerId, folderId).all();
  return jsonResponse(env, {items: (result.results || []).map((row) => serializeRecord(request, row as Record<string, unknown>))});
}

async function createGalleryRecord(request: Request, env: Env, ownerId: string): Promise<Response> {
  requireStorage(env);
  const body = await readJson(request) as GalleryRecordInput;
  const id = safeId(body.id);
  const existing = await env.DB.prepare("SELECT owner_id FROM gallery_records WHERE id = ?").bind(id).first();
  if (existing && String((existing as Record<string, unknown>).owner_id || "") !== ownerId) {
    throw new HttpError(409, "Gallery record already belongs to another user.");
  }
  const image = String(body.image || "");
  if (!image) throw new Error("Image data is required.");
  const createdAt = Number(body.createdAt || Date.now());
  const workflow = String(body.workflow || "compose").slice(0, 40);
  const prompt = String(body.prompt || "").slice(0, 20000);
  const optimizedPrompt = String(body.optimizedPrompt || "").slice(0, 20000);
  const folderId = String(body.folderId || "").slice(0, 120);
  const metadata = JSON.stringify(body.metadata || {});

  await env.GALLERY_BUCKET.put(imageKey(id), decodeBase64(image), {
    httpMetadata: {contentType: "image/png", cacheControl: "private, max-age=3600"},
    customMetadata: {workflow, createdAt: String(createdAt)},
  });
  await env.DB.prepare(
    `INSERT INTO gallery_records
      (id, owner_id, created_at, workflow, prompt, optimized_prompt, metadata_json, object_key, folder_id)
     VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
     ON CONFLICT(id) DO UPDATE SET
       owner_id = excluded.owner_id,
       created_at = excluded.created_at,
       workflow = excluded.workflow,
       prompt = excluded.prompt,
       optimized_prompt = excluded.optimized_prompt,
       metadata_json = excluded.metadata_json,
       object_key = excluded.object_key,
       folder_id = excluded.folder_id`,
  ).bind(id, ownerId, createdAt, workflow, prompt, optimizedPrompt, metadata, imageKey(id), folderId).run();

  const result = await env.DB.prepare("SELECT * FROM gallery_records WHERE id = ? AND owner_id = ?").bind(id, ownerId).first();
  return jsonResponse(env, {record: serializeRecord(request, result as Record<string, unknown>)});
}

async function serveGalleryImage(request: Request, env: Env, id: string, ownerId: string): Promise<Response> {
  requireStorage(env);
  const record = await env.DB.prepare("SELECT object_key FROM gallery_records WHERE id = ? AND owner_id = ?").bind(id, ownerId).first();
  if (!record) return jsonResponse(env, {error: "Image not found."}, 404);
  const object = await env.GALLERY_BUCKET.get(imageKey(id));
  if (!object) return jsonResponse(env, {error: "Image not found."}, 404);
  const headers = new Headers(corsHeaders(env));
  object.writeHttpMetadata(headers);
  headers.set("ETag", object.httpEtag);
  headers.set("Cache-Control", "private, max-age=3600");
  return new Response(object.body, {headers});
}

async function patchGalleryRecord(request: Request, env: Env, id: string, ownerId: string): Promise<Response> {
  requireStorage(env);
  const body = await readJson(request);
  const folderId = String(body.folderId || "").slice(0, 120);
  await env.DB.prepare("UPDATE gallery_records SET folder_id = ? WHERE id = ? AND owner_id = ?").bind(folderId, id, ownerId).run();
  return jsonResponse(env, {ok: true});
}

async function galleryFolders(request: Request, env: Env, ownerId: string): Promise<Response> {
  requireStorage(env);
  const result = await env.DB.prepare("SELECT * FROM gallery_folders WHERE owner_id = ? ORDER BY created_at ASC").bind(ownerId).all();
  return jsonResponse(env, {
    items: (result.results || []).map((row) => ({
      id: String((row as any).id), name: String((row as any).name),
      createdAt: Number((row as any).created_at), updatedAt: Number((row as any).updated_at),
    })),
  });
}

async function createGalleryFolder(request: Request, env: Env, ownerId: string): Promise<Response> {
  requireStorage(env);
  const body = await readJson(request);
  const id = safeId(body.id);
  const name = String(body.name || "").trim().slice(0, 80);
  if (!name) throw new Error("Folder name is required.");
  const createdAt = Number(body.createdAt || Date.now());
  await env.DB.prepare("INSERT INTO gallery_folders (id, owner_id, name, created_at, updated_at) VALUES (?, ?, ?, ?, ?)")
    .bind(id, ownerId, name, createdAt, createdAt).run();
  return jsonResponse(env, {id, name, createdAt, updatedAt: createdAt});
}

async function patchGalleryFolder(request: Request, env: Env, id: string, ownerId: string): Promise<Response> {
  requireStorage(env);
  const body = await readJson(request);
  const name = String(body.name || "").trim().slice(0, 80);
  if (!name) throw new Error("Folder name is required.");
  const updatedAt = Number(body.updatedAt || Date.now());
  await env.DB.prepare("UPDATE gallery_folders SET name = ?, updated_at = ? WHERE id = ? AND owner_id = ?")
    .bind(name, updatedAt, id, ownerId).run();
  return jsonResponse(env, {ok: true});
}

async function deleteGalleryFolder(env: Env, id: string, ownerId: string): Promise<Response> {
  requireStorage(env);
  await env.DB.batch([
    env.DB.prepare("UPDATE gallery_records SET folder_id = '' WHERE folder_id = ? AND owner_id = ?").bind(id, ownerId),
    env.DB.prepare("DELETE FROM gallery_folders WHERE id = ? AND owner_id = ?").bind(id, ownerId),
  ]);
  return jsonResponse(env, {ok: true});
}

async function favorites(request: Request, env: Env, ownerId: string): Promise<Response> {
  requireStorage(env);
  const result = await env.DB.prepare("SELECT * FROM favorite_groups WHERE owner_id = ? ORDER BY created_at ASC").bind(ownerId).all();
  return jsonResponse(env, {
    items: (result.results || []).map((row) => ({
      id: String((row as any).id), name: String((row as any).name),
      items: parseMetadata((row as any).items_json) as unknown as unknown[],
      createdAt: Number((row as any).created_at), updatedAt: Number((row as any).updated_at),
    })),
  });
}

async function putFavorite(request: Request, env: Env, id: string, ownerId: string): Promise<Response> {
  requireStorage(env);
  const body = await readJson(request);
  const name = String(body.name || "favorite").trim().slice(0, 80);
  const items = Array.isArray(body.items) ? body.items.slice(0, 200) : [];
  const createdAt = Number(body.createdAt || Date.now());
  const updatedAt = Number(body.updatedAt || Date.now());
  const existing = await env.DB.prepare("SELECT owner_id FROM favorite_groups WHERE id = ?").bind(id).first();
  if (existing && String((existing as Record<string, unknown>).owner_id || "") !== ownerId) {
    throw new HttpError(409, "Favorite group already belongs to another user.");
  }
  await env.DB.prepare(
    `INSERT INTO favorite_groups (id, owner_id, name, items_json, created_at, updated_at)
     VALUES (?, ?, ?, ?, ?, ?)
     ON CONFLICT(id) DO UPDATE SET owner_id = excluded.owner_id, name = excluded.name, items_json = excluded.items_json, updated_at = excluded.updated_at`,
  ).bind(id, ownerId, name, JSON.stringify(items), createdAt, updatedAt).run();
  return jsonResponse(env, {ok: true});
}

async function deleteFavorite(env: Env, id: string, ownerId: string): Promise<Response> {
  requireStorage(env);
  await env.DB.prepare("DELETE FROM favorite_groups WHERE id = ? AND owner_id = ?").bind(id, ownerId).run();
  return jsonResponse(env, {ok: true});
}

async function handleCloudStorage(request: Request, env: Env, ownerId: string): Promise<Response | null> {
  const parts = new URL(request.url).pathname.split("/").filter(Boolean);
  if (parts[0] !== "api") return null;

  if (parts[1] === "gallery" && parts[2] === "health" && request.method === "GET") {
    requireStorage(env);
    return jsonResponse(env, {storage: "cloudflare", imageStore: "r2", database: "d1"});
  }
  if (parts[1] === "gallery" && parts[2] === "image" && parts[3] && request.method === "GET") {
    return serveGalleryImage(request, env, safeId(parts[3]), ownerId);
  }
  if (parts[1] === "gallery" && parts[2] === "records") {
    if (request.method === "GET") return galleryRecords(request, env, ownerId);
    if (request.method === "POST") return createGalleryRecord(request, env, ownerId);
    if (parts[3] && request.method === "PATCH") return patchGalleryRecord(request, env, safeId(parts[3]), ownerId);
  }
  if (parts[1] === "gallery" && parts[2] === "folders") {
    if (request.method === "GET") return galleryFolders(request, env, ownerId);
    if (request.method === "POST") return createGalleryFolder(request, env, ownerId);
    if (parts[3] && request.method === "PATCH") return patchGalleryFolder(request, env, safeId(parts[3]), ownerId);
    if (parts[3] && request.method === "DELETE") return deleteGalleryFolder(env, safeId(parts[3]), ownerId);
  }
  if (parts[1] === "favorites") {
    if (request.method === "GET") return favorites(request, env, ownerId);
    if (parts[2] && request.method === "PUT") return putFavorite(request, env, safeId(parts[2]), ownerId);
    if (parts[2] && request.method === "DELETE") return deleteFavorite(env, safeId(parts[2]), ownerId);
  }
  // Only gallery and favorites belong to the Worker. Generation, streaming,
  // and prompt-catalog endpoints must continue to the Vast backend.
  if (parts[1] === "gallery" || parts[1] === "favorites") {
    return jsonResponse(env, {error: "Cloud storage route not found."}, 404);
  }
  return null;
}

async function proxyToBackend(request: Request, env: Env, ownerId: string): Promise<Response> {
  if (!env.BACKEND_URL) return new Response("BACKEND_URL is not configured.", {status: 503, headers: corsHeaders(env)});
  if (!env.BACKEND_SHARED_SECRET) return new Response("BACKEND_SHARED_SECRET is not configured.", {status: 503, headers: corsHeaders(env)});
  const incoming = new URL(request.url);
  const upstream = new URL(env.BACKEND_URL);
  upstream.pathname = incoming.pathname;
  upstream.search = incoming.search;
  const headers = new Headers(request.headers);
  for (const name of ["host", "cf-connecting-ip", "cf-ipcountry", "cf-ray", "x-forwarded-proto", "x-backend-key", "x-app-user-id"]) headers.delete(name);
  headers.set("X-Backend-Key", env.BACKEND_SHARED_SECRET);
  headers.set("X-App-User-Id", ownerId);
  const upstreamRequest = new Request(upstream.toString(), {
    method: request.method,
    headers,
    body: request.method === "GET" || request.method === "HEAD" ? undefined : request.body,
    redirect: "manual",
  });
  const upstreamResponse = await fetch(upstreamRequest);
  const responseHeaders = new Headers(upstreamResponse.headers);
  for (const [key, value] of Object.entries(corsHeaders(env))) responseHeaders.set(key, value);
  responseHeaders.set("Cache-Control", "no-store");
  return new Response(upstreamResponse.body, {status: upstreamResponse.status, statusText: upstreamResponse.statusText, headers: responseHeaders});
}

export default {
  async fetch(request: Request, env: Env): Promise<Response> {
    if (request.method === "OPTIONS") return new Response(null, {status: 204, headers: corsHeaders(env)});
    try {
      const ownerId = await requireAuthenticatedUser(request, env);
      const cloudResponse = await handleCloudStorage(request, env, ownerId);
      if (cloudResponse) return cloudResponse;
      return await proxyToBackend(request, env, ownerId);
    } catch (error) {
      console.error(error);
      const status = error instanceof HttpError ? error.status : 500;
      return jsonResponse(env, {error: error instanceof Error ? error.message : "Worker request failed."}, status);
    }
  },
};

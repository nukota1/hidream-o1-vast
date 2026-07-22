export interface Env {
  BACKEND_URL: string;
  BACKEND_TOKEN?: string;
  ALLOWED_ORIGIN?: string;
}

function corsHeaders(env: Env): HeadersInit {
  return {
    "Access-Control-Allow-Origin": env.ALLOWED_ORIGIN || "*",
    "Access-Control-Allow-Methods": "GET,POST,OPTIONS",
    "Access-Control-Allow-Headers": "Content-Type,Authorization",
  };
}

export default {
  async fetch(request: Request, env: Env): Promise<Response> {
    if (request.method === "OPTIONS") {
      return new Response(null, { status: 204, headers: corsHeaders(env) });
    }

    if (!env.BACKEND_URL) {
      return new Response("BACKEND_URL is not configured.", { status: 500 });
    }

    const incoming = new URL(request.url);
    const upstream = new URL(env.BACKEND_URL);
    upstream.pathname = incoming.pathname;
    upstream.search = incoming.search;
    // Vast browser URLs may require a per-instance query token. Keep it only
    // in a Worker secret so it is never sent to the browser or committed.
    if (env.BACKEND_TOKEN) {
      upstream.searchParams.set("token", env.BACKEND_TOKEN);
    }

    const headers = new Headers(request.headers);
    headers.delete("host");
    headers.delete("cf-connecting-ip");
    headers.delete("cf-ipcountry");
    headers.delete("cf-ray");
    headers.delete("x-forwarded-proto");

    const upstreamRequest = new Request(upstream.toString(), {
      method: request.method,
      headers,
      body: request.body,
      redirect: "manual",
    });

    const upstreamResponse = await fetch(upstreamRequest);
    const responseHeaders = new Headers(upstreamResponse.headers);
    for (const [key, value] of Object.entries(corsHeaders(env))) {
      responseHeaders.set(key, value);
    }
    responseHeaders.set("Cache-Control", "no-store");

    return new Response(upstreamResponse.body, {
      status: upstreamResponse.status,
      statusText: upstreamResponse.statusText,
      headers: responseHeaders,
    });
  },
};

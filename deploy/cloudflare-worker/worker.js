function corsHeaders(env) {
  return {
    "Access-Control-Allow-Origin": env.ALLOWED_ORIGIN || "*",
    "Access-Control-Allow-Methods": "GET,POST,OPTIONS",
    "Access-Control-Allow-Headers": "Content-Type,Authorization",
  };
}

export default {
  async fetch(request, env) {
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
    if (env.BACKEND_TOKEN) {
      upstream.searchParams.set("token", env.BACKEND_TOKEN);
    }

    const headers = new Headers(request.headers);
    headers.delete("host");
    headers.delete("cf-connecting-ip");
    headers.delete("cf-ipcountry");
    headers.delete("cf-ray");
    headers.delete("x-forwarded-proto");

    const upstreamResponse = await fetch(new Request(upstream.toString(), {
      method: request.method,
      headers,
      body: request.body,
      redirect: "manual",
    }));
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

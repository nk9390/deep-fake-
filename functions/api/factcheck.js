// Cloudflare Pages Function: GET /api/factcheck?q=<claim>
//
// Proxies Google's Fact Check Tools API so the API key stays on the server (set it as the
// encrypted variable FACTCHECK_API_KEY in the Pages project settings). Mirrors
// summarize_claims() in cyber/factcheck.py.

const API_URL = "https://factchecktools.googleapis.com/v1alpha1/claims:search";
const MAX_QUERY_CHARS = 300;

const FALSE_RATINGS = /\b(false|fake|pants on fire|incorrect|inaccurate|misleading|no evidence|fabricated|hoax|satire|baseless|unproven|distorts|wrong|scam|altered|not true|four pinocchios)\b/i;
const TRUE_RATINGS = /\b(true|correct|accurate|mostly true|verified)\b/i;
const MIXED_RATINGS = /\b(half[- ]true|mixture|mixed|partly|partially|needs context|missing context|cherry[- ]picks)\b/i;

export function classifyRating(rating) {
  rating = rating || "";
  if (MIXED_RATINGS.test(rating)) return "mixed";
  if (FALSE_RATINGS.test(rating)) return "false";
  if (TRUE_RATINGS.test(rating)) return "true";
  return "mixed";
}

// Only http(s) links survive, so a javascript: URL in upstream data can't become a link.
function safeUrl(url) {
  return typeof url === "string" && /^https?:\/\//i.test(url) ? url : null;
}

function str(value, max = 1000) {
  return typeof value === "string" ? value.slice(0, max) : "";
}

export function summarizeClaims(data) {
  const counts = { false: 0, true: 0, mixed: 0 };
  const claims = (data.claims || []).map((claim) => ({
    text: str(claim.text),
    claimant: str(claim.claimant, 200),
    date: str(claim.claimDate, 40),
    reviews: (claim.claimReview || []).map((review) => {
      const verdict = classifyRating(review.textualRating);
      counts[verdict] += 1;
      const publisher = review.publisher || {};
      return {
        publisher: str(publisher.name || publisher.site, 200) || "unknown",
        rating: str(review.textualRating, 200),
        verdict,
        title: str(review.title, 500),
        url: safeUrl(review.url),
        date: str(review.reviewDate, 40),
      };
    }),
  }));
  return { claims, counts };
}

function json(body, status = 200, headers = {}) {
  return new Response(JSON.stringify(body), {
    status,
    headers: {
      "Content-Type": "application/json; charset=utf-8",
      "X-Content-Type-Options": "nosniff",
      ...headers,
    },
  });
}

export async function onRequestGet({ request, env }) {
  const params = new URL(request.url).searchParams;
  const query = (params.get("q") || "").trim().slice(0, MAX_QUERY_CHARS);
  const language = /^[a-z]{2}(-[A-Z]{2})?$/.test(params.get("lang") || "") ? params.get("lang") : "en";
  if (!query) return json({ error: "Type a headline or claim to check." }, 400);
  if (!env.FACTCHECK_API_KEY) {
    return json({ error: "Fact-checking isn't set up on this site yet: the FACTCHECK_API_KEY setting is missing." }, 503);
  }

  const upstream = new URL(API_URL);
  upstream.searchParams.set("query", query);
  upstream.searchParams.set("languageCode", language);
  upstream.searchParams.set("pageSize", "10");
  upstream.searchParams.set("key", env.FACTCHECK_API_KEY);

  let res;
  try {
    // Cache identical queries at Cloudflare's edge for an hour to save API quota.
    res = await fetch(upstream.toString(), { cf: { cacheTtl: 3600, cacheEverything: true } });
  } catch {
    return json({ error: "Couldn't reach the fact-check service. Try again in a minute." }, 502);
  }
  // Never forward the upstream body or URL: they can contain the key.
  if (!res.ok) return json({ error: `The fact-check service returned an error (HTTP ${res.status}).` }, 502);

  const summary = summarizeClaims(await res.json());
  return json({ query, ...summary }, 200, { "Cache-Control": "public, max-age=600" });
}

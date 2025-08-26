# main.py
import os
import feedparser
import requests
import pandas as pd
from datetime import datetime, timedelta, timezone
from dotenv import load_dotenv
from typing import List, Dict, Any
from tqdm import tqdm

# LLM client
from litellm import completion

# -------------------------
# Config
# -------------------------
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    raise ValueError("Missing GROQ_API_KEY in .env")

TWEETS_CSV = os.getenv("TWEETS_CSV_PATH", "tweets_dataset.csv")
YOUTUBE_CSV = os.getenv("YOUTUBE_CSV_PATH", "youtube_dataset.csv")

PRIMARY_MODEL = "groq/llama-3.3-70b-versatile"
FALLBACK_MODEL = "groq/llama-3.1-8b-instant"

SEC_CSV = "sec_filings.csv"
TWEETS_CSV_OUT = "twitter_posts.csv"
YOUTUBE_CSV_OUT = "youtube_posts.csv"
SENTIMENT_CSV = "sentiment_results.csv"
SENTIMENT_XLSX = "sentiment_results.xlsx"

# -------------------------
# Helpers: SEC Form 4 fetch
# -------------------------
def fetch_sec_filings(limit: int = 20) -> List[Dict[str, Any]]:
    url = "https://www.sec.gov/cgi-bin/browse-edgar"
    params = {"action": "getcurrent", "type": "4", "count": "100", "output": "atom"}
    headers = {"User-Agent": "Mozilla/5.0 (compatible; crewai-script/1.0; +contact@example.com)"}
    try:
        r = requests.get(url, params=params, headers=headers, timeout=30)
        r.raise_for_status()
    except Exception as e:
        print("⚠️ SEC request failed:", e)
        return []

    feed = feedparser.parse(r.text)
    two_days_ago = datetime.now(timezone.utc) - timedelta(hours=48)
    filings = []

    for entry in feed.entries:
        updated_raw = entry.get("updated") or entry.get("published") or ""
        filing_dt = None
        try:
            filing_dt = datetime.fromisoformat(updated_raw.replace("Z", "+00:00"))
        except Exception:
            if entry.get("updated_parsed"):
                tp = entry["updated_parsed"]
                filing_dt = datetime(*tp[:6], tzinfo=timezone.utc)
        if filing_dt is None:
            continue
        if filing_dt.tzinfo is None:
            filing_dt = filing_dt.replace(tzinfo=timezone.utc)
        filing_dt_utc = filing_dt.astimezone(timezone.utc)
        if filing_dt_utc < two_days_ago:
            continue

        summary = entry.get("summary", "")
        company = entry.get("title", "")
        link = entry.get("link", "")
        filings.append({
            "company": company,
            "link": link,
            "updated": filing_dt_utc.isoformat(),
            "summary": summary
        })
        if len(filings) >= limit:
            break

    if filings:
        pd.DataFrame(filings).to_csv(SEC_CSV, index=False)
        print(f"✅ Saved {len(filings)} SEC filings → {SEC_CSV}")
    else:
        print("⚠️ No recent SEC Form 4 filings found in last 48 hours.")

    return filings

# -------------------------
# Helpers: Read and normalize CSV (Twitter or YouTube)
# -------------------------
def read_csv_file(path: str, per_user_limit: int = 5):
    if not os.path.exists(path):
        print(f"⚠️ CSV not found: {path}, skipping...")
        return []

    try:
        # Only use first two columns and skip bad lines
        df = pd.read_csv(path, engine='python', usecols=[0,1], names=['user','content'], header=0)
    except Exception as e:
        print(f"⚠️ CSV error, skipping malformed lines: {e}")
        df = pd.read_csv(path, engine='python', error_bad_lines=False, usecols=[0,1], names=['user','content'], header=0)

    if df.empty:
        return []

    rows, counts = [], {}
    for _, r in df.iterrows():
        user = str(r['user'])
        counts.setdefault(user, 0)
        if counts[user] >= per_user_limit:
            continue
        counts[user] += 1
        rows.append({'user': user, 'content': str(r['content'])})

    normalized_path = os.path.splitext(path)[0] + "_normalized.csv"
    pd.DataFrame(rows).to_csv(normalized_path, index=False)
    print(f"✅ Normalized CSV saved → {normalized_path} ({len(rows)} rows)")
    return rows

# -------------------------
# Helpers: LLM calls with fallback
# -------------------------
def groq_completion(prompt: str, model: str = PRIMARY_MODEL, max_tokens: int = 1024) -> str:
    def _call(m):
        return completion(model=m, api_key=GROQ_API_KEY,
                          messages=[{"role": "user", "content": prompt}],
                          max_tokens=max_tokens)
    try:
        resp = _call(model)
        return resp["choices"][0]["message"]["content"]
    except Exception as e:
        msg = str(e)
        if "decommission" in msg.lower() or "model_decommissioned" in msg.lower() or "invalid_request_error" in msg.lower():
            try:
                print(f"⚠️ Primary model {model} failed; attempting fallback {FALLBACK_MODEL}...")
                resp = _call(FALLBACK_MODEL)
                return resp["choices"][0]["message"]["content"]
            except Exception as e2:
                return f"[LLM error fallback] {e2}"
        return f"[LLM error] {e}"

# -------------------------
# Process: Summarize SEC filings
# -------------------------
def summarize_sec(filings: List[Dict[str, Any]]) -> str:
    if not filings:
        return "No SEC filings to summarize."
    items = [f"- {f.get('company','')}\n  link: {f.get('link','')}\n  updated: {f.get('updated','')}\n" for f in filings[:20]]
    prompt = (
        "You are an expert financial analyst. Summarize the following recent SEC Form 4 insider filings for an investor:\n\n"
        + "\n".join(items)
        + "\n\nProvide 4-6 concise bullet points: notable insider buys/sells, companies to watch, and whether the activity is notable."
    )
    out = groq_completion(prompt)
    return out

# -------------------------
# Process: Summarize social posts (Twitter/YouTube)
# -------------------------
def summarize_posts(posts: List[Dict[str, Any]], platform: str = "X") -> str:
    if not posts:
        return f"No {platform} posts to summarize."

    compact_posts = [{"user": t["user"], "content": t["content"][:400]+"..." if len(t["content"])>400 else t["content"]} for t in posts]
    prompt = (
        f"You are an expert market/entertainment analyst. For {platform} posts below, "
        "label sentiment as 'positive', 'negative', or 'neutral', give 1-2 word main theme, "
        "and summarize overall in 3 sentences.\n\n"
        f"{compact_posts}\n\nRespond in JSON with keys: per_post (list of {{user, content, label, theme}}), overall (summary)."
    )
    out = groq_completion(prompt, max_tokens=1200)
    return out

# -------------------------
# Main pipeline
# -------------------------
def main():
    print("▶️ Starting pipeline...")

    # SEC filings
    filings = fetch_sec_filings(limit=20)
    sec_summary = summarize_sec(filings)
    print("\n📌 SEC Summary:\n", sec_summary)

    # Twitter
    tweets = read_csv_file(TWEETS_CSV, per_user_limit=5)
    tweet_summary = summarize_posts(tweets, platform="Twitter")
    print("\n📌 Twitter Summary:\n", tweet_summary)

    # YouTube
    youtube_posts = read_csv_file(YOUTUBE_CSV, per_user_limit=5)
    youtube_summary = summarize_posts(youtube_posts, platform="YouTube")
    print("\n📌 YouTube Summary:\n", youtube_summary)

    # Save final CSV
    summary_rows = [
        {"section": "sec_summary", "content": sec_summary},
        {"section": "twitter_summary", "content": tweet_summary},
        {"section": "youtube_summary", "content": youtube_summary},
    ]
    pd.DataFrame(summary_rows).to_csv(SENTIMENT_CSV, index=False)

    # Save XLSX
    with pd.ExcelWriter(SENTIMENT_XLSX) as writer:
        if filings:
            pd.DataFrame(filings).to_excel(writer, sheet_name="SEC Filings", index=False)
        if tweets:
            pd.DataFrame(tweets).to_excel(writer, sheet_name="Twitter", index=False)
        if youtube_posts:
            pd.DataFrame(youtube_posts).to_excel(writer, sheet_name="YouTube", index=False)
        pd.DataFrame(summary_rows).to_excel(writer, sheet_name="Summaries", index=False)

    print(f"\n✅ Done. Outputs: {SEC_CSV}, {TWEETS_CSV_OUT}, {YOUTUBE_CSV_OUT}, {SENTIMENT_CSV}, {SENTIMENT_XLSX}")


if __name__ == "__main__":
    main()

# Shlok_LiveNews_India_ML.py
import os
import sys
import requests
import time
from datetime import datetime, timezone, timedelta
from collections import deque
from flask import Flask, request, jsonify
from flask_cors import CORS

try:
	from zoneinfo import ZoneInfo
except ImportError:
	ZoneInfo = None

# -------------------------
# Configuration
# -------------------------
API_KEY = os.getenv("NEWSAPI_KEY", "ce516549b2a341febc179d5498ee4429").strip()
EVERYTHING_URL = "https://newsapi.org/v2/everything"

UPDATE_INTERVAL = 600  # 10 minutes
# Expanded to cover weather and advisories
KEYWORDS = [
	"tourism", "crime", "disaster", "festival", "event", "safety", "accident", "tourist",
	"advisory", "curfew", "section 144", "ban", "protest", "strike",
	"heatwave", "cold wave", "snow", "snowfall", "rain", "heavy rain", "flooding", "landslide",
	"temperature", "storm", "cyclone", "thunderstorm"
]
SOURCES = "the-hindu,times-of-india,indian-express,hindustan-times"

# -------------------------
# Timezone helper
# -------------------------

def get_ist_tz():
	if ZoneInfo:
		try:
			return ZoneInfo("Asia/Kolkata")
		except:
			pass
	return timezone(timedelta(hours=5, minutes=30))
IST = get_ist_tz()

def to_ist(iso_ts: str) -> str:
	try:
		dt = datetime.fromisoformat((iso_ts or "").replace("Z", "+00:00"))
		dt = dt.astimezone(IST)
		return dt.strftime("%Y-%m-%d %H:%M:%S %Z")
	except:
		return iso_ts or "N/A"

# -------------------------
# Risk classification and tags
# -------------------------

CRIME_TERMS = {"crime", "murder", "assault", "robbery", "theft", "kidnap", "rape", "violence"}
DISASTER_TERMS = {"earthquake", "flood", "storm", "disaster", "landslide", "cyclone", "tsunami", "accident", "crash", "stampede"}
ADVISORY_TERMS = {"advisory", "curfew", "section 144", "ban", "restriction", "protest", "strike", "alert", "warning"}
WEATHER_TERMS = {"heatwave", "cold wave", "snow", "snowfall", "rain", "heavy rain", "flooding", "temperature", "thunderstorm"}
JOY_TERMS = {"festival", "event", "tourist", "attraction", "celebration", "fair", "carnival", "opening", "reopen", "heritage"}


def compute_tags(title, description, content):
	text = " ".join(filter(None, [title, description, content])).lower()
	tags = set()
	if any(t in text for t in CRIME_TERMS):
		tags.add("crime")
	if any(t in text for t in DISASTER_TERMS):
		tags.add("disaster")
	if any(t in text for t in ADVISORY_TERMS):
		tags.add("advisory")
	if any(t in text for t in WEATHER_TERMS):
		tags.add("weather")
	if any(t in text for t in JOY_TERMS):
		tags.add("joy")
	return sorted(tags)


def classify_article(title, description, content):
	text = " ".join(filter(None, [title, description, content])).lower()
	# priority: crime > disaster > advisory/weather negative > joyful > other
	if any(k in text for k in CRIME_TERMS):
		return "Crime", 0.9
	elif any(k in text for k in DISASTER_TERMS):
		return "Disaster", 0.8
	elif any(k in text for k in ADVISORY_TERMS):
		return "Advisory", 0.7
	elif any(k in text for k in JOY_TERMS):
		return "Tourist-Friendly", 0.5
	else:
		return "Other", 0.2


def normalize_articles(articles):
	normalized = []
	for a in articles:
		risk, score = classify_article(a.get("title"), a.get("description"), a.get("content"))
		tags = compute_tags(a.get("title"), a.get("description"), a.get("content"))
		normalized.append({
			"title": a.get("title"),
			"source": (a.get("source") or {}).get("name"),
			"author": a.get("author"),
			"description": a.get("description"),
			"url": a.get("url"),
			"publishedAt": to_ist(a.get("publishedAt")),
			"content": a.get("content"),
			"risk_category": risk,
			"risk_score": score,
			"tags": tags
		})
	return normalized

# -------------------------
# Simple ML utilities (no heavy deps)
# -------------------------

POSITIVE_WORDS = {
	"festival", "celebration", "award", "record", "growth", "tourist", "safe", "security",
	"improve", "opening", "reopen", "heritage", "clean", "win", "success", "happy", "enjoy"
}
NEGATIVE_WORDS = {
	"crime", "murder", "assault", "robbery", "theft", "kidnap", "riot", "curfew",
	"flood", "landslide", "storm", "earthquake", "heatwave", "coldwave", "drought", "accident",
	"crash", "stampede", "violence", "terror", "alert", "warning"
}

INDIA_STATES = {
	"andhra pradesh", "arunachal pradesh", "assam", "bihar", "chhattisgarh", "goa", "gujarat",
	"haryana", "himachal pradesh", "jharkhand", "karnataka", "kerala", "madhya pradesh", "maharashtra",
	"manipur", "meghalaya", "mizoram", "nagaland", "odisha", "punjab", "rajasthan", "sikkim",
	"tamil nadu", "telangana", "tripura", "uttar pradesh", "uttarakhand", "west bengal",
	"delhi", "jammu and kashmir", "ladakh", "puducherry", "andaman and nicobar islands",
	"lakshadweep", "chandigarh", "dadra and nagar haveli and daman and diu"
}

CITY_HINTS = {
	"mumbai", "delhi", "bengaluru", "bangalore", "chennai", "kolkata", "hyderabad", "pune",
	"ahmedabad", "jaipur", "lucknow", "kanpur", "nagpur", "indore", "thane", "bhopal", "patna",
	"visakhapatnam", "vadodara", "ghaziabad", "ludhiana", "agra", "nashik", "faridabad", "meerut",
	"rajkot", "varanasi", "srinagar", "ranchi", "amritsar", "allahabad", "prayagraj", "coimbatore",
	"kochi", "trivandrum", "guwahati", "shillong", "imphal", "aizawl", "kohima", "itanagar",
	"gangtok", "panaji", "shimla", "dehradun", "udaipur", "jodhpur", "udaipur", "mysuru", "ooty",
	"darjeeling", "goa","sion"
}

# Basic place extraction using keyword hints

def extract_places(record):
	text = " ".join(filter(None, [record.get("title"), record.get("description"), record.get("content")])).lower()
	found = set()
	for state in INDIA_STATES:
		if state in text:
			found.add(state)
	for city in CITY_HINTS:
		if city in text:
			found.add(city)
	# If explicitly tagged (e.g., via search query), include it
	forced = (record.get("forced_place") or "").lower().strip()
	if forced:
		found.add(forced)
	# If no place found, fallback to source hint
	if not found and record.get("source"):
		src = (record.get("source") or "").lower()
		for state in INDIA_STATES:
			if state in src:
				found.add(state)
	return sorted(found)

# Very light sentiment scoring

def simple_sentiment_score(record):
	text = " ".join(filter(None, [record.get("title"), record.get("description"), record.get("content")])).lower()
	score = 0
	for w in POSITIVE_WORDS:
		if w in text:
			score += 1
	for w in NEGATIVE_WORDS:
		if w in text:
			score -= 1
	return max(-5, min(5, score))

# Optional: fetch current temperature via Open-Meteo (lat/long hardcoded for major cities)
CITY_COORDS = {
	"mumbai": (19.0760, 72.8777),
	"delhi": (28.6139, 77.2090),
	"bengaluru": (12.9716, 77.5946),
	"bangalore": (12.9716, 77.5946),
	"chennai": (13.0827, 80.2707),
	"kolkata": (22.5726, 88.3639),
	"hyderabad": (17.3850, 78.4867),
	"pune": (18.5204, 73.8567),
	"jaipur": (26.9124, 75.7873),
	"goa": (15.2993, 74.1240),
	"guwahati": (26.1445, 91.7362),
	"shillong": (25.5788, 91.8933)
}


def fetch_temperature(place: str):
	place_l = (place or "").lower()
	coords = CITY_COORDS.get(place_l)
	if not coords:
		return None
	lat, lon = coords
	try:
		resp = requests.get(
			"https://api.open-meteo.com/v1/forecast",
			params={"latitude": lat, "longitude": lon, "current_weather": True},
			timeout=10,
		)
		resp.raise_for_status()
		data = resp.json()
		cur = (data.get("current_weather") or {}).get("temperature")
		return cur
	except Exception:
		return None

# -------------------------
# In-memory index and aggregation
# -------------------------

PLACE_INDEX = {}
# structure: {
#   place: {
#     "articles": [record, ... (recent N)],
#     "scores": {
#         "risk": float,
#         "sentiment": float,
#         "temperature": float|None,
#         "last_updated": iso
#     }
#   }
# }
MAX_ARTICLES_PER_PLACE = 50


def update_place_index(records):
	global PLACE_INDEX
	for r in records:
		places = extract_places(r)
		sent = simple_sentiment_score(r)
		for p in places or ["india"]:
			bucket = PLACE_INDEX.setdefault(p, {"articles": deque(maxlen=MAX_ARTICLES_PER_PLACE), "scores": {}})
			bucket["articles"].appendleft(r)
			# aggregate scores
			risk_val = 1.0 if r.get("risk_category") in ("Crime", "Disaster") else (0.3 if r.get("risk_category") == "Other" else 0.6)
			scores = bucket["scores"]
			prev_risk = scores.get("risk", 0.5)
			prev_sent = scores.get("sentiment", 0.0)
			# exponential moving average
			scores["risk"] = 0.7 * prev_risk + 0.3 * risk_val
			scores["sentiment"] = 0.7 * prev_sent + 0.3 * (sent / 5.0)
			scores["last_updated"] = datetime.now(IST).strftime("%Y-%m-%d %H:%M:%S %Z")


def get_place_snapshot(place: str):
	key = (place or "").lower()
	b = PLACE_INDEX.get(key)
	if not b:
		return None
	# attempt temperature on demand
	temp = fetch_temperature(key)
	if temp is not None:
		b["scores"]["temperature"] = temp
	return {
		"place": key,
		"scores": b.get("scores", {}),
		"articles": list(b.get("articles", []))[:10]
	}


def recommend_place(place: str):
	snap = get_place_snapshot(place)
	if not snap:
		return {
			"place": place,
			"decision": "insufficient-data",
			"reason": "No recent articles indexed for this place.",
			"scores": {}
		}
	s = snap["scores"]
	risk = s.get("risk", 0.5)
	sent = s.get("sentiment", 0.0)
	# simple rule-based decision combining risk (low better) and sentiment (high better)
	score = (1.0 - risk) * 0.6 + (0.5 + sent / 2) * 0.4
	decision = "go" if score >= 0.55 else ("caution" if score >= 0.45 else "avoid")
	reasons = []
	if risk >= 0.7:
		reasons.append("elevated recent risk signals")
	elif risk <= 0.4:
		reasons.append("low recent risk signals")
	if sent >= 0.2:
		reasons.append("positive tourism-related sentiment")
	elif sent <= -0.2:
		reasons.append("negative tourism-related sentiment")
	if s.get("temperature") is not None:
		temp = s.get("temperature")
		if temp >= 38:
			reasons.append("very hot weather")
		elif temp <= 8:
			reasons.append("very cold weather")
	return {
		"place": snap["place"],
		"decision": decision,
		"score": round(score, 3),
		"scores": s,
		"top_articles": [
			{"title": a.get("title"), "url": a.get("url"), "risk": a.get("risk_category"), "tags": a.get("tags")}
			for a in snap["articles"]
			if set(a.get("tags", [])) & {"crime", "disaster", "advisory", "weather", "joy"}
		][:10],
		"reasons": reasons
	}

# -------------------------
# Fetch news from API
# -------------------------

def call_newsapi(params):
	headers = {"X-Api-Key": API_KEY}
	try:
		resp = requests.get(EVERYTHING_URL, params=params, headers=headers, timeout=20)
		resp.raise_for_status()
	except requests.exceptions.RequestException as e:
		print("Network/API error:", e)
		return []
	data = resp.json()
	if data.get("status") != "ok":
		print("API returned error:", data)
		return []
	return data.get("articles", [])

# One-off fetch for a specific place

def fetch_place_once(place: str, hours: int = 48):
	q = f'"{place}" AND (' + " OR ".join(KEYWORDS) + ")"
	from_ts = (datetime.now(timezone.utc) - timedelta(hours=hours)).isoformat()
	params = {
		"q": q,
		"language": "en",
		"sortBy": "publishedAt",
		"pageSize": 50,
		"from": from_ts
	}
	articles = call_newsapi(params)
	if not articles and SOURCES:
		params["sources"] = SOURCES
		articles = call_newsapi(params)
	if not articles:
		return 0
	normalized = normalize_articles(articles)
	forced = (place or "").lower().strip()
	for r in normalized:
		r["forced_place"] = forced
	update_place_index(normalized)
	return len(normalized)

# -------------------------
# Live India news fetcher
# -------------------------

def live_news_fetcher():
	seen_urls = deque(maxlen=2000)  # last 2000 URLs to avoid duplicates

	while True:
		query = "India AND (" + " OR ".join(KEYWORDS) + ")"
		params = {
			"q": query,
			"language": "en",
			"sortBy": "publishedAt",
			"pageSize": 50,
			"sources": SOURCES
		}

		articles = call_newsapi(params)

		# Filter duplicates
		new_articles = []
		for a in articles:
			url = a.get("url")
			if url and url not in seen_urls:
				seen_urls.append(url)
				new_articles.append(a)

		if not new_articles:
			print(f"[{datetime.now(IST)}] No new India articles found.")
		else:
			normalized = normalize_articles(new_articles)
			print(f"[{datetime.now(IST)}] Fetched {len(normalized)} new India articles:")
			for i, r in enumerate(normalized[:10], 1):  # show top 10 for monitoring
				print(f"{i}. {r['title']} [{r['risk_category']}] tags={r.get('tags')}")

			# feed ML index
			update_place_index(normalized)

		print(f"Waiting {UPDATE_INTERVAL/60} minutes for next update...\n")
		time.sleep(UPDATE_INTERVAL)

# -------------------------
# CLI helpers
# -------------------------

def print_recommendation(place: str):
	res = recommend_place(place)
	if res.get("decision") == "insufficient-data":
		print("No data found. Fetching latest news for this place...")
		added = fetch_place_once(place)
		if added > 0:
			res = recommend_place(place)
		else:
			print("Still no news found for this place in the recent window.")
			return
	print(f"Place: {res.get('place')}")
	print(f"Decision: {res.get('decision').upper()} (score={res.get('score')})")
	s = res.get("scores", {})
	print(f"Risk={round(s.get('risk', 0.0),3)} Sentiment={round(s.get('sentiment',0.0),3)} Temp={s.get('temperature')}")
	print("Tourist-relevant news:")
	for a in res.get("top_articles", [])[:5]:
		print(f" - {a['title']} [{a['risk']}] {a.get('tags')} -> {a['url']}")
	if res.get("reasons"):
		print("Reasons:")
		for r in res["reasons"]:
			print(f" * {r}")


def print_search(place: str):
	snap = get_place_snapshot(place)
	if not snap:
		print("No data for this place yet. Fetching latest news...")
		added = fetch_place_once(place)
		if added == 0:
			print("Still no news found for this place in the recent window.")
			return
		snap = get_place_snapshot(place)
	relevant = [
		{"title": a.get("title"), "url": a.get("url"), "risk": a.get("risk_category"), "tags": a.get("tags")}
		for a in snap["articles"]
		if set(a.get("tags", [])) & {"crime", "disaster", "advisory", "weather", "joy"}
	][:10]
	print(f"Latest for {snap['place']} (risk={round(snap['scores'].get('risk',0.0),3)}, sentiment={round(snap['scores'].get('sentiment',0.0),3)})")
	if not relevant:
		print("No tourist-relevant news items found yet.")
	for a in relevant:
		print(f" - {a['title']} [{a['risk']}] {a.get('tags')} -> {a['url']}")

# -------------------------
# Run
# -------------------------
app = Flask(__name__)
CORS(app)  # Allow requests from your frontend

@app.route("/api/place", methods=["GET"])
def api_place():
    place = request.args.get("place", "")
    if not place:
        return jsonify({"error": "Missing place parameter"}), 400
    res = recommend_place(place)
    if res.get("decision") == "insufficient-data":
        fetch_place_once(place)
        res = recommend_place(place)
    return jsonify(res)

if __name__ == "__main__":
    if not API_KEY or API_KEY.lower().startswith("your_"):
        print("⚠️  Missing API key. Set NEWSAPI_KEY env var or edit the API_KEY variable.")
        sys.exit(1)
    # CLI modes:
    # python call.py fetch        -> run live fetcher loop
    # python call.py recommend goa -> compute recommendation for place
    # python call.py search goa   -> show latest 10 tourist-relevant articles for place
    if len(sys.argv) >= 2:
        cmd = sys.argv[1].lower()
        if cmd == "fetch":
            live_news_fetcher()
        elif cmd == "recommend" and len(sys.argv) >= 3:
            print_recommendation(" ".join(sys.argv[2:]))
        elif cmd == "search" and len(sys.argv) >= 3: 
            print_search(" ".join(sys.argv[2:]))
        else:
            print("Usage: python call.py [fetch|recommend <place>|search <place>]")
    else:
        # Start Flask API if no CLI args
        app.run(host="0.0.0.0", port=5000, debug=True)
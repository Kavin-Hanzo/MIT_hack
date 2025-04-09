import os
import logging
from serpapi import GoogleSearch

# Setup
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger()

# SerpAPI Key
SERP_API_KEY = "9b8d363edd68fae877cecb70f7bcccc48ec02d14b308fe71470b981062d7d0e1"  # Replace with your real key
os.environ["SERPAPI_API_KEY"] = SERP_API_KEY

# Trusted sources
TRUSTED_DOMAINS = [
    "mayoclinic.org", "webmd.com", "nhs.uk", "healthline.com",
    "clevelandclinic.org", "who.int", "medlineplus.gov", "medicalnewstoday.com"
]

# Queries to fetch for each category
CATEGORY_QUERIES = {
    "Illness overview": "{} overview",
    "Home remedies": "{} home remedies",
    "Doctor advice": "{} when to see a doctor"
}


def fetch_category_result(category_query):
    params = {
        "engine": "google",
        "q": category_query,
        "num": 10,
        "api_key": os.environ["SERPAPI_API_KEY"],
    }
    search = GoogleSearch(params)
    results = search.get_dict()

    if "organic_results" not in results:
        return None

    for res in results["organic_results"]:
        link = res.get("link", "")
        if any(domain in link for domain in TRUSTED_DOMAINS):
            return {
                "title": res.get("title", ""),
                "snippet": res.get("snippet", ""),
                "url": link
            }
    return None


def analyze_health_query_smart(query):
    categorized_info = {}

    for category, query_format in CATEGORY_QUERIES.items():
        full_query = query_format.format(query)
        logger.info(f"Fetching: {full_query}")
        result = fetch_category_result(full_query)

        if result:
            combined = f"{result['title']}. {result['snippet']}"
            categorized_info[category] = {
                "snippet": combined,
                "source": result['url']
            }
        else:
            categorized_info[category] = {
                "snippet": "Sorry, I couldn't find trusted info for this section.",
                "source": None
            }

    return categorized_info


def display_results(health_info, query):
    print("\n" + "=" * 60)
    print(f"🩺 Health Insight for: '{query}'")
    print("=" * 60)

    for category, info in health_info.items():
        print(f"\n🔹 {category}:")
        if info["source"]:
            print(f"📌 {info['snippet']} (🔗 [source]({info['source']}))")
        else:
            print(f"📌 {info['snippet']}")

    print("\nℹ️ This is general information. Please consult a doctor for personalized medical advice.")


# === RUN ===
query = input("Enter your health query (e.g., 'knee pain'): ").strip()
result = analyze_health_query_smart(query)
display_results(result, query)

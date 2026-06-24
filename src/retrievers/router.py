# # retriever/router.py
import logging
from llm.llm_client import get_client, get_model_name, active_model, LLMProvider

logger = logging.getLogger(__name__)

COLLECTION_DESCRIPTIONS = {
    "jobs":              "Job listings, company info, requirements, skills, location, job type",
    "applyjobs":         "Job applications, cover letters, applicant status, portfolio links",
    "blogs":             "Blog posts, articles, environment, technology, summaries and full content",
    "courseideas":       "User-submitted course ideas and suggestions, skill levels, key topics",
    "courses":           "Published courses available to enroll in, with price, duration, category, lessons",
    "joinmentorcoaches": "Mentor and coach profiles, expertise, availability, experience",
    "reviews":           "Reviews and ratings for events, courses, mentors, and services",
    "media":             "Media files, PDFs, webinar slides, videos and descriptions",
    "events":            "Upcoming and past events, workshops, bootcamps, career fairs, dates, registration, pricing",
    "subscriptionplans": "Platform subscription plans, pricing, billing, features, trial periods",
}
COLLECTION_NAMES = list(COLLECTION_DESCRIPTIONS.keys())

COLLECTION_LABELS = {
    "jobs":              "Job listings",
    "applyjobs":         "Job applications",
    "blogs":             "Articles & blog posts",
    "courseideas":       "Course ideas & suggestions",
    "courses":           "Available courses",
    "joinmentorcoaches": "Mentors & coaches",
    "reviews":           "Reviews & ratings",
    "media":             "Media & resources",
    "events":            "Events & workshops",
    "subscriptionplans": "Subscription plans",
}


def _build_router_prompt(query: str) -> str:
    collection_list = "\n".join(
        f"- {name}: {desc}"
        for name, desc in COLLECTION_DESCRIPTIONS.items()
    )
    return f"""You are a routing assistant for a RAG system.

Read the user query and return the 1-3 most relevant collection names, comma-separated.

Rules:
- Return ONLY collection names from the list below, comma-separated. Nothing else.
- Return the most relevant first.
- For broad queries spanning multiple areas, return up to 3 names.
- Distinguish: "courseideas" = user suggestions; "courses" = published enrollable courses.
- If genuinely unsure, return: all

Available collections:
{collection_list}

User query: {query}

Collection names:"""


def route_query(query: str) -> list[str]:
    """Route a query to 1-3 relevant collections."""
    try:
        client     = get_client()
        model_name = get_model_name()
        prompt     = _build_router_prompt(query)

        # OpenAI and Groq both use the chat completions interface
        if active_model in (LLMProvider.OPENAI, LLMProvider.GROQ):
            response = client.chat.completions.create(
                model=model_name,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=20,
                temperature=0.0,
            )
            raw = response.choices[0].message.content.strip().lower()

        else:  # HuggingFace
            raw = client.text_generation(
                prompt,
                model=model_name,
                max_new_tokens=20,
                temperature=0.1,
                stop_sequences=["\n"],
            ).strip().lower()

        logger.info(f"Router raw response: '{raw}'")

        candidates  = [c.strip() for c in raw.split(",")]
        collections = [c for c in candidates if c in COLLECTION_NAMES]

        if not collections:
            logger.warning(f"Router returned no valid collections ('{raw}') — searching all")
            return COLLECTION_NAMES

        logger.info(f"Routed '{query[:60]}' → {collections}")
        return collections

    except Exception as e:
        logger.error(f"Router failed: {e} — falling back to all collections")
        return COLLECTION_NAMES
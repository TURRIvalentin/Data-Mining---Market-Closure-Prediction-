"""
Categorización heurística de mercados Polymarket en 7 grupos.

Reglas auditables — orden de evaluación importa:
  Sports > Politics > Crypto > Finance > Tech > Entertainment > Other

Principios de diseño:
- Finance se evalúa ANTES que Tech para que "Apple hit $240" → Finance, no Tech.
- Todos los keywords se evalúan en lowercase.
- Algunos keywords incluyen espacios para evitar falsos positivos:
    "nfl "  (no "nfl") para no capturar "nflx" (ticker de Netflix)
    " bnb"  (no "bnb") para no capturar "airbnb"
"""

_CAT_RULES: list[tuple[str, list[str]]] = [
    ("Sports", [
        # Ligas y deportes
        "nba", "nhl", "nfl ", " nfl)", "mlb", "mls", "ufc", "mma",
        "soccer", "basketball", "baseball", "hockey", "tennis",
        "golf", "formula 1", "f1 ", "formula1", "cycling",
        # Competencias
        "serie a", "premier league", "bundesliga", "la liga", "ligue 1",
        "champions league", "europa league", "world cup", "olympics",
        "super bowl", "superbowl", "stanley cup",
        # Formato de mercado
        "playoffs", "trophy", "tournament", "win the match", "win on ",
        "finish in the top", "league cup", "fa cup", "copa del",
        "both teams to score", "end in a draw", "win on penalties",
        # Ligas regionales y equipos que caían en Other
        "super lig", "trabzonspor", "fenerbahce", "galatasaray",
        "eastern conference", "western conference",
        # Premios deportivos (NHL awards que caían en Other: 30 porteros × Vezina, etc.)
        "vezina", "jack adams award", "hart trophy", "conn smythe",
        "calder trophy", "norris trophy", "selke trophy", "lady byng",
    ]),
    ("Politics", [
        # Política electoral y de gobierno
        "election", "president", "senate", "congress", "parliament",
        "democrat", "republican", "trump", "biden", "harris", "zelensky",
        "vote", "candidate", "governor", "legislation",
        "white house", "prime minister", "supreme court", "nato",
        # Personas políticas
        "netanyahu", "khamenei", "modi ", "macron ", "scholz", "meloni",
        "gabbard", "tulsi ", "rubio ", "warsh",
        # Geopolítica y militar — captura los ~92 mercados que caían en Other
        "russia ", "russian ", "ukraine", "ukrainian",
        "israel", "israeli", "iran ", "iranian", "gaza",
        "hamas", "hezbollah", "houthi", "hormuz",
        "ceasefire", "ground operation", "military action",
        "warship", "strait of", "troops", "offensive",
        "saudi arabia", "yemen", "syria", "north korea",
        "taiwan ", "nuclear deal", "sanctions on",
        "regime fall", "declares independence",
        "peace deal", "peace talks", "diplomatic",
        "cuba ", "us x cuba", "krg ",
    ]),
    ("Crypto", [
        "bitcoin", "btc", "ethereum", "eth", "solana", " sol ",
        "xrp", "dogecoin", "doge", " bnb",   # espacios para evitar "solar", "airbnb"
        "crypto", "defi", "blockchain",
        "nft", "coinbase", "binance", "usdc", "usdt",
        "hyperliquid", "megaeth", "sui ", "avalanche", "polygon",
    ]),
    ("Finance", [
        # Mercados financieros
        "stock", "nasdaq", "s&p", "ipo", "earnings",
        "federal reserve", "fed rate", "inflation", "gdp",
        "recession", "interest rate", "market cap",
        "hit $", "reach $", "above $", "below $", "dip to $", "hit (",
        "nyse", " shares", "trading at",
        # Bancos centrales — captura los ~37 mercados que caían en Other
        "ecb", "bank of canada", "bank of england", "bank of japan",
        "bank of brazil", "bank of colombia", "bank of mexico",
        "bank of korea", "bank of israel",
        "european central bank", "reserve bank",
        "federal open market", "fomc",
        "basis point", " bps ", "bps decrease", "bps increase",
        "selic", "rate decision", "rate cut", "rate hike",
        "rate pause", "rate increase", "rate decrease",
        "central bank of", "the fed ", "fed decision", "fed chair",
        "fed cut", "fed pause",
        # Activos no financieros pero con precio
        "crude oil", "gold price", "silver price",
        "median home", "home value", "housing price",
    ]),
    ("Tech", [
        "openai", "anthropic", "gpt-", "chatgpt", "nvidia",
        "apple", "google", "microsoft", "meta ", "amazon",
        "tesla", "spacex", "elon musk", "tweets", " ai ",
        "iphone", "android", "starship", "cloudflare",
        # Modelos de IA que caían en Other
        "gemini ", "deepseek", "grok", "llama ", "claude ",
        "x money",
    ]),
    ("Entertainment", [
        "oscar", "grammy", "emmy", "box office", "album",
        "movie", "film", "netflix", "disney", "celebrity",
        "actor", "actress",
        # Charts musicales y festivales — captura los ~35 mercados que caían en Other
        "spotify", "monthly listeners", "billboard",
        "music festival", "todo mundo", "coachella", "lollapalooza",
        "concert", "world tour",
    ]),
]


def infer_category_coarse(question: str) -> str:
    """Categoriza un mercado en 7 grupos usando keyword matching sobre el texto de la pregunta."""
    q = question.lower()
    for category, keywords in _CAT_RULES:
        for kw in keywords:
            if kw in q:
                return category
    return "Other"

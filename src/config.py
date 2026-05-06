RANDOM_SEED = 42

# Ventana de observación (días desde apertura del mercado)
OBSERVATION_WINDOW_DAYS = 7

# Filtros de dataset
MIN_MARKET_DURATION_DAYS = 30
DATE_START = "2023-01-01"
MIN_PRICE_POINTS = 3   # mínimo de puntos de precio en la ventana de 7 días para incluir en el dataset

# Split temporal
TRAIN_RATIO = 0.70
VAL_RATIO = 0.10
TEST_RATIO = 0.20

# API
GAMMA_API_BASE = "https://gamma-api.polymarket.com"
CLOB_API_BASE = "https://clob.polymarket.com"
API_PAGE_SIZE = 100
API_RATE_LIMIT_PAUSE = 1.0  # segundos entre requests

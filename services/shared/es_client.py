import logging
from elasticsearch import Elasticsearch
from shared.config import ElasticsearchConfig

logger = logging.getLogger(__name__)


def create_es_client(es_config: ElasticsearchConfig) -> Elasticsearch:
    """Create an Elasticsearch client from config."""
    kwargs: dict = {
        "request_timeout": 30,
        "retry_on_timeout": True,
        "max_retries": 3,
    }

    if es_config.api_key:
        kwargs["api_key"] = es_config.api_key
    elif es_config.username and es_config.password:
        kwargs["basic_auth"] = (es_config.username, es_config.password)

    return Elasticsearch(es_config.host, **kwargs)


def check_es_connection(client: Elasticsearch, index: str) -> bool:
    """Return True if ES is reachable and the index exists."""
    try:
        if not client.ping():
            logger.error("ES ping failed")
            return False
        if not client.indices.exists(index=index):
            logger.warning(f"ES index '{index}' not found")
            return False
        return True
    except Exception as e:
        logger.error(f"ES connection check failed: {e}")
        return False

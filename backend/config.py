import os
import chromadb
import logging
import sys
from chromadb import Settings
from base64 import b64encode
from bs4 import BeautifulSoup
from typing import TypeVar, Generic, Union
from pathlib import Path
import json
import yaml

import markdown
import requests
import shutil

from secrets import token_bytes
from constants import ERROR_MESSAGES


try:
    from dotenv import load_dotenv, find_dotenv

    load_dotenv(find_dotenv("../.env"))
except ImportError:
    print("dotenv not installed, skipping...")

WEBUI_NAME = "Techpeek AI"
shutil.copyfile("../build/favicon.png", "./static/favicon.png")


from authlib.integrations.starlette_client import OAuth

oauth = OAuth()

oauth.register(
    name='google',
    client_id='763749380527-s493gmjqulracsgaa4ae4umr2pj8k7kv.apps.googleusercontent.com',
    client_secret='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',
    authorize_url='https://accounts.google.com/o/oauth2/v2/auth',
    authorize_params=None,
    access_token_url='https://www.googleapis.com/oauth2/v4/token',
    access_token_params=None,
    refresh_token_url=None,
    redirect_uri='https://chat.techpeek.in',
    client_kwargs={
        'scope': 'openid email profile',
        'token_endpoint_auth_method': 'client_secret_post'
    },
)


# Add this to your config.py or equivalent configuration file

MAIL_CONFIG = {
    "MAIL_USERNAME": "Ronit Virwani",
    "MAIL_PASSWORD": "ronit@0330",
    "MAIL_FROM": "ronitvirwani1@gmail.com",
    "MAIL_PORT": 465,  
    "MAIL_SERVER": "smtp.gmail.com",
    "MAIL_STARTTLS": False,  
    "MAIL_SSL_TLS": True,  
    "USE_CREDENTIALS": True,
    "VALIDATE_CERTS": True
}

####################################
# ENV (dev,test,prod)
####################################

ENV = os.environ.get("ENV", "dev")

try:
    with open(f"../package.json", "r") as f:
        PACKAGE_DATA = json.load(f)
except:
    PACKAGE_DATA = {"version": "0.0.0"}

VERSION = PACKAGE_DATA["version"]


# Function to parse each section
def parse_section(section):
    items = []
    for li in section.find_all("li"):
        # Extract raw HTML string
        raw_html = str(li)

        # Extract text without HTML tags
        text = li.get_text(separator=" ", strip=True)

        # Split into title and content
        parts = text.split(": ", 1)
        title = parts[0].strip() if len(parts) > 1 else ""
        content = parts[1].strip() if len(parts) > 1 else text

        items.append({"title": title, "content": content, "raw": raw_html})
    return items


try:
    with open("../CHANGELOG.md", "r") as file:
        changelog_content = file.read()
except:
    changelog_content = ""

# Convert markdown content to HTML
html_content = markdown.markdown(changelog_content)

# Parse the HTML content
soup = BeautifulSoup(html_content, "html.parser")

# Initialize JSON structure
changelog_json = {}

# Iterate over each version
for version in soup.find_all("h2"):
    version_number = version.get_text().strip().split(" - ")[0][1:-1]  # Remove brackets
    date = version.get_text().strip().split(" - ")[1]

    version_data = {"date": date}

    # Find the next sibling that is a h3 tag (section title)
    current = version.find_next_sibling()

    while current and current.name != "h2":
        if current.name == "h3":
            section_title = current.get_text().lower()  # e.g., "added", "fixed"
            section_items = parse_section(current.find_next_sibling("ul"))
            version_data[section_title] = section_items

        # Move to the next element
        current = current.find_next_sibling()

    changelog_json[version_number] = version_data


CHANGELOG = changelog_json


####################################
# LOGGING
####################################
log_levels = ["CRITICAL", "ERROR", "WARNING", "INFO", "DEBUG"]

GLOBAL_LOG_LEVEL = os.environ.get("GLOBAL_LOG_LEVEL", "").upper()
if GLOBAL_LOG_LEVEL in log_levels:
    logging.basicConfig(stream=sys.stdout, level=GLOBAL_LOG_LEVEL, force=True)
else:
    GLOBAL_LOG_LEVEL = "INFO"

log = logging.getLogger(__name__)
log.info(f"GLOBAL_LOG_LEVEL: {GLOBAL_LOG_LEVEL}")

log_sources = [
    "AUDIO",
    "COMFYUI",
    "CONFIG",
    "DB",
    "IMAGES",
    "LITELLM",
    "MAIN",
    "MODELS",
    "OLLAMA",
    "OPENAI",
    "RAG",
    "WEBHOOK",
]

SRC_LOG_LEVELS = {}

for source in log_sources:
    log_env_var = source + "_LOG_LEVEL"
    SRC_LOG_LEVELS[source] = os.environ.get(log_env_var, "").upper()
    if SRC_LOG_LEVELS[source] not in log_levels:
        SRC_LOG_LEVELS[source] = GLOBAL_LOG_LEVEL
    log.info(f"{log_env_var}: {SRC_LOG_LEVELS[source]}")

log.setLevel(SRC_LOG_LEVELS["CONFIG"])


####################################
# CUSTOM_NAME
####################################

CUSTOM_NAME = os.environ.get("CUSTOM_NAME", "")
if CUSTOM_NAME:
    try:
        r = requests.get(f"https://api.openwebui.com/api/v1/custom/{CUSTOM_NAME}")
        data = r.json()
        if r.ok:
            if "logo" in data:
                url = (
                    f"https://api.openwebui.com{data['logo']}"
                    if data["logo"][0] == "/"
                    else data["logo"]
                )

                r = requests.get(url, stream=True)
                if r.status_code == 200:
                    with open("./static/favicon.png", "wb") as f:
                        r.raw.decode_content = True
                        shutil.copyfileobj(r.raw, f)

            WEBUI_NAME = data["name"]
    except Exception as e:
        print(e)
        pass


####################################
# DATA/FRONTEND BUILD DIR
####################################

DATA_DIR = str(Path(os.getenv("DATA_DIR", "./data")).resolve())
FRONTEND_BUILD_DIR = str(Path(os.getenv("FRONTEND_BUILD_DIR", "../build")))

try:
    with open(f"{DATA_DIR}/config.json", "r") as f:
        CONFIG_DATA = json.load(f)
except:
    CONFIG_DATA = {}


####################################
# Static DIR
####################################

STATIC_DIR = str(Path(os.getenv("STATIC_DIR", "./static")).resolve())

shutil.copyfile(f"{FRONTEND_BUILD_DIR}/favicon.png", f"{STATIC_DIR}/favicon.png")


####################################
# File Upload DIR
####################################

UPLOAD_DIR = f"{DATA_DIR}/uploads"
Path(UPLOAD_DIR).mkdir(parents=True, exist_ok=True)


####################################
# Cache DIR
####################################

CACHE_DIR = f"{DATA_DIR}/cache"
Path(CACHE_DIR).mkdir(parents=True, exist_ok=True)


####################################
# Docs DIR
####################################

DOCS_DIR = f"{DATA_DIR}/docs"
Path(DOCS_DIR).mkdir(parents=True, exist_ok=True)

####################################
# Config helpers
####################################


def save_config():
    try:
        with open(f"{DATA_DIR}/config.json", "w") as f:
            json.dump(CONFIG_DATA, f, indent="\t")
    except Exception as e:
        log.exception(e)


def get_config_value(config_path: str):
    path_parts = config_path.split(".")
    cur_config = CONFIG_DATA
    for key in path_parts:
        if key in cur_config:
            cur_config = cur_config[key]
        else:
            return None
    return cur_config


T = TypeVar("T")


class PersistentConfig(Generic[T]):
    def __init__(self, env_name: str, config_path: str, env_value: T):
        self.env_name = env_name
        self.config_path = config_path
        self.env_value = env_value
        self.config_value = get_config_value(config_path)
        if self.config_value is not None:
            log.info(f"'{env_name}' loaded from config.json")
            self.value = self.config_value
        else:
            self.value = env_value

    def __str__(self):
        return str(self.value)

    @property
    def __dict__(self):
        raise TypeError(
            "PersistentConfig object cannot be converted to dict, use config_get or .value instead."
        )

    def __getattribute__(self, item):
        if item == "__dict__":
            raise TypeError(
                "PersistentConfig object cannot be converted to dict, use config_get or .value instead."
            )
        return super().__getattribute__(item)

    def save(self):
        # Don't save if the value is the same as the env value and the config value
        if self.env_value == self.value:
            if self.config_value == self.value:
                return
        log.info(f"Saving '{self.env_name}' to config.json")
        path_parts = self.config_path.split(".")
        config = CONFIG_DATA
        for key in path_parts[:-1]:
            if key not in config:
                config[key] = {}
            config = config[key]
        config[path_parts[-1]] = self.value
        save_config()
        self.config_value = self.value


class AppConfig:
    _state: dict[str, PersistentConfig]

    def __init__(self):
        super().__setattr__("_state", {})

    def __setattr__(self, key, value):
        if isinstance(value, PersistentConfig):
            self._state[key] = value
        else:
            self._state[key].value = value
            self._state[key].save()

    def __getattr__(self, key):
        return self._state[key].value

####################################
# LITELLM_CONFIG
####################################


def create_config_file(file_path):
    directory = os.path.dirname(file_path)

    # Check if directory exists, if not, create it
    if not os.path.exists(directory):
        os.makedirs(directory)

    # Data to write into the YAML file
    config_data = {
        "general_settings": {},
        "litellm_settings": {},
        "model_list": [],
        "router_settings": {},
    }

    # Write data to YAML file
    with open(file_path, "w") as file:
        yaml.dump(config_data, file)


LITELLM_CONFIG_PATH = f"{DATA_DIR}/litellm/config.yaml"

if not os.path.exists(LITELLM_CONFIG_PATH):
    print("Config file doesn't exist. Creating...")
    create_config_file(LITELLM_CONFIG_PATH)
    print("Config file created successfully.")


####################################
# OLLAMA_BASE_URL
####################################

OLLAMA_API_BASE_URL = os.environ.get(
    "OLLAMA_API_BASE_URL", "http://localhost:11434/api"
)

OLLAMA_BASE_URL = os.environ.get("OLLAMA_BASE_URL", "")
K8S_FLAG = os.environ.get("K8S_FLAG", "")
USE_OLLAMA_DOCKER = os.environ.get("USE_OLLAMA_DOCKER", "false")

if OLLAMA_BASE_URL == "" and OLLAMA_API_BASE_URL != "":
    OLLAMA_BASE_URL = (
        OLLAMA_API_BASE_URL[:-4]
        if OLLAMA_API_BASE_URL.endswith("/api")
        else OLLAMA_API_BASE_URL
    )

if ENV == "prod":
    if OLLAMA_BASE_URL == "/ollama" and not K8S_FLAG:
        if USE_OLLAMA_DOCKER.lower() == "true":
            # if you use all-in-one docker container (Open WebUI + Ollama)
            # with the docker build arg USE_OLLAMA=true (--build-arg="USE_OLLAMA=true") this only works with http://localhost:11434
            OLLAMA_BASE_URL = "http://localhost:11434"
        else:
            OLLAMA_BASE_URL = "http://host.docker.internal:11434"
    elif K8S_FLAG:
        OLLAMA_BASE_URL = "http://ollama-service.open-webui.svc.cluster.local:11434"


OLLAMA_BASE_URLS = os.environ.get("OLLAMA_BASE_URLS", "")
OLLAMA_BASE_URLS = OLLAMA_BASE_URLS if OLLAMA_BASE_URLS != "" else OLLAMA_BASE_URL

OLLAMA_BASE_URLS = [url.strip() for url in OLLAMA_BASE_URLS.split(";")]
OLLAMA_BASE_URLS = PersistentConfig(
    "OLLAMA_BASE_URLS", "ollama.base_urls", OLLAMA_BASE_URLS
)

####################################
# OPENAI_API
####################################

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")
OPENAI_API_BASE_URL = os.environ.get("OPENAI_API_BASE_URL", "")


if OPENAI_API_BASE_URL == "":
    OPENAI_API_BASE_URL = "https://api.openai.com/v1"

OPENAI_API_KEYS = os.environ.get("OPENAI_API_KEYS", "")
OPENAI_API_KEYS = OPENAI_API_KEYS if OPENAI_API_KEYS != "" else OPENAI_API_KEY

OPENAI_API_KEYS = [url.strip() for url in OPENAI_API_KEYS.split(";")]


OPENAI_API_BASE_URLS = os.environ.get("OPENAI_API_BASE_URLS", "")
OPENAI_API_BASE_URLS = (
    OPENAI_API_BASE_URLS if OPENAI_API_BASE_URLS != "" else OPENAI_API_BASE_URL
)

OPENAI_API_BASE_URLS = [
    url.strip() if url != "" else "https://api.openai.com/v1"
    for url in OPENAI_API_BASE_URLS.split(";")
]


####################################
# WEBUI
####################################

ENABLE_SIGNUP = os.environ.get("ENABLE_SIGNUP", "True").lower() == "true"
DEFAULT_MODELS = os.environ.get("DEFAULT_MODELS", None)


DEFAULT_PROMPT_SUGGESTIONS = (
    CONFIG_DATA["ui"]["prompt_suggestions"]
    if "ui" in CONFIG_DATA
    and "prompt_suggestions" in CONFIG_DATA["ui"]
    and type(CONFIG_DATA["ui"]["prompt_suggestions"]) is list
    else [
        {
            "title": ["Define Articles", "legal articles and its precise explaination"],
            "content": "Define Article 14 of the Indian constitution and its legal implications.",
        },
        {
            "title": ["Some major acts", "know more about some major acts"],
            "content": "What major Indian acts does a private company need to be aware of?",
        },
        {
            "title": ["Special economic zones", "insights on some taxation benefits"],
            "content": "What are special economic zones and what are their taxation benefits?",
        },
        {
            "title": ["Create agreements","snippets related to legal documents"],
            "content": "Create a service agreement for building a website.",
        },
    ]
)


DEFAULT_USER_ROLE = os.getenv("DEFAULT_USER_ROLE", "pending")
USER_PERMISSIONS_CHAT_DELETION = (
    os.environ.get("USER_PERMISSIONS_CHAT_DELETION", "True").lower() == "true"
)

USER_PERMISSIONS = {"chat": {"deletion": USER_PERMISSIONS_CHAT_DELETION}}


#MODEL_FILTER_ENABLED = os.environ.get("MODEL_FILTER_ENABLED", "False").lower() == "true"
#MODEL_FILTER_LIST = os.environ.get("MODEL_FILTER_LIST", "")
#MODEL_FILTER_LIST = [model.strip() for model in MODEL_FILTER_LIST.split(";")]

#WEBHOOK_URL = os.environ.get("WEBHOOK_URL", "")

ENABLE_MODEL_FILTER = PersistentConfig(
    "ENABLE_MODEL_FILTER",
    "model_filter.enable",
    os.environ.get("ENABLE_MODEL_FILTER", "False").lower() == "true",
)
MODEL_FILTER_LIST = os.environ.get("MODEL_FILTER_LIST", "")
MODEL_FILTER_LIST = PersistentConfig(
    "MODEL_FILTER_LIST",
    "model_filter.list",
    [model.strip() for model in MODEL_FILTER_LIST.split(";")],
)

WEBHOOK_URL = PersistentConfig(
    "WEBHOOK_URL", "webhook_url", os.environ.get("WEBHOOK_URL", "")
)
####################################
# WEBUI_VERSION
####################################

WEBUI_VERSION = os.environ.get("WEBUI_VERSION", "v1.0.0-alpha.100")

####################################
# WEBUI_AUTH (Required for security)
####################################

WEBUI_AUTH = True

####################################
# WEBUI_SECRET_KEY
####################################

WEBUI_SECRET_KEY = os.environ.get(
    "WEBUI_SECRET_KEY",
    os.environ.get(
        "WEBUI_JWT_SECRET_KEY", "t0p-s3cr3t"
    ),  # DEPRECATED: remove at next major version
)

if WEBUI_AUTH and WEBUI_SECRET_KEY == "":
    raise ValueError(ERROR_MESSAGES.ENV_VAR_NOT_FOUND)

####################################
# RAG
####################################

CHROMA_DATA_PATH = f"{DATA_DIR}/vector_db"
CHROMA_TENANT = os.environ.get("CHROMA_TENANT", chromadb.DEFAULT_TENANT)
CHROMA_DATABASE = os.environ.get("CHROMA_DATABASE", chromadb.DEFAULT_DATABASE)
CHROMA_HTTP_HOST = os.environ.get("CHROMA_HTTP_HOST", "")
CHROMA_HTTP_PORT = int(os.environ.get("CHROMA_HTTP_PORT", "8000"))
# Comma-separated list of header=value pairs
CHROMA_HTTP_HEADERS = os.environ.get("CHROMA_HTTP_HEADERS", "")
if CHROMA_HTTP_HEADERS:
    CHROMA_HTTP_HEADERS = dict(
        [pair.split("=") for pair in CHROMA_HTTP_HEADERS.split(",")]
    )
else:
    CHROMA_HTTP_HEADERS = None
CHROMA_HTTP_SSL = os.environ.get("CHROMA_HTTP_SSL", "false").lower() == "true"
# this uses the model defined in the Dockerfile ENV variable. If you dont use docker or docker based deployments such as k8s, the default embedding model will be used (sentence-transformers/all-MiniLM-L6-v2)

RAG_TOP_K = PersistentConfig(
    "RAG_TOP_K", "rag.top_k", int(os.environ.get("RAG_TOP_K", "5"))
)
RAG_RELEVANCE_THRESHOLD = PersistentConfig(
    "RAG_RELEVANCE_THRESHOLD",
    "rag.relevance_threshold",
    float(os.environ.get("RAG_RELEVANCE_THRESHOLD", "0.0")),
)

ENABLE_RAG_HYBRID_SEARCH = PersistentConfig(
    "ENABLE_RAG_HYBRID_SEARCH",
    "rag.enable_hybrid_search",
    os.environ.get("ENABLE_RAG_HYBRID_SEARCH", "").lower() == "true",
)

ENABLE_RAG_WEB_LOADER_SSL_VERIFICATION = PersistentConfig(
    "ENABLE_RAG_WEB_LOADER_SSL_VERIFICATION",
    "rag.enable_web_loader_ssl_verification",
    os.environ.get("ENABLE_RAG_WEB_LOADER_SSL_VERIFICATION", "True").lower() == "true",
)

RAG_EMBEDDING_ENGINE = PersistentConfig(
    "RAG_EMBEDDING_ENGINE",
    "rag.embedding_engine",
    os.environ.get("RAG_EMBEDDING_ENGINE", ""),
)

PDF_EXTRACT_IMAGES = PersistentConfig(
    "PDF_EXTRACT_IMAGES",
    "rag.pdf_extract_images",
    os.environ.get("PDF_EXTRACT_IMAGES", "False").lower() == "true",
)

RAG_EMBEDDING_MODEL = PersistentConfig(
    "RAG_EMBEDDING_MODEL",
    "rag.embedding_model",
    os.environ.get("RAG_EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"),
)
log.info(f"Embedding model set: {RAG_EMBEDDING_MODEL.value}"),

RAG_EMBEDDING_MODEL_AUTO_UPDATE = (
    os.environ.get("RAG_EMBEDDING_MODEL_AUTO_UPDATE", "").lower() == "true"
)

RAG_EMBEDDING_MODEL_TRUST_REMOTE_CODE = (
    os.environ.get("RAG_EMBEDDING_MODEL_TRUST_REMOTE_CODE", "").lower() == "true"
)

RAG_RERANKING_MODEL = PersistentConfig(
    "RAG_RERANKING_MODEL",
    "rag.reranking_model",
    os.environ.get("RAG_RERANKING_MODEL", ""),
)
if RAG_RERANKING_MODEL.value != "":
    log.info(f"Reranking model set: {RAG_RERANKING_MODEL.value}"),

RAG_RERANKING_MODEL_AUTO_UPDATE = (
    os.environ.get("RAG_RERANKING_MODEL_AUTO_UPDATE", "").lower() == "true"
)

RAG_RERANKING_MODEL_TRUST_REMOTE_CODE = (
    os.environ.get("RAG_RERANKING_MODEL_TRUST_REMOTE_CODE", "").lower() == "true"
)


if CHROMA_HTTP_HOST != "":
    CHROMA_CLIENT = chromadb.HttpClient(
        host=CHROMA_HTTP_HOST,
        port=CHROMA_HTTP_PORT,
        headers=CHROMA_HTTP_HEADERS,
        ssl=CHROMA_HTTP_SSL,
        tenant=CHROMA_TENANT,
        database=CHROMA_DATABASE,
        settings=Settings(allow_reset=True, anonymized_telemetry=False),
    )
else:
    CHROMA_CLIENT = chromadb.PersistentClient(
        path=CHROMA_DATA_PATH,
        settings=Settings(allow_reset=True, anonymized_telemetry=False),
        tenant=CHROMA_TENANT,
        database=CHROMA_DATABASE,
    )


# device type embedding models - "cpu" (default), "cuda" (nvidia gpu required) or "mps" (apple silicon) - choosing this right can lead to better performance
USE_CUDA = os.environ.get("USE_CUDA_DOCKER", "false")

if USE_CUDA.lower() == "true":
    DEVICE_TYPE = "cuda"
else:
    DEVICE_TYPE = "cpu"


CHUNK_SIZE = PersistentConfig(
    "CHUNK_SIZE", "rag.chunk_size", int(os.environ.get("CHUNK_SIZE", "1500"))
)

CHUNK_OVERLAP = PersistentConfig(
    "CHUNK_OVERLAP",
    "rag.chunk_overlap",
    int(os.environ.get("CHUNK_OVERLAP", "100")),
)

DEFAULT_RAG_TEMPLATE = """Use the following context as your learned knowledge, inside <context></context> XML tags.
<context>
    [context]
</context>

When answer to user:
- If you don't know, just say that you don't know.
- If you don't know when you are not sure, ask for clarification.
Avoid mentioning that you obtained the information from the context.
And answer according to the language of the user's question.

Given the context information, answer the query.
Query: [query]"""

RAG_TEMPLATE = PersistentConfig(
    "RAG_TEMPLATE",
    "rag.template",
    os.environ.get("RAG_TEMPLATE", DEFAULT_RAG_TEMPLATE),
)

RAG_OPENAI_API_BASE_URL = PersistentConfig(
    "RAG_OPENAI_API_BASE_URL",
    "rag.openai_api_base_url",
    os.getenv("RAG_OPENAI_API_BASE_URL", OPENAI_API_BASE_URL),
)
RAG_OPENAI_API_KEY = PersistentConfig(
    "RAG_OPENAI_API_KEY",
    "rag.openai_api_key",
    os.getenv("RAG_OPENAI_API_KEY", OPENAI_API_KEY),
)

ENABLE_RAG_LOCAL_WEB_FETCH = (
    os.getenv("ENABLE_RAG_LOCAL_WEB_FETCH", "False").lower() == "true"
)

YOUTUBE_LOADER_LANGUAGE = PersistentConfig(
    "YOUTUBE_LOADER_LANGUAGE",
    "rag.youtube_loader_language",
    os.getenv("YOUTUBE_LOADER_LANGUAGE", "en").split(","),
)

####################################
# Transcribe
####################################

WHISPER_MODEL = os.getenv("WHISPER_MODEL", "base")
WHISPER_MODEL_DIR = os.getenv("WHISPER_MODEL_DIR", f"{CACHE_DIR}/whisper/models")
WHISPER_MODEL_AUTO_UPDATE = (
    os.environ.get("WHISPER_MODEL_AUTO_UPDATE", "").lower() == "true"
)


####################################
# Images
####################################

IMAGE_GENERATION_ENGINE = PersistentConfig(
    "IMAGE_GENERATION_ENGINE",
    "image_generation.engine",
    os.getenv("IMAGE_GENERATION_ENGINE", ""),
)

ENABLE_IMAGE_GENERATION = PersistentConfig(
    "ENABLE_IMAGE_GENERATION",
    "image_generation.enable",
    os.environ.get("ENABLE_IMAGE_GENERATION", "").lower() == "true",
)
AUTOMATIC1111_BASE_URL = PersistentConfig(
    "AUTOMATIC1111_BASE_URL",
    "image_generation.automatic1111.base_url",
    os.getenv("AUTOMATIC1111_BASE_URL", ""),
)

COMFYUI_BASE_URL = PersistentConfig(
    "COMFYUI_BASE_URL",
    "image_generation.comfyui.base_url",
    os.getenv("COMFYUI_BASE_URL", ""),
)

IMAGES_OPENAI_API_BASE_URL = PersistentConfig(
    "IMAGES_OPENAI_API_BASE_URL",
    "image_generation.openai.api_base_url",
    os.getenv("IMAGES_OPENAI_API_BASE_URL", OPENAI_API_BASE_URL),
)
IMAGES_OPENAI_API_KEY = PersistentConfig(
    "IMAGES_OPENAI_API_KEY",
    "image_generation.openai.api_key",
    os.getenv("IMAGES_OPENAI_API_KEY", OPENAI_API_KEY),
)

IMAGE_SIZE = PersistentConfig(
    "IMAGE_SIZE", "image_generation.size", os.getenv("IMAGE_SIZE", "512x512")
)

IMAGE_STEPS = PersistentConfig(
    "IMAGE_STEPS", "image_generation.steps", int(os.getenv("IMAGE_STEPS", 50))
)

IMAGE_GENERATION_MODEL = PersistentConfig(
    "IMAGE_GENERATION_MODEL",
    "image_generation.model",
    os.getenv("IMAGE_GENERATION_MODEL", ""),
)

####################################
# Audio
####################################

AUDIO_OPENAI_API_BASE_URL = PersistentConfig(
    "AUDIO_OPENAI_API_BASE_URL",
    "audio.openai.api_base_url",
    os.getenv("AUDIO_OPENAI_API_BASE_URL", OPENAI_API_BASE_URL),
)
AUDIO_OPENAI_API_KEY = PersistentConfig(
    "AUDIO_OPENAI_API_KEY",
    "audio.openai.api_key",
    os.getenv("AUDIO_OPENAI_API_KEY", OPENAI_API_KEY),
)
AUDIO_OPENAI_API_MODEL = PersistentConfig(
    "AUDIO_OPENAI_API_MODEL",
    "audio.openai.api_model",
    os.getenv("AUDIO_OPENAI_API_MODEL", "tts-1"),
)
AUDIO_OPENAI_API_VOICE = PersistentConfig(
    "AUDIO_OPENAI_API_VOICE",
    "audio.openai.api_voice",
    os.getenv("AUDIO_OPENAI_API_VOICE", "alloy"),
)

####################################
# LiteLLM
####################################


ENABLE_LITELLM = os.environ.get("ENABLE_LITELLM", "True").lower() == "true"

LITELLM_PROXY_PORT = int(os.getenv("LITELLM_PROXY_PORT", "14365"))
if LITELLM_PROXY_PORT < 0 or LITELLM_PROXY_PORT > 65535:
    raise ValueError("Invalid port number for LITELLM_PROXY_PORT")
LITELLM_PROXY_HOST = os.getenv("LITELLM_PROXY_HOST", "127.0.0.1")

####################################
# LEGAL SEARCH
####################################

CHROMADB_PATH = f"{DATA_DIR}/chromadb"
# this uses the model defined in the Dockerfile ENV variable. If you dont use docker or docker based deployments such as k8s, the default embedding model will be used (all-Mi>
LEGAL_EMBEDDING_MODEL = os.environ.get("LEGAL_EMBEDDING_MODEL", "all-MiniLM-L6-v2")
# device type ebbeding models - "cpu" (default), "cuda" (nvidia gpu required) or "mps" (apple silicon) - choosing this right can lead to better performance
LEGAL_EMBEDDING_MODEL_DEVICE_TYPE = os.environ.get(
    "LEGAL_EMBEDDING_MODEL_DEVICE_TYPE", "cpu"
)
CHROMA_CLI = chromadb.PersistentClient(
    path=CHROMADB_PATH,
    settings=Settings(allow_reset=True, anonymized_telemetry=False),
)
CHUNK_SIZE = 1500
CHUNK_OVERLAP = 100

LEGAL_TEMPLATE = """Use the following context as your learned knowledge, inside <context></context> XML tags.
<context>
    [context]
</context>

When answer to user:
- If you don't know, just say that you don't know.
- If you don't know when you are not sure, ask for clarification.
Avoid mentioning that you obtained the information from the context.
And answer according to the language of the user's question.

Given the context information, answer the query.
Query: [query]"""




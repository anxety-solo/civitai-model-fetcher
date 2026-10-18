""" CivitAI Model Info Fetcher | by ANXETY """

import importlib.util
import argparse
import logging
import asyncio
import aiohttp
import random
import json
import sys
import re
import os

from urllib.parse import urlsplit, parse_qs
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


# ~~ CONSTANTS ~~

BASE_URL                = 'https://civitai.red'
API_BASE                = f"{BASE_URL}/api/v1"
EP_MODELS               = f"{API_BASE}/models"
EP_MODEL_VER            = f"{API_BASE}/model-versions"
EP_BY_HASH              = f"{API_BASE}/model-versions/by-hash"
EP_PERMISSIONS          = f"{API_BASE}/permissions/check"
CIVITAI_DOMAINS         = ('civitai.com', 'civitai.red', 'civitai.green')
USER_AGENT              = 'civitai-model-fetcher/2.0'

DATA_DIR                = 'data'
CONFIG_FILE             = 'config.json'
EXPORTS_SUBDIR          = 'exports'

API_KEY_DISPLAY_CHARS   = 8
MAX_API_KEYS            = 5
MAX_CONCURRENT_REQUESTS = 10
MAX_RETRIES             = 3
REQUEST_TIMEOUT         = 30
CONNECT_TIMEOUT         = 10
URL_REGEX               = r'https?://[^\s,]+'


def get_model_url(model_id: int | str) -> str:
    return f"{EP_MODELS}/{model_id}"


def get_version_url(version_id: int | str) -> str:
    return f"{EP_MODEL_VER}/{version_id}?nsfw=true"


def get_by_hash_url(sha256: str) -> str:
    return f"{EP_BY_HASH}/{sha256}?nsfw=true"


def get_model_page_url(model_id: int | str, version_id: int | str | None = None) -> str:
    url = f"{BASE_URL}/models/{model_id}"
    if version_id:
        url += f"?modelVersionId={version_id}"
    return url


def get_permissions_url(version_ids: list[int | str]) -> str:
    ids = ','.join(str(v) for v in version_ids)
    return f"{EP_PERMISSIONS}?entityIds={ids}&entityType=ModelVersion"


# ~~ COLORS ~~

RESET         = '\033[0m'
RED           = '\033[31m'
GREEN         = '\033[32m'
YELLOW        = '\033[33m'
BLUE          = '\033[34m'
MAGENTA       = '\033[35m'
CYAN          = '\033[36m'
BRIGHT_YELLOW = '\033[93m'


# ~~ LOGGING ~~

class ColorFormatter(logging.Formatter):
    """Colorize log records by level"""

    FORMATS = {
        logging.DEBUG:    CYAN + '%(levelname)s' + RESET + ': %(message)s',
        logging.INFO:     '%(message)s',    # plain message, no level tag
        logging.WARNING:  YELLOW + '%(levelname)s' + RESET + ': %(message)s',
        logging.ERROR:    RED + '%(levelname)s' + RESET + ': %(message)s',
    }

    def __init__(self):
        super().__init__()
        self._formatters: dict[int, logging.Formatter] = {}

    def format(self, record: logging.LogRecord) -> str:
        formatter = self._formatters.get(record.levelno)
        if formatter is None:
            pattern = self.FORMATS.get(record.levelno, '%(levelname)s: %(message)s')
            formatter = logging.Formatter(pattern)
            self._formatters[record.levelno] = formatter
        return formatter.format(record)


class Logger:
    """Colored console logger singleton"""
    _logger: logging.Logger = None

    @classmethod
    def setup(cls, verbose=False) -> logging.Logger:
        """Configure the console logger"""
        level = logging.DEBUG if verbose else logging.INFO

        if cls._logger is None:
            logger = logging.getLogger('civitai-fetcher')
            logger.propagate = False
            handler = logging.StreamHandler()
            handler.setFormatter(ColorFormatter())
            logger.addHandler(handler)
            cls._logger = logger

        cls._logger.setLevel(level)
        for handler in cls._logger.handlers:
            handler.setLevel(level)

        return cls._logger

    @classmethod
    def get(cls) -> logging.Logger:
        """Return the configured logger, creating it on demand"""
        return cls._logger or cls.setup()

    @classmethod
    def debug(cls, message: str):
        cls.get().debug(message)

    @classmethod
    def info(cls, message: str):
        cls.get().info(message)

    @classmethod
    def warning(cls, message: str):
        cls.get().warning(message)

    @classmethod
    def error(cls, message: str):
        cls.get().error(message)


# ~~ CONFIG ~~

def validate_api_key(key: str) -> tuple[bool, str | None]:
    """Check an API key shape; returns (accepted, message)"""
    if not key or any(char.isspace() for char in key):
        return False, 'Key must be a non-empty string without spaces'
    if not (len(key) == 32 and key.isalnum()):
        return True, 'Key does not look like a standard 32-char CivitAI key'
    return True, None


class ConfigStore:
    """Persist API keys in data/config.json"""

    def __init__(self, data_dir=DATA_DIR):
        self.data_dir = Path(data_dir)
        self.path = self.data_dir / CONFIG_FILE
        self._cache: dict[str, list[str]] = None

    def load(self) -> dict[str, list[str]]:
        """Load the config mapping, cached in memory"""
        if self._cache is not None:
            return self._cache

        self._cache = {'api_keys': []}
        if self.path.exists():
            try:
                data = json.loads(self.path.read_text(encoding='utf-8'))
                keys = data.get('api_keys') if isinstance(data, dict) else None
                if isinstance(keys, list):
                    self._cache = {'api_keys': keys}
                else:
                    Logger.warning(f"Unexpected config layout in {self.path.name}; ignoring it")
            except (json.JSONDecodeError, OSError, UnicodeDecodeError) as exc:
                Logger.warning(f"Config is unreadable ({exc}); starting fresh")
        return self._cache

    def keys(self) -> list[str]:
        """Return the stored API keys"""
        return list(self.load().get('api_keys', []))

    def save(self, keys: list[str]):
        """Write the API keys to disk and refresh the cache"""
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps({'api_keys': keys}, indent=2), encoding='utf-8')
        self._cache = {'api_keys': list(keys)}

    def add(self, key: str) -> bool:
        """Append a key, evicting the oldest past the cap; returns False if duplicate"""
        keys = self.keys()
        if key in keys:
            return False

        if len(keys) >= MAX_API_KEYS:
            keys = keys[1:]

        keys.append(key)
        self.save(keys)
        return True

    def remove(self, index: int) -> str | None:
        """Remove a key by zero-based index; returns the removed key or None"""
        keys = self.keys()
        if not 0 <= index < len(keys):
            return None

        removed = keys.pop(index)
        self.save(keys)
        return removed


def resolve_keys(token: str = None, store: ConfigStore = None) -> list[str]:
    """Resolve API keys: --token, then env, then stored config"""
    token = (token or '').strip()
    if token:
        return [token]

    env = os.environ.get('CIVITAI_API_KEY', '')
    env_keys = [part.strip() for part in env.split(',') if part.strip()]
    if env_keys:
        return env_keys

    return (store or ConfigStore()).keys()


# ~~ URL PARSING ~~

def is_civitai_url(url: str) -> bool:
    """Check whether a URL points at a supported CivitAI domain"""
    try:
        host = urlsplit(url).hostname or ''
    except ValueError:
        return False
    return host.removeprefix('www.') in CIVITAI_DOMAINS


def split_inputs(values: list[str] | None) -> list[str]:
    """Flatten CLI strings into comma/space separated tokens"""
    tokens = []
    for value in values or []:
        tokens.extend(token for token in re.split(r'[,\s]+', value) if token)
    return tokens


def classify_target(token: str) -> dict[str, str | None] | None:
    """Map a token to {'kind', 'model_id', 'version_id', 'hash', 'raw'} or None"""
    token = token.strip()

    if token.isdigit():
        return {'kind': 'model', 'model_id': token, 'version_id': None, 'hash': None, 'raw': token}

    if re.fullmatch(r'[0-9a-fA-F]{8,64}', token):
        return {'kind': 'hash', 'model_id': None, 'version_id': None, 'hash': token, 'raw': token}

    if not is_civitai_url(token):
        return None

    parts = urlsplit(token)

    download = re.search(r'/api/download/models/(\d+)', parts.path)
    if download:
        return {'kind': 'version', 'model_id': None, 'version_id': download.group(1), 'hash': None, 'raw': token}

    page = re.search(r'/models/(\d+)', parts.path)
    if page:
        version_id = (parse_qs(parts.query).get('modelVersionId') or [None])[0]
        return {'kind': 'model', 'model_id': page.group(1), 'version_id': version_id, 'hash': None, 'raw': token}

    return None


def collect_targets(tokens: list[str]) -> list[dict[str, str | None]]:
    """Classify tokens, preserving first-seen order and dropping duplicates"""
    targets          = []
    seen: set[tuple] = set()
    model_ids        = 0
    hashes           = 0
    urls             = 0

    for token in tokens:
        target = classify_target(token)
        if not target:
            Logger.debug(f"Skipping invalid input: {token}")
            continue

        key = (target['kind'], target['model_id'], target['version_id'], target['hash'])
        if key in seen:
            continue

        seen.add(key)
        targets.append(target)

        if target['kind'] == 'model' and target['version_id'] is None:
            model_ids += 1
        elif target['kind'] == 'hash':
            hashes += 1
        else:
            urls += 1

    Logger.debug(f"Model IDs found: {model_ids}")
    Logger.debug(f"Hashes found: {hashes}")
    Logger.debug(f"URLs found: {urls}")
    Logger.debug(f"Total unique targets to process: {len(targets)}")
    for t in targets:
        Logger.debug(f"Target to process: {t['raw']}")

    return targets


# ~~ FILE SCANNING ~~

def _entry_urls(value: Any) -> list[str]:
    """Extract CivitAI URLs from a list of entry dicts"""
    urls = []
    if isinstance(value, list):
        for item in value:
            url = item.get('url') if isinstance(item, dict) else None
            if isinstance(url, str) and is_civitai_url(url):
                urls.append(url)
    return urls


def _build_node(mapping: dict) -> dict:
    """Recursively turn a dict into {key: subtree|url-list}, pruning empty branches"""
    node = {}
    for key, value in mapping.items():
        if isinstance(value, dict):
            child = _build_node(value)
            if child:
                node[key] = child
        elif isinstance(value, list):
            urls = _entry_urls(value)
            if urls:
                node[key] = urls
    return node


def build_catalog_tree(namespace: dict) -> dict:
    """Build a catalog tree from a module namespace, using top-level dicts as roots"""
    candidates = {
        name: value for name, value in namespace.items()
        if not name.startswith('_') and isinstance(value, (dict, list))
    }
    candidate_ids = {id(value) for value in candidates.values()}

    nested_ids = set()
    for value in candidates.values():
        if not isinstance(value, dict):
            continue

        for child in value.values():
            if id(child) in candidate_ids:
                nested_ids.add(id(child))

    tree = {}
    for name, value in candidates.items():
        if id(value) in nested_ids:
            continue

        node = _build_node(value) if isinstance(value, dict) else _entry_urls(value)
        if node:
            tree[name] = node

    return tree


def select_urls(tree: dict, ask) -> list[str]:
    """Walk the tree, asking which branches to include at each group level"""
    return _select_node(tree, (), ask)


def _select_node(node: dict, path: tuple, ask) -> list[str]:
    """Recurse into a tree node; leaf levels are included in full"""
    while len(node) == 1 and isinstance(next(iter(node.values())), dict):
        key  = next(iter(node))
        node = node[key]
        path += (key,)

    if all(isinstance(value, list) for value in node.values()):
        urls = []
        for value in node.values():
            urls.extend(value)
        return urls

    urls = []
    for key in ask(path, list(node.keys())):
        value = node[key]
        if isinstance(value, list):
            urls.extend(value)
        else:
            urls.extend(_select_node(value, path + (key,), ask))

    return urls


def _prompt_choice(title: str, labels: list[str], prompt_text: str) -> list[str]:
    """Prompt for one or more labels from a numbered list"""
    print(f"\n{MAGENTA}{title}:{RESET}")
    for index, label in enumerate(labels, 1):
        print(f"{BLUE}{index:2}.{RESET} {label}")

    try:
        choice = input(f"\n{YELLOW}{prompt_text}: {RESET}").strip().lower()
    except (KeyboardInterrupt, EOFError):
        Logger.error('\nOperation cancelled!')
        sys.exit(1)

    if choice in ('', 'all'):
        return list(labels)

    return [
        labels[int(part) - 1]
        for part in choice.split(',')
        if part.strip().isdigit() and 1 <= int(part) <= len(labels)
    ]


def ask_groups(path: tuple, keys: list[str]) -> list[str]:
    """Prompt for catalog branches at one level"""
    title = ' / '.join(path) if path else 'Catalog'
    return _prompt_choice(title, keys, 'Select (comma-separated/all)')


def load_python_catalog(file_path: Path | str, ask=None) -> list[str]:
    """Load a Python catalog module and return the selected CivitAI URLs"""
    try:
        spec = importlib.util.spec_from_file_location('civitai_catalog', file_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    except Exception as exc:
        Logger.error(f"Failed to load Python catalog {Path(file_path).name}: {exc}")
        return []

    tree = build_catalog_tree(vars(module))
    if not tree:
        Logger.warning(f"No CivitAI URLs found in {Path(file_path).name}")
        return []

    if ask is None:
        ask = ask_groups if sys.stdin.isatty() else (lambda path, keys: keys)

    return select_urls(tree, ask)


# --- Flat Files ---

def extract_urls_from_dict(data: Any) -> list[str]:
    """Recursively collect CivitAI URLs from a nested structure"""
    urls = []
    if isinstance(data, dict):
        url = data.get('url')
        if isinstance(url, str) and is_civitai_url(url):
            urls.append(url)
        for value in data.values():
            urls.extend(extract_urls_from_dict(value))
    elif isinstance(data, (list, tuple, set)):
        for item in data:
            urls.extend(extract_urls_from_dict(item))

    return urls


def extract_urls_from_text(content: str) -> list[str]:
    """Pull CivitAI URLs out of free text"""
    urls = []
    for match in re.findall(URL_REGEX, content):
        clean = match.rstrip('.,;')
        if is_civitai_url(clean):
            urls.append(clean)

    return urls


def resolve_file_path(file_path: str) -> list[Path]:
    """Resolve a possibly extension-less path to matching files"""
    path = Path(file_path)
    if path.exists():
        return [path]
    if not path.parent.is_dir():
        return []

    prefix = f"{path.name}."
    return sorted(child for child in path.parent.iterdir() if child.name.startswith(prefix))


def _choose_files(matches: list[Path]) -> list[Path]:
    """Prompt for one or more files among extension-less matches"""
    chosen = _prompt_choice(
        'Multiple files found',
        [match.name for match in matches],
        'Select file (comma-separated/all)',
    )
    return [match for match in matches if match.name in chosen]


def process_file(file_path: str, ask=None) -> list[str]:
    """Extract CivitAI URLs from JSON, Python, or TXT files"""
    paths = resolve_file_path(file_path)
    if not paths:
        Logger.error(f"No matching files found for: {file_path}")
        return []
    if len(paths) >= 2:
        paths = _choose_files(paths)

    urls = []
    for path in paths:
        Logger.debug(f"Processing file: {path.name}")
        if not path.is_file():
            Logger.error(f"Not a file: {path}")
            continue

        try:
            suffix = path.suffix.lower()
            if suffix == '.json':
                urls.extend(extract_urls_from_dict(json.loads(path.read_text(encoding='utf-8'))))
            elif suffix == '.py':
                urls.extend(load_python_catalog(path, ask))
            elif suffix == '.txt':
                urls.extend(extract_urls_from_text(path.read_text(encoding='utf-8')))
            else:
                Logger.error(f"Unsupported file type: {path.suffix}")
        except json.JSONDecodeError as exc:
            Logger.error(f"Invalid JSON in {path.name}: {exc}")
        except UnicodeDecodeError:
            Logger.error(f"Encoding error in {path.name} (use UTF-8)")
        except OSError as exc:
            Logger.error(f"Cannot read {path.name}: {exc}")

    return urls


# ~~ API CLIENT ~~

class CivitaiAPI:
    """Async CivitAI API client with retries and per-id caching"""

    def __init__(self, tokens: list[str] = None, session: aiohttp.ClientSession = None):
        self.tokens = list(tokens or [])
        self.session = session
        self._owns_session = session is None
        self._connector: aiohttp.TCPConnector = None
        self._models: dict[str, asyncio.Task] = {}
        self._versions: dict[str, asyncio.Task] = {}
        self._hashes: dict[str, asyncio.Task] = {}

    async def __aenter__(self):
        """Create the HTTP session unless one was injected"""
        if self.session is None:
            self._connector = aiohttp.TCPConnector(
                limit=MAX_CONCURRENT_REQUESTS * 2,
                limit_per_host=MAX_CONCURRENT_REQUESTS,
                ttl_dns_cache=300,
                use_dns_cache=True,
            )
            self.session = aiohttp.ClientSession(
                connector=self._connector,
                timeout=aiohttp.ClientTimeout(total=REQUEST_TIMEOUT, connect=CONNECT_TIMEOUT),
            )
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Close the connector and any session we created"""
        if self._connector is not None:
            await self._connector.close()
        if self._owns_session and self.session is not None:
            await self.session.close()

    def _headers(self) -> dict[str, str]:
        """Build request headers, adding a random bearer token when available"""
        headers = {'User-Agent': USER_AGENT}
        if self.tokens:
            headers['Authorization'] = f"Bearer {random.choice(self.tokens)}"
        return headers

    async def request_json(self, url: str) -> dict | None:
        """GET a JSON document, retrying transient failures"""
        for attempt in range(MAX_RETRIES):
            failure = None
            try:
                async with self.session.get(url, headers=self._headers()) as response:
                    if response.status == 200:
                        return await response.json()
                    if response.status == 401:
                        Logger.error('Unauthorized (401) - add a key with -k or --token')
                        return None
                    if response.status == 404:
                        Logger.error(f"HTTP Error 404: Not Found ({url})")
                        return None
                    if response.status == 429 or response.status >= 500:
                        failure = f"HTTP {response.status}"
                    else:
                        Logger.error(f"HTTP {response.status} for {url}")
                        return None
            except (aiohttp.ClientError, asyncio.TimeoutError, ValueError) as exc:
                failure = exc

            if attempt == MAX_RETRIES - 1:
                Logger.error(f"Request failed after {MAX_RETRIES} attempts ({failure}): {url}")
                return None

            wait = 2 ** attempt
            Logger.warning(f"Transient error ({failure}); retrying in {wait}s")
            await asyncio.sleep(wait)
        return None

    async def _cached(self, cache: dict, key: str, url: str) -> dict | None:
        """Return a cached in-flight/complete request, starting one if needed"""
        task = cache.get(key)
        if task is None:
            task = asyncio.ensure_future(self.request_json(url))
            cache[key] = task

        return await task

    async def fetch_model(self, model_id: str) -> dict | None:
        """Fetch a model document, cached by id (single-flight)"""
        return await self._cached(self._models, model_id, get_model_url(model_id))

    async def fetch_version(self, version_id: str) -> dict | None:
        """Fetch a model-version document, cached by id (single-flight)"""
        return await self._cached(self._versions, version_id, get_version_url(version_id))

    async def fetch_version_by_hash(self, file_hash: str) -> dict | None:
        """Fetch a model-version by any supported file hash"""
        return await self._cached(
            self._hashes,
            file_hash.lower(),
            get_by_hash_url(file_hash),
        )

    async def check_permissions(self, version_ids: list[str | int]) -> dict[str, bool]:
        """Return {version_id: can_access} via /permissions/check (false = gated)"""
        if not version_ids:
            return {}

        url = get_permissions_url(version_ids)
        data = await self.request_json(url)
        if not isinstance(data, dict):
            return {}

        return {str(k): bool(v) for k, v in data.items()}

    async def resolve_target(self, target: dict) -> tuple[dict, str | None] | None:
        """Return (model_data, current_version_id) for a classified target"""
        if target['kind'] == 'hash':
            version = await self.fetch_version_by_hash(target['hash'])
            if not version:
                return None

            model_id = version.get('modelId')
            model = await self.fetch_model(str(model_id)) if model_id else None
            if not model:
                Logger.warning(f"Could not load model for hash {target['hash']}")
                return None

            return model, str(version.get('id'))

        if target['kind'] == 'model':
            model = await self.fetch_model(target['model_id'])
            if not model:
                Logger.debug(f"Model ID {target['model_id']} not found or invalid")
                return None

            versions = model.get('modelVersions') or []
            current = target['version_id'] or (str(versions[0].get('id')) if versions else None)

            return model, current

        version = await self.fetch_version(target['version_id'])
        if not version:
            return None

        model_id = version.get('modelId')
        model = await self.fetch_model(str(model_id)) if model_id else None
        if not model:
            Logger.warning(f"Could not load model for version {target['version_id']}")
            return None

        return model, target['version_id']

    async def resolve_all(self, targets: list[dict]) -> list:
        """Resolve targets concurrently, preserving input order"""
        semaphore = asyncio.Semaphore(MAX_CONCURRENT_REQUESTS)

        async def worker(target):
            async with semaphore:
                try:
                    return await self.resolve_target(target)
                except Exception as exc:
                    Logger.error(f"Failed to resolve {target}: {exc}")
                    return None

        return await asyncio.gather(*(worker(target) for target in targets))


# ~~ AVAILABILITY ~~

def get_availability_label(version_data: dict, gated: bool = None) -> str:
    """Human-readable availability status for a model version"""
    is_gated = gated if gated is not None else version_data.get('paidAccess')
    if not is_gated:
        return 'Public'

    deadline = (
        version_data.get('earlyAccessDeadline')
        or version_data.get('earlyAccessEndsAt')
        or (version_data.get('earlyAccessConfig') or {}).get('endsAt')
    )
    if not deadline:
        return 'Paid'

    date = str(deadline).split('T')[0]
    try:
        if isinstance(deadline, (int, float)):
            end = datetime.fromtimestamp(deadline, tz=timezone.utc)
        else:
            end = datetime.fromisoformat(str(deadline).replace('Z', '+00:00'))
            end = end if end.tzinfo else end.replace(tzinfo=timezone.utc)
        left = max(0, (end.date() - datetime.now(timezone.utc).date()).days)
    except (ValueError, OSError, OverflowError):
        return f"Early Access (until {date})"

    tail = 'ends today' if left == 0 else f"{left} day{'s' * (left != 1)} left"
    return f"Early Access (until {date}, {tail})"


# ~~ DISPLAY ~~

def print_model_header(model: dict, current_version_id: str | None, raw_target: str):
    """Print the model-level header block"""
    model_id = model.get('id', 'N/A')
    print(f"\n{MAGENTA}>> Processing:{RESET} {CYAN}{raw_target}{RESET}\n")

    print(f"{MAGENTA}Model:{RESET} {CYAN}{model.get('name', 'Unknown')}{RESET}")

    versions = model.get('modelVersions') or []
    current = next(
        (version for version in versions if str(version.get('id')) == str(current_version_id)),
        None,
    )
    if current:
        print(f"{MAGENTA}Version:{RESET} {CYAN}{current.get('name', 'Unknown')}{RESET}")

    print(f"{MAGENTA}Type:{RESET} {CYAN}{model.get('type', 'N/A')}{RESET}")
    print(f"{MAGENTA}Model ID:{RESET} {CYAN}{model_id}{RESET}")

    model_url = get_model_page_url(model_id, current_version_id)
    print(f"{MAGENTA}Model URL:{RESET} {model_url}")


def print_versions(
    model_id: Any,
    versions: list[dict],
    current_version_id: str | None,
    args,
    gated_map: dict[str, bool] = None,
):
    """Print the version table with dynamic column widths"""
    print(f"\n{GREEN}>> Available Versions:{RESET}\n")

    multi     = len(versions) >= 2
    gated_map = gated_map or {}
    PAD       = 5 # extra spaces after each value

    # Build rows of plain values first
    rows = []
    for index, version in enumerate(versions):
        version_id = str(version.get('id', 'N/A'))

        is_current = version_id == str(current_version_id)
        is_latest  = index == 0
        highlight  = multi and is_current

        cells = [
            ('Ver.ID', version_id),
            ('Name', str(version.get('name', 'Unknown'))),
            ('Base', str(version.get('baseModel', 'N/A'))),
        ]
        if args.show_url:
            cells.append(('URL', get_model_page_url(model_id, version_id)))
        if args.show_download and version.get('downloadUrl'):
            cells.append(('Download', version['downloadUrl']))

        tags = []
        if multi:
            if is_current:
                tags.append(f"{YELLOW}<< Current{RESET}")
            elif is_latest:
                tags.append(f"{GREEN}<< Latest{RESET}")

        can_access = gated_map.get(version_id)
        gated = (can_access is False) if can_access is not None else None
        label = get_availability_label(version, gated=gated)
        if label != 'Public':
            tags.append(f"{BRIGHT_YELLOW}<< {label}{RESET}")

        rows.append((cells, tags, highlight))

    # Compute max width per column (label + value + padding)
    col_count = max(len(r[0]) for r in rows) if rows else 0
    widths = [0] * col_count
    for cells, _, _ in rows:
        for i, (label, value) in enumerate(cells):
            needed = len(label) + 2 + len(str(value)) + PAD
            if needed > widths[i]:
                widths[i] = needed

    # Render
    for cells, tags, highlight in rows:
        parts = []
        for i, (label, value) in enumerate(cells):
            color = YELLOW if highlight else CYAN
            pad = widths[i] - len(label) - 2
            parts.append(f"{color}{label}:{RESET} {str(value).ljust(pad)}")

        line = ' | '.join(parts)
        if tags:
            line += ' ' + ' '.join(tags)

        print(line)


def print_file_format_help():
    """Show supported input file formats"""
    print(f"\n{MAGENTA}File Format Requirements:{RESET}")

    print(f"{GREEN}JSON:{RESET}")
    print('''```
{
    "My Models": [
        {"url": "https://civitai.red/models/1234", "name": "model1.safetensors"}
    ]
}
```''')

    print(f"\n{GREEN}Python catalog (nested, any names):{RESET}")
    print('''```
SD = {
    "model": {
        "Model Anime": [
            {"url": "https://civitai.red/api/download/models/5678", "name": "anime.safetensors"}
        ]
    },
    "vae": {}
}

MODEL_CATEGORIES = {"SD": SD}
```''')

    print(f"\n{GREEN}Text file:{RESET}")
    print('''```
https://civitai.com/models/1234
https://civitai.green/models/5678 model1.safetensors
https://civitai.red/api/download/models/9012, https://civitai.com/models/3456
```''')


# ~~ EXPORT ~~

def format_size(size_kb: float | int | None) -> str:
    """Convert a size in KB to a human-readable string"""
    if not size_kb:
        return 'N/A'

    size = float(size_kb) * 1024
    for unit in ('B', 'KB', 'MB', 'GB'):
        if size < 1024.0:
            return f"{size:.1f} {unit}"
        size /= 1024.0
    return f"{size:.1f} GB"


def prepare_export_data(
    model: dict,
    base_models: list[str] = None,
    types: list[str] = None,
) -> dict | None:
    """Build the export record for a model document, applying optional filters"""
    model_type = model.get('type', 'N/A')
    if types and model_type not in types:
        return None

    model_id = model.get('id', 'N/A')
    record = {
        'model_id':   model_id,
        'model_name': model.get('name', 'N/A'),
        'model_type': model_type,
        'url':        get_model_page_url(model_id),
        'versions':   [],
    }

    for version in model.get('modelVersions') or []:
        base_model = version.get('baseModel', 'N/A')
        if base_models and base_model not in base_models:
            continue

        files = version.get('files') or [{}]
        version_id = version.get('id', 'N/A')
        record['versions'].append({
            'version_id':    version_id,
            'version_name':  version.get('name', 'N/A'),
            'created_at':    version.get('createdAt', 'N/A'),
            'updated_at':    version.get('updatedAt', 'N/A'),
            'download_url':  version.get('downloadUrl', 'N/A'),
            'early_access':  version.get('availability') == 'EarlyAccess',
            'size':          format_size(files[0].get('sizeKB', 0)),
            'base_model':    base_model,
            'trained_words': version.get('trainedWords', []),
            'url':           get_model_page_url(model_id, version_id),
        })

    if not record['versions']:
        return None

    return record


def export_to_json(records: list[dict], filename: str):
    """Merge records into data/exports/<filename>, deduplicated by model_id"""
    if not records:
        Logger.warning('No data to export')
        return

    if not filename.endswith('.json'):
        filename += '.json'
    if filename == 'export.json':
        filename = f"export_{datetime.now():%Y%m%d_%H%M%S}.json"

    export_dir = Path(DATA_DIR) / EXPORTS_SUBDIR
    export_dir.mkdir(parents=True, exist_ok=True)
    export_path = export_dir / filename

    existing = []
    if export_path.exists():
        try:
            loaded = json.loads(export_path.read_text(encoding='utf-8'))
            if isinstance(loaded, list):
                existing = loaded
            else:
                Logger.warning(f"{export_path.name} is not a list; overwriting it")
        except (json.JSONDecodeError, OSError) as exc:
            Logger.warning(f"Cannot read {export_path.name} ({exc}); overwriting it")

    known = {item.get('model_id') for item in existing if isinstance(item, dict)}
    fresh = []
    for record in records:
        if record['model_id'] in known:
            continue
        known.add(record['model_id'])
        fresh.append(record)

    if existing and not fresh:
        Logger.info('No new data to add to existing export file')
        return

    export_path.write_text(
        json.dumps(existing + fresh, indent=2, ensure_ascii=False),
        encoding='utf-8',
    )

    Logger.info(f"Successfully exported to: {export_path}")


# ~~ CLI ~~

def build_parser() -> argparse.ArgumentParser:
    """Build the argument parser"""
    parser = argparse.ArgumentParser(
        description='CivitAI Model Information Fetcher - Supports URLs, Model IDs and file hashes',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument('urls', nargs='*', help='Input URLs, model IDs or file hashes')
    parser.add_argument('-i', '--input', nargs='+', help='Additional URLs, model IDs or hashes')
    parser.add_argument('-f', '--file', help='Path to model list file (JSON/Python/TXT)')
    parser.add_argument('-u', '--show-url', action='store_true', help='Display model URLs')
    parser.add_argument('-d', '--show-download', action='store_true', help='Display download URLs')
    parser.add_argument('-k', '--change-key', action='store_true', help='Manage API keys (add/remove)')
    parser.add_argument('-t', '--token', help='Use this API key for this run (not stored)')
    parser.add_argument('--show-help', action='store_true', help='Show file format examples')
    parser.add_argument('--export', nargs='?', const='export.json', help='Export data to JSON')
    parser.add_argument('--json', action='store_true', help='Output raw API response as JSON')
    parser.add_argument('--base-model', nargs='+', help='Filter export by baseModel values')
    parser.add_argument('--type', nargs='+', help='Filter export by model type values')
    parser.add_argument('-v', '--verbose', action='store_true', help='Enable verbose logging')

    return parser


def manage_api_keys(store: ConfigStore):
    """Interactive add/remove/list of stored API keys"""
    while True:
        keys = store.keys()

        print(f"\n{MAGENTA}Current API Keys ({len(keys)}/{MAX_API_KEYS}):{RESET}")
        for index, key in enumerate(keys, 1):
            print(f"{index}. ...{key[-API_KEY_DISPLAY_CHARS:]}")
        print('\n1. Add key\n2. Remove key\n3. Exit')

        try:
            choice = input(f"{YELLOW}Select option: {RESET}").strip()
        except (KeyboardInterrupt, EOFError):
            print()
            return

        if choice == '1':
            try:
                key = input(f"{CYAN}Enter new API key: {RESET}").strip()
            except (KeyboardInterrupt, EOFError):
                print()
                return

            accepted, message = validate_api_key(key)
            if not accepted:
                Logger.error(message)
            elif store.add(key):
                Logger.info('Key added')
            else:
                Logger.warning('Key already exists')

            if accepted and message:
                Logger.warning(message)

        elif choice == '2':
            try:
                index = input(f"{CYAN}Number to remove: {RESET}").strip()
            except (KeyboardInterrupt, EOFError):
                print()
                return

            if index.isdigit() and store.remove(int(index) - 1):
                Logger.info('Key removed')
            else:
                Logger.error('Invalid selection')

        elif choice == '3':
            return

        else:
            Logger.error('Invalid option')


def print_usage():
    """Print CLI usage examples"""
    print(f"\n{YELLOW}Usage examples:{RESET}")
    print('  python civitai_fetcher.py 1234')
    print('  python civitai_fetcher.py https://civitai.red/models/5678')
    print('  python civitai_fetcher.py A5F58EB1C3')
    print('  python civitai_fetcher.py -i url1 url2 -f models.json')
    print('  python civitai_fetcher.py -f models.py -d --json')
    print('  python civitai_fetcher.py --export --base-model "SD 1.5" SDXL --type LORA')
    print('  python civitai_fetcher.py --show-help')


async def run(targets: list[dict], tokens: list[str], args) -> list[dict]:
    """Resolve and display every target, returning export records"""
    records      = []
    raw_payloads = []
    newer_count  = 0
    processed    = 0

    async with CivitaiAPI(tokens) as api:
        Logger.debug(f"Starting batch processing of {len(targets)} target(s)\n")
        resolved = await api.resolve_all(targets)

        all_version_ids = []
        for payload in resolved:
            if not payload:
                continue

            model, _ = payload
            for v in model.get('modelVersions') or []:
                vid = v.get('id')
                if vid is not None:
                    all_version_ids.append(vid)

        gated_map: dict[str, bool] = {}
        if all_version_ids:
            unique_ids = list(dict.fromkeys(str(vid) for vid in all_version_ids))
            Logger.debug(f"Checking permissions for {len(unique_ids)} version(s)")
            gated_map = await api.check_permissions(unique_ids)

        for target, payload in zip(targets, resolved):
            if not payload:
                continue

            model, current_version_id = payload
            versions = model.get('modelVersions') or []
            if not versions:
                Logger.warning(f"No versions for model {model.get('id', 'N/A')}")
                continue

            processed += 1

            latest_id = str(versions[0].get('id')) if versions else None
            if current_version_id and latest_id and str(current_version_id) != latest_id:
                newer_count += 1

            if args.json:
                raw_payloads.append(model)
                continue

            if len(targets) >= 2:
                print(f"\n{RED}{'-' * 60}{RESET}")

            print_model_header(model, current_version_id, target['raw'])
            print_versions(model.get('id'), versions, current_version_id, args, gated_map)

            if args.export:
                record = prepare_export_data(model, args.base_model, args.type)
                if record:
                    records.append(record)

    if args.json:
        print(json.dumps(
            raw_payloads if len(raw_payloads) >= 2 else (raw_payloads[0] if raw_payloads else {}),
            indent=2,
            ensure_ascii=False,
        ))
    else:
        print(f"\n{GREEN}>> Summary:{RESET}")
        print(f"  Processed: {CYAN}{processed}{RESET} model(s)")
        if newer_count:
            print(f"  Newer versions available: {CYAN}{newer_count}{RESET}")
        else:
            print(f"  No models with newer versions found")

    Logger.debug(f"Finished processing {len(targets)} target(s)")
    return records


def main():
    args = build_parser().parse_args()
    Logger.setup(args.verbose)

    if args.show_help:
        print_file_format_help()
        return

    store = ConfigStore()
    if args.change_key:
        manage_api_keys(store)
        return

    sys.dont_write_bytecode = True

    tokens = resolve_keys(args.token, store)
    if not tokens:
        Logger.warning('No API key found - running anonymously (public endpoints only). Add one with -k')
    else:
        Logger.debug(f"Using {len(tokens)} API key(s)")

    tokens_in = split_inputs(args.urls) + split_inputs(args.input)
    if args.file:
        tokens_in.extend(process_file(args.file))

    targets = collect_targets(tokens_in)
    if not targets:
        Logger.error('No valid input sources provided!')
        print_usage()
        sys.exit(1)

    try:
        records = asyncio.run(run(targets, tokens, args))
        if args.export and records:
            export_to_json(records, args.export)
    except KeyboardInterrupt:
        Logger.error('\nOperation cancelled by user!')
        sys.exit(1)
    except aiohttp.ClientError as exc:
        Logger.error(f"Network error: {exc}")
        sys.exit(1)


if __name__ == '__main__':
    main()

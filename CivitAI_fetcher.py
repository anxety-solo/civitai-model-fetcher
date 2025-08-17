"""
CivitAI Model Information Fetcher
Author: ANXETY (anxety_solo)
"""

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
from typing import Dict, List, Set, Any, Optional, Tuple
from colorama import init, Fore, Back, Style
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path


# Initialize colorama for cross-platform color support
init(autoreset=True)


# ==============================================
# CONFIGURATION CONSTANTS (Edit these as needed)
# ==============================================
DATA_DIR                = 'data'                        # Directory for storing data files
CONFIG_FILE             = 'civitai_config.json'         # Name of the configuration file
API_KEY_DISPLAY_CHARS   = 8                             # Number of API key characters to display (for masking)
MAX_API_KEYS            = 5                             # Maximum number of API keys allowed
URL_REGEX               = r'https?://[^\s,]+'           # Regular expression for matching URLs in text files
MAX_CONCURRENT_REQUESTS = 10                            # Maximum number of concurrent API requests

# This constant defines the width of columns for table-like output in the console
COLUMN_WIDTH = {
    'id': 20,
    'index': 15,
    'name': 40,
    'url': 65,
    'download': 65
}

# ==============================================


@dataclass
class ModelInfo:
    """Data class for model information"""
    model_id: str
    model_name: str
    model_type: str
    url: str
    versions: List[Dict] = field(default_factory=list)

@dataclass
class VersionInfo:
    """Data class for version information"""
    version_id: str
    version_name: str
    created_at: str
    updated_at: str
    download_url: str
    early_access: bool
    size: str
    url: str


class ColorFormatter(logging.Formatter):
    """Custom log formatter with colored output"""
    FORMATS = {
        logging.DEBUG: Fore.CYAN + '%(levelname)s' + Style.RESET_ALL + ': %(message)s',
        logging.INFO: Fore.GREEN + '%(levelname)s' + Style.RESET_ALL + ': %(message)s',
        logging.WARNING: Fore.YELLOW + '%(levelname)s' + Style.RESET_ALL + ': %(message)s',
        logging.ERROR: Fore.RED + '%(levelname)s' + Style.RESET_ALL + ': %(message)s',
        logging.CRITICAL: Back.RED + Fore.WHITE + '%(levelname)s' + Style.RESET_ALL + ': %(message)s'
    }

    def format(self, record):
        log_fmt = self.FORMATS.get(record.levelno)
        formatter = logging.Formatter(log_fmt)
        return formatter.format(record)


class Logger:
    """Logger class for unified application logging."""
    _instance = None
    _logger = None
    _verbose = False

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    @classmethod
    def setup(cls, verbose: bool = False):
        """Setup logger with console handler"""
        cls._verbose = verbose
        if cls._logger is None:
            logger = logging.getLogger(__name__)
            logger.setLevel(logging.DEBUG if verbose else logging.INFO)
            logger.handlers = []

            console_handler = logging.StreamHandler()
            console_handler.setFormatter(ColorFormatter())
            console_handler.setLevel(logging.DEBUG if verbose else logging.INFO)
            logger.addHandler(console_handler)
            cls._logger = logger
        return cls._logger

    @classmethod
    def get(cls):
        """Get logger instance"""
        if cls._logger is None:
            cls.setup()
        return cls._logger

    # Simplified logging methods
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

    @classmethod
    def critical(cls, message: str):
        cls.get().critical(message)

    @classmethod
    def exception(cls, message: str):
        cls.get().exception(message)


class ConfigManager:
    """Handles API key storage and retrieval with caching"""
    _config_cache = None
    _config_path = None

    @classmethod
    def _get_config_path(cls):
        """Get config file path with caching"""
        if cls._config_path == None:
            cls._config_path = os.path.join(DATA_DIR, CONFIG_FILE)
        return cls._config_path

    @classmethod
    def load_config(cls) -> Dict:
        """Load configuration with caching"""
        if cls._config_cache != None:
            return cls._config_cache

        config_path = cls._get_config_path()
        if os.path.exists(config_path):
            try:
                with open(config_path, 'r') as f:
                    cls._config_cache = json.load(f)
                    return cls._config_cache
            except json.JSONDecodeError as e:
                Logger.error(f"Config file is corrupted: {e}")
                cls._config_cache = {'api_keys': []}
                return cls._config_cache
        cls._config_cache = {'api_keys': []}
        return cls._config_cache

    @classmethod
    def save_config(cls, keys: List[str]):
        """Save API keys and update cache"""
        if not os.path.exists(DATA_DIR):
            os.makedirs(DATA_DIR, exist_ok=True)

        config_path = cls._get_config_path()
        try:
            with open(config_path, 'w') as f:
                json.dump({'api_keys': keys}, f)
            cls._config_cache = {'api_keys': keys}
        except IOError as e:
            Logger.error(f"Failed to save config: {e}")

    @classmethod
    def add_key(cls, key: str) -> bool:
        """Add a new API key to config"""
        config = cls.load_config()
        keys = config.get('api_keys', [])

        if key in keys:
            return False

        if len(keys) >= MAX_API_KEYS:
            keys.pop(0)

        keys.append(key)
        cls.save_config(keys)
        return True

    @classmethod
    def clear_cache(cls):
        """Clear config cache"""
        cls._config_cache = None


class CivitAIAPI:
    """Handles communication with CivitAI API with connection pooling"""

    def __init__(self, civitai_tokens: Optional[List[str]] = None):
        self.tokens = civitai_tokens or []
        self.base_url = 'https://civitai.com/api/v1'
        self.current_token = None
        self.session = None
        self._connector = None

    async def __aenter__(self):
        """Setup connection pool and session"""
        try:
            # Create connection pool
            self._connector = aiohttp.TCPConnector(
                limit=MAX_CONCURRENT_REQUESTS * 2,
                limit_per_host=MAX_CONCURRENT_REQUESTS,
                ttl_dns_cache=300,
                use_dns_cache=True
            )
            self.session = aiohttp.ClientSession(
                connector=self._connector,
                timeout=aiohttp.ClientTimeout(total=30, connect=10)
            )
            return self
        except Exception as e:
            Logger.error(f"Failed to create HTTP session: {e}")
            raise

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Cleanup session and connector"""
        if self.session:
            await self.session.close()
        if self._connector:
            await self._connector.close()

    def _get_random_token(self) -> Optional[str]:
        """Get a random API token"""
        if not self.tokens:
            return None
        self.current_token = random.choice(self.tokens)
        return self.current_token

    def _prepare_headers(self) -> Dict[str, str]:
        """Prepare headers with API token"""
        headers = {'User-Agent': 'CivitaiLink:Automatic1111'}
        token = self._get_random_token()
        if token:
            headers['Authorization'] = f'Bearer {token}'
        return headers

    async def _fetch_json(self, url: str) -> Optional[Dict]:
        """Make async HTTP request with retry logic"""
        max_retries = 3
        for attempt in range(max_retries):
            try:
                headers = self._prepare_headers()
                async with self.session.get(url, headers=headers) as response:
                    response.raise_for_status()
                    return await response.json()
            except aiohttp.ClientConnectorError as e:
                Logger.error(f"Connection error (attempt {attempt + 1}): {e}")
                if attempt == max_retries - 1:
                    raise ConnectionError("Failed to connect to CivitAI. Please check your internet connection.")
                await asyncio.sleep(1)
            except aiohttp.ClientResponseError as e:
                token_display = f"...{self.current_token[-API_KEY_DISPLAY_CHARS:]}" if self.current_token else 'None'
                Logger.error(f"HTTP Error {e.status} (Token: {token_display}): {e.message}")
                if e.status == 429:  # Rate limit
                    await asyncio.sleep(2 ** attempt)
                else:
                    return None
            except asyncio.TimeoutError:
                Logger.error(f"Request timed out (attempt {attempt + 1})")
                if attempt == max_retries - 1:
                    return None
                await asyncio.sleep(1)
            except Exception as e:
                Logger.error(f"Unexpected error in _fetch_json: {e}")
                return None
        return None

    async def _get_model_data(self, url: str) -> Tuple[Optional[Dict], Optional[str]]:
        """Fetch model data from API"""
        # Note: currently, 'source' is not used anywhere - but this is for future use?
        try:
            # Handle model page URLs
            if "civitai.com/models/" in url:
                if '?modelVersionId=' in url:
                    base_url = url.split('?')[0]
                    model_id = re.search(r'/models/(\d+)', base_url).group(1)
                    version_id = url.split('modelVersionId=')[1].split('&')[0]
                    source = 'url'
                else:
                    model_id = re.search(r'/models/(\d+)', url).group(1)
                    model_data = await self._fetch_json(f"{self.base_url}/models/{model_id}")
                    if not model_data:
                        return None, None
                    version_id = model_data['modelVersions'][0].get('id')
                    source = 'api'

            # Handle direct download URLs
            elif '/api/download/models/' in url:
                version_part = url.split('/api/download/models/')[1]
                version_id = version_part.split('?')[0].split('/')[0]
                source = 'url'
            else:
                return None, None

            data = await self._fetch_json(f"{self.base_url}/model-versions/{version_id}")
            return data, source
        except (KeyError, IndexError, AttributeError) as e:
            token_display = f"...{self.current_token[-API_KEY_DISPLAY_CHARS:]}" if self.current_token else 'None'
            Logger.error(f"API Error (Token: {token_display}): {e}")
            return None, None
        except Exception as e:
            Logger.error(f"Unexpected error in _get_model_data: {e}")
            return None, None

    async def fetch_data(self, url: str) -> Tuple[Optional[Dict], Optional[str]]:
        """Public method to fetch model data"""
        return await self._get_model_data(url)

    async def fetch_model_info(self, model_id: str) -> Optional[Dict]:
        """Fetch detailed model information"""
        url = f"{self.base_url}/models/{model_id}"
        return await self._fetch_json(url)

    async def validate_model_id(self, model_id: str) -> bool:
        """Validate if a model ID exists and is accessible"""
        try:
            # Check if model_id is a valid number
            if not model_id.isdigit():
                return False
            
            # Try to fetch model info to verify it exists
            model_data = await self.fetch_model_info(model_id)
            return model_data != None
        except Exception as e:
            Logger.debug(f"Error validating model ID {model_id}: {e}")
            return False

    async def batch_fetch(self, urls: List[str]) -> List[Tuple[Optional[Dict], Optional[str]]]:
        """Process multiple URLs concurrently"""
        semaphore = asyncio.Semaphore(MAX_CONCURRENT_REQUESTS)

        async def limited_fetch(url):
            async with semaphore:
                try:
                    return await self.fetch_data(url)
                except Exception as e:
                    Logger.error(f"Error processing URL {url}: {e}")
                    return None, None

        # Create tasks and gather results
        tasks = [limited_fetch(url) for url in urls]
        return await asyncio.gather(*tasks, return_exceptions=True)


class URLProcessor:
    """Handles URL processing and validation"""

    @staticmethod
    def is_valid_civitai_url(url: str) -> bool:
        """Check if URL is a valid CivitAI URL"""
        if not url.startswith(('http://', 'https://')):
            return False
        if 'civitai.com' not in url:
            return False
        return True

    @staticmethod
    def extract_urls_from_text(content: str) -> Set[str]:
        """Extract URLs from text content using regex"""
        urls = set()
        matches = re.findall(URL_REGEX, content)
        for url in matches:
            clean_url = url.rstrip('.,;')
            if URLProcessor.is_valid_civitai_url(clean_url):
                urls.add(clean_url)
            else:
                Logger.debug(f"Skipping non-CivitAI URL: {clean_url}")
        return urls

    @staticmethod
    def extract_urls_from_dict(data: Dict) -> Set[str]:
        """Recursively extract URLs from nested dictionary structures"""
        urls = set()

        if isinstance(data, dict):
            if 'url' in data:
                url = data['url']
                if URLProcessor.is_valid_civitai_url(url):
                    urls.add(url)
                else:
                    Logger.debug(f"Skipping non-CivitAI URL in dict: {url}")
            for value in data.values():
                urls.update(URLProcessor.extract_urls_from_dict(value))
        elif isinstance(data, (list, tuple, set)):
            for item in data:
                urls.update(URLProcessor.extract_urls_from_dict(item))

        return urls

    @staticmethod
    def extract_urls_from_python_file(file_path: Path) -> Dict[str, Any]:
        """Load model dictionaries from Python file"""
        try:
            sys.dont_write_bytecode = True
            spec = importlib.util.spec_from_file_location('models_module', file_path)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)

            models = {}
            for var in dir(module):
                if not var.startswith('_'):
                    attr = getattr(module, var)
                    if isinstance(attr, dict):
                        urls = URLProcessor.extract_urls_from_dict(attr)
                        if urls:
                            models[var] = [{'url': url} for url in urls]
                            Logger.debug(f"Found {len(urls)} URLs in dictionary '{var}'")

            return models
        except Exception as e:
            Logger.error(f"Error loading Python file: {e}")
            return {}

    @staticmethod
    def process_inputs(args: argparse.Namespace) -> Set[str]:
        """Combine input sources into unique URLs and model IDs"""
        urls = set()
        model_ids = set()

        # Handle positional arguments (URLs or model IDs without -i flag)
        if args.urls:
            for item in args.urls:
                for u in re.split(r'[,	\s]+', item):
                    if URLProcessor.is_valid_civitai_url(u):
                        urls.add(u)
                    elif u.isdigit():
                        model_ids.add(u)
                    else:
                        Logger.debug(f"Skipping invalid URL or model ID from args: {u}")

        # Handle -i/--input flag
        if args.input:
            for item in args.input:
                for u in re.split(r'[,	\s]+', item):
                    if URLProcessor.is_valid_civitai_url(u):
                        urls.add(u)
                    elif u.isdigit():
                        model_ids.add(u)
                    else:
                        Logger.debug(f"Skipping invalid URL or model ID from input: {u}")

        if args.file:
            try:
                file_urls = FileProcessor.process_file(args.file)
                urls.update(file_urls)
            except Exception as e:
                Logger.error(f"Error processing file: {e}")

        # Add model IDs as-is (they will be handled separately in process_urls)
        urls.update(model_ids)

        Logger.debug(f"Model IDs found: {len(model_ids)}")
        Logger.debug(f"Total unique URLs and model IDs to process: {len(urls)}")
        for url in sorted(urls):
            Logger.debug(f"URL/ID to process: {url}")

        return urls


class FileProcessor:
    """Handles file processing"""

    @staticmethod
    def process_file(file_path: str) -> Set[str]:
        """Extract model URLs from supported files"""
        path = Path(file_path)

        if not path.exists():
            found_path = FileProcessor._find_file_without_extension(file_path)
            if found_path:
                path = found_path
                Logger.info(f"Using file: {path.name}")
            else:
                Logger.error(f"No matching files found for: {file_path}")
                return set()

        urls = set()

        Logger.debug(f"Processing file: {path.name}")

        try:
            if path.suffix == '.json':
                with open(path, 'r') as f:
                    data = json.load(f)
                extracted_urls = URLProcessor.extract_urls_from_dict(data)
                urls.update(extracted_urls)
                Logger.debug(f"Found {len(extracted_urls)} URLs in JSON file")

            elif path.suffix == '.py':
                model_data = URLProcessor.extract_urls_from_python_file(path)
                if not model_data:
                    Logger.error('No valid model URLs found in Python file')
                    return urls

                selected = FileProcessor._select_model_groups(model_data)
                for group in selected:
                    group_urls = [model['url'] for model in model_data[group]]
                    urls.update(group_urls)
                    Logger.debug(f"Selected {len(group_urls)} URLs from group '{group}'")

            elif path.suffix == '.txt':
                with open(path, 'r', encoding='utf-8') as f:
                    content = f.read()
                extracted_urls = URLProcessor.extract_urls_from_text(content)
                Logger.info(f"Found {len(extracted_urls)} unique URLs in {path.name}")
                urls.update(extracted_urls)
                for url in extracted_urls:
                    Logger.debug(f"Extracted URL: {url}")

            else:
                Logger.error(f"Unsupported file type: {path.suffix}")

        except json.JSONDecodeError as e:
            Logger.error(f"Invalid JSON format: {e}")
        except UnicodeDecodeError:
            Logger.error('File encoding error. Please use UTF-8 encoded text files.')
        except Exception as e:
            Logger.error(f"Error processing file {path.name}: {e}")

        Logger.debug(f"Total valid URLs found in file{'s' if len(urls) != 1 else ''}: {len(urls)}")

        return urls

    @staticmethod
    def _find_file_without_extension(base_name: str) -> Optional[Path]:
        """Find file with any extension matching base name"""
        directory = Path('.')
        matches = list(directory.glob(f"{base_name}.*"))

        if not matches:
            return None

        if len(matches) == 1:
            return matches[0]

        print(f"\n{Fore.MAGENTA}Multiple files found:{Style.RESET_ALL}")
        for idx, file in enumerate(matches, 1):
            print(f"{idx}. {file.name}")

        try:
            choice = input(f"\n{Fore.YELLOW}Select file (comma-separated/all): {Style.RESET_ALL}").strip()
            if choice.lower() == 'all':
                return matches[0]

            selected = []
            for c in choice.split(','):
                if c.strip().isdigit() and 1 <= int(c) <= len(matches):
                    selected.append(matches[int(c) - 1])
            return selected[0] if selected else None
        except KeyboardInterrupt:
            Logger.error('\nOperation cancelled!')
            sys.exit(1)
        except Exception as e:
            Logger.error(f"Error in file selection: {e}")
            return None

    @staticmethod
    def _select_model_groups(model_data: Dict[str, Any]) -> List[str]:
        """Interactive selection of model groups"""
        print(f"\n{Fore.MAGENTA}Available Model Groups:{Style.RESET_ALL}")
        for idx, name in enumerate(model_data.keys(), 1):
            print(f"{Fore.BLUE}{idx:2}.{Style.RESET_ALL} {name}")

        try:
            choice = input(f"\n{Fore.YELLOW}Select groups (comma-separated/all): {Style.RESET_ALL}").strip()
            if choice.lower() == 'all':
                return list(model_data.keys())

            selected = []
            for c in choice.split(','):
                if c.strip().isdigit() and 1 <= int(c) <= len(model_data):
                    selected.append(list(model_data.keys())[int(c) - 1])
            return selected
        except KeyboardInterrupt:
            Logger.error('\nOperation cancelled!')
            sys.exit(1)
        except Exception as e:
            Logger.error(f"Error in group selection: {e}")
            return []


class DataExporter:
    """Handles data export functionality"""

    @staticmethod
    def format_size(bytes_size: int) -> str:
        """Convert bytes to human-readable format"""
        if not bytes_size:
            return 'N/A'

        for unit in ['B', 'KB', 'MB', 'GB']:
            if bytes_size < 1024.0:
                return f"{bytes_size:.1f} {unit}"
            bytes_size /= 1024.0
        return f"{bytes_size:.1f} GB"

    @staticmethod
    def prepare_export_data(model_data: Dict, version_data: Dict, source: str) -> Dict:
        """Prepare structured data for export"""
        model_id = model_data.get('modelId', model_data.get('id', 'N/A'))

        model_info = {
            'model_id': model_id,
            'model_name': model_data.get('name', 'N/A'),
            'model_type': model_data.get('type', 'N/A'),
            'url': f"https://civitai.com/models/{model_id}",
            'versions': []
        }

        for version in model_data.get('modelVersions', []):
            files = version.get('files', [{}])
            version_info = {
                'version_id': version.get('id', 'N/A'),
                'version_name': version.get('name', 'N/A'),
                'created_at': version.get('createdAt', 'N/A'),
                'updated_at': version.get('updatedAt', 'N/A'),
                'download_url': version.get('downloadUrl', 'N/A'),
                'early_access': version.get('availability') == 'EarlyAccess',
                'size': DataExporter.format_size(files[0].get('sizeKB', 0) * 1024 if files else 0),
                'base_model': version.get('baseModel', 'N/A'),
                'trained_words': version.get('trainedWords', []),
                'url': f"https://civitai.com/models/{model_id}?modelVersionId={version.get('id', '')}"
            }
            model_info['versions'].append(version_info)

        return model_info

    @staticmethod
    def export_to_json(data: List[Dict], filename: str):
        """Export model data to JSON file"""
        if not data:
            Logger.warning('No data to export')
            return

        try:
            if not filename.endswith('.json'):
                filename += '.json'

            if filename == 'civitai_export.json':
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"civitai_export_{timestamp}.json"

            export_dir = os.path.join(DATA_DIR, 'exports')
            os.makedirs(export_dir, exist_ok=True)
            export_path = os.path.join(export_dir, filename)

            # Handle existing files
            if os.path.exists(export_path):
                with open(export_path, 'r', encoding='utf-8') as f:
                    existing_data = json.load(f)

                existing_ids = {item['model_id'] for item in existing_data}
                new_data = [item for item in data if item['model_id'] not in existing_ids]

                if not new_data:
                    Logger.info('No new data to add to existing export file')
                    return

                data = existing_data + new_data
                Logger.info(f"Adding {len(new_data)} new models to existing export file")

            with open(export_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            Logger.info(f"Successfully exported to: {export_path}")
            Logger.debug(f"Exported {len(data)} model records")
        except IOError as e:
            Logger.error(f"File I/O error during export: {e}")
        except Exception as e:
            Logger.error(f"Failed to export JSON: {e}")


class DisplayManager:
    """Handles console output and formatting"""

    @staticmethod
    def format_colored_column(label: str, value: str, width: int, is_current: bool = False) -> str:
        """Format a column with colored label and white value"""
        color = Fore.YELLOW if is_current else Fore.CYAN
        return f"{color}{label}:{Style.RESET_ALL} {value.ljust(width - len(label) - 2)}"

    @staticmethod
    def print_model_header(model_data: Dict, version_data: Optional[Dict] = None, source: str = 'api'):
        """Display model header information"""
        model_name = model_data.get('name', 'Unknown Model')
        model_id = model_data.get('id', 'N/A')
        model_type = model_data.get('type', 'N/A')

        print(f"\n{Fore.MAGENTA}Model:{Style.RESET_ALL} {Fore.CYAN}{model_name}{Style.RESET_ALL}")

        if version_data:
            version_name = version_data.get('name', 'Unknown Version')
            model_url = f"https://civitai.com/models/{model_id}"

            print(f"{Fore.MAGENTA}Version:{Style.RESET_ALL} {Fore.CYAN}{version_name}{Style.RESET_ALL}")
            print(f"{Fore.MAGENTA}Type:{Style.RESET_ALL} {Fore.CYAN}{model_type}{Style.RESET_ALL}")
            print(f"{Fore.MAGENTA}Model URL:{Style.RESET_ALL} {model_url}")

        print(f"{Fore.MAGENTA}Model ID:{Style.RESET_ALL} {Fore.CYAN}{model_id}{Style.RESET_ALL}")

    @staticmethod
    def print_version_info(version: Dict, idx: int, model_id: str, current_id: str, source: str, args: argparse.Namespace):
        """Display formatted version information"""
        version_id = version.get('id', 'N/A')
        name = version.get('name', 'Unknown')[:COLUMN_WIDTH['name']]
        download_url = version.get('downloadUrl', 'N/A')
        files = version.get('files', [{}])

        is_current = version_id == current_id
        is_latest = idx == 0

        columns = [
            DisplayManager.format_colored_column('Ver.ID', str(version_id), COLUMN_WIDTH['id'], is_current),
            DisplayManager.format_colored_column('Index', str(idx), COLUMN_WIDTH['index'], is_current),
            DisplayManager.format_colored_column('Name', name, COLUMN_WIDTH['name'], is_current)
        ]

        if args.show_url:
            url = f"https://civitai.com/models/{model_id}?modelVersionId={version_id}"
            columns.append(DisplayManager.format_colored_column('URL', url, COLUMN_WIDTH['url'], is_current))

        if args.show_download and download_url != 'N/A':
            columns.append(DisplayManager.format_colored_column('Download', download_url, COLUMN_WIDTH['download'], is_current))

        line = ' | '.join(columns)

        if is_latest:
            line += f" {Fore.GREEN}<< Latest{Style.RESET_ALL}"
        elif is_current:
            line += f" {Fore.YELLOW}<< Current{Style.RESET_ALL}"
        if version.get('availability') == 'EarlyAccess':
            line += f" {Fore.LIGHTYELLOW_EX}<< Paid{Style.RESET_ALL}"

        print(line)

    @staticmethod
    def print_file_format_help():
        """Display file format examples"""
        print(f"\n{Fore.MAGENTA}File Format Requirements:{Style.RESET_ALL}")
        print(f"{Fore.GREEN}JSON Example:{Style.RESET_ALL}")
        print('''{
    "My Models 1": [
        {"url": "https://...", "name": "model1.safetensors"},
        {"url": "https://...", "name": "model2.safetensors"}
    ],
    "My Models 2": [
        {"url": "https://...", "name": "model3.safetensors"}
    ]
}''')

        print(f"\n{Fore.GREEN}Python Example:{Style.RESET_ALL}")
        print('''model_list = {
    "Category 1": [
        {"url": "https://...", "name": "modelA.safetensors"}
    ]
}

another_collection = {
    "Category 2": [
        {"url": "https://...", "name": "modelB.safetensors"}
    ]
}''')

        print(f"\n{Fore.GREEN}Text File Example:{Style.RESET_ALL}")
        print('''https://civitai.com/models/1234
https://civitai.com/models/5678 model1.safetensors
https://civitai.com/models/9012, https://civitai.com/models/3456''')

        print(f"\n{Fore.YELLOW}Multiple dictionaries are supported. Names can vary.{Style.RESET_ALL}")


class APIKeyManager:
    """Handles API key management"""

    @staticmethod
    def validate_api_key(key: str) -> bool:
        """Validate CivitAI API key format"""
        return len(key) == 32 and key.isalnum()

    @staticmethod
    def get_api_key() -> List[str]:
        """Retrieve and validate API keys"""
        config = ConfigManager.load_config()
        keys = config.get('api_keys', [])

        if keys:
            Logger.info(f"Using {len(keys)} saved API key{'s' if len(keys) != 1 else ''}. Use --change-key (-k) to update")
            return keys

        while True:
            try:
                key = input(f"{Fore.CYAN}Enter CivitAI API key (32 chars): {Style.RESET_ALL}").strip()
                if APIKeyManager.validate_api_key(key):
                    ConfigManager.add_key(key)
                    return [key]
                Logger.error('Invalid key format! Must be 32 alphanumeric characters')
            except KeyboardInterrupt:
                Logger.error("\nOperation cancelled!")
                sys.exit(1)
            except Exception as e:
                Logger.error(f"Error getting API key: {e}")
                sys.exit(1)

    @staticmethod
    def manage_api_keys():
        """Manage API keys (add/remove/list)"""
        config = ConfigManager.load_config()
        keys = config.get('api_keys', [])

        while True:
            print(f"\n{Fore.MAGENTA}Current API Keys ({len(keys)}/{MAX_API_KEYS}):{Style.RESET_ALL}")
            for i, key in enumerate(keys, 1):
                print(f"{i}. ...{key[-API_KEY_DISPLAY_CHARS:]}")

            print('\nOptions:')
            print('1. Add new key')
            print('2. Remove key')
            print('3. Exit')

            print() # Add extra space between options
            try:
                choice = input(f"{Fore.YELLOW}Select option: {Style.RESET_ALL}").strip()

                if choice == '1':
                    if len(keys) >= MAX_API_KEYS:
                        Logger.error(f"Maximum {MAX_API_KEYS} keys reached!")
                        continue

                    key = input(f"{Fore.CYAN}Enter new API key: {Style.RESET_ALL}").strip()
                    if APIKeyManager.validate_api_key(key):
                        if key in keys:
                            Logger.warning('Key already exists!')
                        else:
                            ConfigManager.add_key(key)
                            keys = ConfigManager.load_config().get('api_keys', [])
                            Logger.info(f"{Fore.GREEN}Key added! Total keys: {len(keys)}{Style.RESET_ALL}")
                    else:
                        Logger.error('Invalid key format!')

                elif choice == '2':
                    if not keys:
                        Logger.error('No keys to remove!')
                        continue

                    idx = input(f"{Fore.CYAN}Enter key number to remove: {Style.RESET_ALL}").strip()
                    if idx.isdigit() and 1 <= int(idx) <= len(keys):
                        removed_key = keys.pop(int(idx) - 1)
                        ConfigManager.save_config(keys)
                        Logger.info(f"{Fore.GREEN}Removed key ...{removed_key[-API_KEY_DISPLAY_CHARS:]}{Style.RESET_ALL}")
                    else:
                        Logger.error('Invalid selection!')

                elif choice == '3':
                    break

                else:
                    Logger.error('Invalid option!')
            except KeyboardInterrupt:
                Logger.error("\nOperation cancelled!")
                break
            except Exception as e:
                Logger.error(f"Error in key management: {e}")


async def process_urls(api: CivitAIAPI, urls: Set[str], args: argparse.Namespace) -> List[Dict]:
    """Process all URLs and return collected data"""
    results = []
    valid_urls = []
    model_ids_to_validate = []
    
    for url in sorted(urls):
        if URLProcessor.is_valid_civitai_url(url):
            valid_urls.append(url)
        else:
            # If it doesn't start with http, treat it as a model ID and validate
            if url.isdigit():
                model_ids_to_validate.append(url)
            else:
                Logger.warning(f"Skipping invalid URL or model ID: {url}")

    # Validate model IDs
    if model_ids_to_validate:
        Logger.info(f"Validating {len(model_ids_to_validate)} model ID(s)...")
        for model_id in model_ids_to_validate:
            if await api.validate_model_id(model_id):
                valid_urls.append(f"https://civitai.com/models/{model_id}")
                Logger.info(f"Model ID {Fore.CYAN}{model_id}{Style.RESET_ALL} is valid")
            else:
                Logger.debug(f"Model ID {Fore.CYAN}{model_id}{Style.RESET_ALL} not found or invalid")

    if not valid_urls:
        Logger.error('No valid URLs to process')
        return []

    Logger.debug(f"Starting batch processing of {len(valid_urls)} URLs\n")

    try:
        # Process URLs in batches
        batch_results = await api.batch_fetch(valid_urls)

        for url, result in zip(valid_urls, batch_results):
            if isinstance(result, Exception):
                Logger.error(f"Exception processing URL {url}: {result}")
                continue

            model_data, source = result
            if not model_data:
                Logger.warning(f"Skipping invalid URL: {url}")
                continue

            # Print a red line for section separation if there are 3 or more results
            if len(valid_urls) >= 3:
                print(f"{Fore.RED}{'-' * 60}{Style.RESET_ALL}")
            Logger.info(f"\n{Fore.MAGENTA}>> Processing:{Style.RESET_ALL} {url}")
            model_id = model_data.get('modelId')
            current_version_id = model_data.get('id')

            if model_id:
                Logger.debug(f"Fetching detailed info for model ID: {model_id}")

                detailed_data = await api.fetch_model_info(model_id)
                if detailed_data:
                    DisplayManager.print_model_header(detailed_data, model_data, source)
                    model_data = detailed_data
                else:
                    Logger.warning("Couldn't fetch detailed model info")

            versions = model_data.get('modelVersions', [])
            if not versions:
                Logger.warning('No versions available')
                continue

            if len(versions) > 1:
                Logger.debug(f"Found {len(versions)} versions for model")

            print(f"\n{Fore.GREEN}>> Available Versions:{Style.RESET_ALL}\n")
            for idx, version in enumerate(versions):
                DisplayManager.print_version_info(
                    version=version,
                    idx=idx,
                    model_id=model_id,
                    current_id=current_version_id,
                    source=source,
                    args=args
                )

            # Prepare data for export
            if args.export:
                export_data = DataExporter.prepare_export_data(model_data, model_data, source)
                results.append(export_data)

            print()  # Add extra space between models
    except ConnectionError as e:
        Logger.error(f"Network error: {e}")
        raise
    except Exception as e:
        Logger.error(f"Error processing URLs: {e}")
        Logger.debug(f"Error details: {type(e)} - {str(e)}")

    Logger.debug(f"Finished processing {len(valid_urls)} URLs")

    return results


def main():
    """Main application flow"""
    parser = argparse.ArgumentParser(
        description='CivitAI Model Information Fetcher - Supports URLs and Model IDs',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    # Positional arguments (URLs without flag)
    parser.add_argument('urls', nargs='*',
                       help='Input URLs or model IDs (can be used without -i flag)')
    # Optional arguments
    parser.add_argument('-i', '--input', nargs='+',
                       help='Additional input URLs or model IDs (space/comma separated)')
    parser.add_argument('-f', '--file',
                       help='Path to model list file (JSON/Python/TXT) - extension can be omitted')
    parser.add_argument('-u', '--show-url', action='store_true',
                       help='Display model URLs')
    parser.add_argument('-d', '--show-download', action='store_true',
                       help='Display download URLs')
    parser.add_argument('-k', '--change-key', action='store_true',
                       help='Manage API keys (add/remove)')
    parser.add_argument('--show-help', action='store_true',
                       help='Show file format examples')
    parser.add_argument('--export', nargs='?', const='civitai_export.json',
                       help='Export data to JSON (default: civitai_export.json)')
    parser.add_argument('-v', '--verbose', action='store_true',
                       help='Enable verbose logging to console')

    args = parser.parse_args()

    # Setup logger with verbosity setting
    Logger.setup(args.verbose)

    if args.show_help:
        DisplayManager.print_file_format_help()
        return

    if args.change_key:
        APIKeyManager.manage_api_keys()
        return

    try:
        # Disable pycache generation
        sys.dont_write_bytecode = True

        api_keys = APIKeyManager.get_api_key()
        model_urls = URLProcessor.process_inputs(args)

        if not model_urls:
            Logger.error('No valid input sources provided!')
            print(f"\n{Fore.YELLOW}Usage examples:{Style.RESET_ALL}")
            print('  python script.py url1 url2')
            print('  python script.py 12345 67890  # Model IDs')
            print('  python script.py -i url1 url2 -f models.json')
            print('  python script.py --show-help')
            sys.exit(1)

        async def run():
            try:
                async with CivitAIAPI(api_keys) as api:
                    export_data = await process_urls(api, model_urls, args)

                    if args.export and export_data:
                        DataExporter.export_to_json(export_data, args.export)
            except aiohttp.ClientError as e:
                Logger.error(f"Network error occurred: {e}")
                Logger.error('Please check your internet connection and try again.')
            except Exception as e:
                Logger.error(f"Unexpected error: {e}")
                if args.verbose:
                    Logger.exception('Full error details:')

        asyncio.run(run())

    except KeyboardInterrupt:
        Logger.error('\nOperation cancelled by user!')
        sys.exit(1)
    except Exception as e:
        Logger.error(f"Critical error: {e}")
        if args.verbose:
            Logger.exception('Full error details:')
        sys.exit(1)


if __name__ == '__main__':
    main()
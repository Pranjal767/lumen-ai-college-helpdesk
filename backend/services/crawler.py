"""Safe first-pass HTML link discovery for an approved college host."""

import time
from dataclasses import dataclass
from urllib.parse import urldefrag, urljoin, urlsplit, urlunsplit
from urllib.robotparser import RobotFileParser

import requests
from bs4 import BeautifulSoup


class CrawlError(RuntimeError):
    """Raised when a page cannot be fetched within crawler safety rules."""


@dataclass(frozen=True)
class CrawledPage:
    """A fetched HTML page and its same-host links."""

    url: str
    title: str
    links: list[str]


@dataclass(frozen=True)
class FetchedResource:
    """A bounded response body returned by an approved source."""

    url: str
    content_type: str
    content: bytes


def normalize_allowed_url(url: str, allowed_domain: str) -> str:
    """Normalize a URL or reject it if it leaves the approved HTTPS host."""
    try:
        parsed = urlsplit(url.strip())
        hostname = parsed.hostname
        port = parsed.port
    except ValueError as error:
        raise CrawlError("URL is malformed.") from error

    domain = allowed_domain.lower().strip().rstrip(".")

    if (
        parsed.scheme.lower() != "https"
        or hostname is None
        or hostname.lower() != domain
        or parsed.username is not None
        or parsed.password is not None
        or port not in (None, 443)
    ):
        raise CrawlError("URL must use HTTPS on the approved college domain.")

    path = parsed.path or "/"
    normalized = urlunsplit(
        ("https", domain, path, parsed.query, "")
    )
    return urldefrag(normalized)[0]


class SafeCrawler:
    """Fetch one approved page and discover links without recursive crawling."""

    _REDIRECT_STATUSES = {301, 302, 303, 307, 308}
    _MAX_REDIRECTS = 5

    def __init__(
        self,
        allowed_domain: str,
        *,
        timeout: float = 15,
        delay_seconds: float = 1,
        max_response_bytes: int = 5 * 1024 * 1024,
        user_agent: str = "LumenAI-CollegeHelpdesk/1.0",
    ) -> None:
        if not allowed_domain.strip():
            raise ValueError("An approved college domain is required.")
        if timeout <= 0:
            raise ValueError("The request timeout must be greater than zero.")
        if delay_seconds < 0:
            raise ValueError("The crawl delay cannot be negative.")
        if max_response_bytes < 1:
            raise ValueError("The response size limit must be at least one byte.")

        self.allowed_domain = allowed_domain.lower().strip().rstrip(".")
        self.timeout = timeout
        self.delay_seconds = delay_seconds
        self.max_response_bytes = max_response_bytes
        self.user_agent = user_agent
        self._robots_parser: RobotFileParser | None = None
        self._last_request_at: float | None = None

    def discover_links(self, url: str) -> CrawledPage:
        """Fetch one robots-allowed page and return its same-host links."""
        page_url = normalize_allowed_url(url, self.allowed_domain)
        self._load_robots_policy()
        if not self._robots_parser.can_fetch(self.user_agent, page_url):
            raise CrawlError("robots.txt does not allow fetching this page.")

        response = self._request_with_safe_redirects(
            page_url, max_bytes=self.max_response_bytes
        )
        content_type = response.headers.get("Content-Type", "").lower()
        if "text/html" not in content_type:
            raise CrawlError("The approved URL did not return an HTML page.")

        soup = BeautifulSoup(response.text, "html.parser")
        title_element = soup.find("title")
        title = title_element.get_text(" ", strip=True) if title_element else ""

        links: set[str] = set()
        for anchor in soup.find_all("a", href=True):
            candidate = urljoin(response.url, anchor["href"])
            try:
                links.add(normalize_allowed_url(candidate, self.allowed_domain))
            except CrawlError:
                continue

        return CrawledPage(
            url=response.url,
            title=title,
            links=sorted(links),
        )

    def fetch_resource(self, url: str, *, max_bytes: int) -> FetchedResource:
        """Fetch one approved resource, enforcing robots and a byte limit."""
        if max_bytes < 1:
            raise ValueError("The response size limit must be at least one byte.")
        resource_url = normalize_allowed_url(url, self.allowed_domain)
        self._load_robots_policy()
        if not self._robots_parser.can_fetch(self.user_agent, resource_url):
            raise CrawlError("robots.txt does not allow fetching this resource.")

        response = self._request_with_safe_redirects(
            resource_url, max_bytes=max_bytes
        )
        return FetchedResource(
            url=response.url,
            content_type=response.headers.get("Content-Type", "").lower(),
            content=response.content,
        )

    def _load_robots_policy(self) -> None:
        if self._robots_parser is not None:
            return

        robots_url = f"https://{self.allowed_domain}/robots.txt"
        response = self._request_with_safe_redirects(
            robots_url,
            allow_missing=True,
            max_bytes=self.max_response_bytes,
        )
        if response.status_code in (404, 410):
            robots_lines: list[str] = []
        elif response.status_code < 200 or response.status_code >= 300:
            raise CrawlError(
                f"Could not read robots.txt (HTTP {response.status_code})."
            )
        else:
            robots_lines = response.text.splitlines()

        parser = RobotFileParser(robots_url)
        parser.parse(robots_lines)
        self._robots_parser = parser

    def _request_with_safe_redirects(
        self,
        url: str,
        *,
        allow_missing: bool = False,
        max_bytes: int,
    ) -> requests.Response:
        current_url = normalize_allowed_url(url, self.allowed_domain)

        for redirect_count in range(self._MAX_REDIRECTS + 1):
            self._wait_before_request()
            try:
                response = requests.get(
                    current_url,
                    headers={"User-Agent": self.user_agent},
                    timeout=self.timeout,
                    allow_redirects=False,
                    stream=True,
                )
            except requests.RequestException as error:
                raise CrawlError(f"Request failed for {current_url}: {error}") from error

            try:
                if response.status_code not in self._REDIRECT_STATUSES:
                    if response.status_code < 200 or response.status_code >= 400:
                        if allow_missing and response.status_code in (404, 410):
                            response.url = current_url
                            self._read_limited_body(response, max_bytes)
                            return response
                        raise CrawlError(
                            f"Request for {current_url} returned HTTP "
                            f"{response.status_code}."
                        )
                    response.url = current_url
                    self._read_limited_body(
                        response, max_bytes, max_seconds=self.timeout * 2
                    )
                    return response

                if redirect_count == self._MAX_REDIRECTS:
                    raise CrawlError("The page exceeded the redirect limit.")

                location = response.headers.get("Location")
                if not location:
                    raise CrawlError("The server returned a redirect without a location.")
                current_url = normalize_allowed_url(
                    urljoin(current_url, location), self.allowed_domain
                )
            except CrawlError:
                response.close()
                raise
            finally:
                if response.status_code in self._REDIRECT_STATUSES:
                    response.close()

        raise CrawlError("The page exceeded the redirect limit.")

    @staticmethod
    def _read_limited_body(
        response: requests.Response, max_bytes: int, *, max_seconds: float = 30
    ) -> None:
        """Read a streamed body into memory only while it stays within its cap."""
        content_length = response.headers.get("Content-Length")
        if content_length is not None:
            try:
                declared_size = int(content_length)
            except ValueError as error:
                response.close()
                raise CrawlError("The server returned an invalid Content-Length.") from error
            if declared_size < 0 or declared_size > max_bytes:
                response.close()
                raise CrawlError("The response exceeds the configured size limit.")

        chunks: list[bytes] = []
        total_size = 0
        deadline = time.monotonic() + max_seconds
        try:
            for chunk in response.iter_content(chunk_size=64 * 1024):
                if time.monotonic() > deadline:
                    response.close()
                    raise CrawlError("The response exceeded the total download time limit.")
                if not chunk:
                    continue
                total_size += len(chunk)
                if total_size > max_bytes:
                    response.close()
                    raise CrawlError("The response exceeds the configured size limit.")
                chunks.append(chunk)
        except requests.RequestException as error:
            response.close()
            raise CrawlError(f"Could not read the response body: {error}") from error

        response._content = b"".join(chunks)
        response._content_consumed = True
        response.close()

    def _wait_before_request(self) -> None:
        if self._last_request_at is not None and self.delay_seconds:
            elapsed = time.monotonic() - self._last_request_at
            remaining_delay = self.delay_seconds - elapsed
            if remaining_delay > 0:
                time.sleep(remaining_delay)
        self._last_request_at = time.monotonic()

"""BM25 tokenization with stopwords, stemming, and synonym expansion."""

import hashlib
import logging
import re

logger = logging.getLogger(__name__)

# Expanded stopwords list for BM25 tokenization
# Removes common English words that don't contribute to search relevance
STOP_WORDS = {
    "a", "about", "above", "after", "again", "against", "all", "am", "an", "and",
    "any", "are", "aren't", "as", "at", "be", "because", "been", "before", "being",
    "below", "between", "both", "but", "by", "can", "can't", "cannot", "could",
    "couldn't", "did", "didn't", "do", "does", "doesn't", "doing", "don't", "down",
    "during", "each", "few", "for", "from", "further", "had", "hadn't", "has",
    "hasn't", "have", "haven't", "having", "he", "he'd", "he'll", "he's", "her",
    "here", "here's", "hers", "herself", "him", "himself", "his", "how", "how's",
    "i", "i'd", "i'll", "i'm", "i've", "if", "in", "into", "is", "isn't", "it",
    "it's", "its", "itself", "let's", "me", "more", "most", "mustn't", "my",
    "myself", "no", "nor", "not", "of", "off", "on", "once", "only", "or", "other",
    "ought", "our", "ours", "ourselves", "out", "over", "own", "same", "shan't",
    "she", "she'd", "she'll", "she's", "should", "shouldn't", "so", "some", "such",
    "than", "that", "that's", "the", "their", "theirs", "them", "themselves", "then",
    "there", "there's", "these", "they", "they'd", "they'll", "they're", "they've",
    "this", "those", "through", "to", "too", "under", "until", "up", "very", "was",
    "wasn't", "we", "we'd", "we'll", "we're", "we've", "were", "weren't", "what",
    "what's", "when", "when's", "where", "where's", "which", "while", "who", "who's",
    "whom", "why", "why's", "with", "won't", "would", "wouldn't", "you", "you'd",
    "you'll", "you're", "you've", "your", "yours", "yourself", "yourselves",
}

# Increment TOKENIZER_VERSION when tokenization logic changes to invalidate caches
TOKENIZER_VERSION = 2  # v2: Added EnhancedTokenizer with stemming support


def advanced_tokenize(text: str) -> list[str]:
    """
    Advanced tokenizer for BM25 that removes stopwords and noise.

    - Extracts words with 2+ characters
    - Lowercases everything
    - Filters out common English stopwords

    Args:
        text: Text to tokenize

    Returns:
        List of meaningful tokens
    """
    # Extract words with 2+ alphanumeric characters
    words = re.findall(r'\b\w{2,}\b', text.lower())
    # Filter stopwords
    return [w for w in words if w not in STOP_WORDS]


class EnhancedTokenizer:
    """
    Enhanced tokenizer with configurable stemming and synonym expansion.

    Provides better recall for BM25 search by:
    - Stemming words to their root form (e.g., "countries" -> "countri")
    - Expanding queries with synonyms (e.g., "AML" -> ["AML", "anti-money laundering"])
    """

    VERSION = 2  # Increment when tokenization logic changes

    def __init__(
        self,
        stemmer: str = None,
        synonyms: dict = None,
        language: str = "english",
    ):
        """
        Initialize the enhanced tokenizer.

        Args:
            stemmer: Stemmer type - "porter", "snowball", or None (no stemming)
            synonyms: Dict mapping terms to their synonyms/expansions
                      e.g., {"cdd": ["customer due diligence", "KYC"]}
            language: Language for stemming (default: "english")
        """
        self.stemmer_type = stemmer
        self.synonyms = synonyms or {}
        self.language = language
        self._stemmer = None

        # Initialize stemmer if requested
        if stemmer:
            self._init_stemmer(stemmer, language)

        # Build reverse synonym lookup for query expansion
        # Maps each synonym to its canonical form
        self._synonym_expansion = {}
        for canonical, syns in self.synonyms.items():
            canonical_lower = canonical.lower()
            self._synonym_expansion[canonical_lower] = [canonical_lower]
            for syn in syns:
                syn_lower = syn.lower()
                self._synonym_expansion[canonical_lower].append(syn_lower)
                # Also map synonym back to canonical for bidirectional lookup
                if syn_lower not in self._synonym_expansion:
                    self._synonym_expansion[syn_lower] = [syn_lower, canonical_lower]

    def _init_stemmer(self, stemmer_type: str, language: str) -> None:
        """Initialize the NLTK stemmer."""
        try:
            if stemmer_type == "porter":
                from nltk.stem import PorterStemmer
                self._stemmer = PorterStemmer()
            elif stemmer_type == "snowball":
                from nltk.stem import SnowballStemmer
                self._stemmer = SnowballStemmer(language)
            else:
                logger.warning(f"Unknown stemmer type: {stemmer_type}, disabling stemming")
                self._stemmer = None
        except ImportError:
            logger.warning("NLTK not available, stemming disabled")
            self._stemmer = None
        except Exception as e:
            logger.warning(f"Failed to initialize {stemmer_type} stemmer: {e}")
            self._stemmer = None

    def _stem(self, word: str) -> str:
        """Stem a word if stemmer is available."""
        if self._stemmer:
            return self._stemmer.stem(word)
        return word

    def tokenize(self, text: str) -> list[str]:
        """
        Tokenize text for corpus indexing.

        Applies:
        - Lowercasing
        - Stopword removal
        - Stemming (if enabled)

        Args:
            text: Text to tokenize

        Returns:
            List of tokens for indexing
        """
        # Extract words with 2+ alphanumeric characters
        words = re.findall(r'\b\w{2,}\b', text.lower())

        # Filter stopwords
        words = [w for w in words if w not in STOP_WORDS]

        # Apply stemming if enabled
        if self._stemmer:
            words = [self._stem(w) for w in words]

        return words

    def tokenize_query(self, text: str) -> list[str]:
        """
        Tokenize query text with synonym expansion.

        Applies:
        - Lowercasing
        - Stopword removal
        - Stemming (if enabled)
        - Synonym expansion

        Args:
            text: Query text to tokenize

        Returns:
            List of tokens including expansions
        """
        # Extract words with 2+ alphanumeric characters
        words = re.findall(r'\b\w{2,}\b', text.lower())

        # Filter stopwords
        words = [w for w in words if w not in STOP_WORDS]

        # Expand synonyms and stem
        expanded = []
        for word in words:
            # Check if word has synonyms
            if word in self._synonym_expansion:
                # Add all synonyms (stemmed if applicable)
                for syn in self._synonym_expansion[word]:
                    syn_tokens = syn.split()  # Handle multi-word synonyms
                    for token in syn_tokens:
                        if token not in STOP_WORDS:
                            expanded.append(self._stem(token) if self._stemmer else token)
            else:
                # Just stem the original word
                expanded.append(self._stem(word) if self._stemmer else word)

        return expanded

    def get_config_hash(self) -> str:
        """Get a hash of the tokenizer configuration for cache invalidation."""
        config_str = f"v{self.VERSION}:{self.stemmer_type}:{self.language}:{sorted(self.synonyms.items())}"
        return hashlib.md5(config_str.encode()).hexdigest()[:8]

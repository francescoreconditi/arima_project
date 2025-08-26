"""
Sistema di gestione traduzioni centralizzato per ARIMA Forecaster.

Supporta caricamento dinamico di traduzioni da file JSON locales
per fornire interfacce multilingue in dashboard, report e API.
"""

import json
from pathlib import Path
from typing import Dict, Any, Optional
from functools import lru_cache


class TranslationManager:
    """
    Gestore centralizzato delle traduzioni multi-lingua.
    
    Carica automaticamente traduzioni da file JSON nella directory locales
    e fornisce accesso thread-safe con caching per performance ottimali.
    
    Usage:
        >>> translator = TranslationManager()
        >>> text = translator.get('title', language='en')
        >>> translations = translator.get_all('it')
    """
    
    def __init__(self, locales_dir: Optional[Path] = None):
        """
        Inizializza il gestore traduzioni.
        
        Args:
            locales_dir: Directory contenente i file JSON delle traduzioni.
                        Se None, usa la directory assets/locales del progetto.
        """
        if locales_dir is None:
            # Path relativo al modulo corrente
            current_dir = Path(__file__).parent
            self.locales_dir = current_dir.parent / "assets" / "locales"
        else:
            self.locales_dir = Path(locales_dir)
            
        # Mapping codici lingua per compatibilità
        self.language_mapping = {
            'Italiano': 'it',
            'English': 'en', 
            'Español': 'es',
            'Français': 'fr',
            '中文': 'zh',
            # Aggiungi alias comuni
            'italian': 'it',
            'english': 'en',
            'spanish': 'es',
            'french': 'fr',
            'chinese': 'zh',
            'zh-CN': 'zh',
            'zh-TW': 'zh'
        }
        
        self.supported_languages = ['it', 'en', 'es', 'fr', 'zh']
        self.default_language = 'it'
        
    @lru_cache(maxsize=10)
    def _load_translations(self, language_code: str) -> Dict[str, Any]:
        """
        Carica traduzioni da file JSON con caching per performance.
        
        Args:
            language_code: Codice lingua (es. 'it', 'en', 'zh')
            
        Returns:
            Dizionario delle traduzioni caricate
            
        Raises:
            FileNotFoundError: Se il file della lingua non esiste
            json.JSONDecodeError: Se il file JSON è malformato
        """
        file_path = self.locales_dir / f"{language_code}.json"
        
        if not file_path.exists():
            # Fallback alla lingua di default
            if language_code != self.default_language:
                return self._load_translations(self.default_language)
            else:
                raise FileNotFoundError(f"File traduzioni non trovato: {file_path}")
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except json.JSONDecodeError as e:
            raise json.JSONDecodeError(f"Errore parsing JSON {file_path}: {e}", e.doc, e.pos)
    
    def normalize_language(self, language: str) -> str:
        """
        Normalizza il codice lingua usando il mapping interno.
        
        Args:
            language: Lingua in formato user-friendly (es. 'Italiano', 'English')
            
        Returns:
            Codice lingua normalizzato (es. 'it', 'en')
        """
        normalized = self.language_mapping.get(language, language.lower())
        
        # Verifica se supportata
        if normalized not in self.supported_languages:
            return self.default_language
            
        return normalized
    
    def get(self, key: str, language: str = None, fallback: str = None) -> str:
        """
        Ottieni una traduzione specifica per chiave e lingua.
        
        Args:
            key: Chiave della traduzione (es. 'title', 'warehouse_value')
            language: Lingua target, se None usa default
            fallback: Valore di fallback se la chiave non esiste
            
        Returns:
            Testo tradotto o fallback
            
        Example:
            >>> manager.get('title', 'en')
            'Inventory Management Report - Moretti S.p.A.'
        """
        if language is None:
            language = self.default_language
        
        language_code = self.normalize_language(language)
        
        try:
            translations = self._load_translations(language_code)
            return translations.get(key, fallback or key)
        except (FileNotFoundError, json.JSONDecodeError):
            # Ultimo fallback: ritorna la chiave stessa o il fallback
            return fallback or key
    
    def get_all(self, language: str = None) -> Dict[str, str]:
        """
        Ottieni tutte le traduzioni per una lingua specifica.
        
        Args:
            language: Lingua target, se None usa default
            
        Returns:
            Dizionario completo delle traduzioni
            
        Example:
            >>> translations = manager.get_all('zh')
            >>> print(translations['title'])
            '库存管理报告 - Moretti S.p.A.'
        """
        if language is None:
            language = self.default_language
            
        language_code = self.normalize_language(language)
        
        try:
            return self._load_translations(language_code)
        except (FileNotFoundError, json.JSONDecodeError):
            # Fallback: ritorna dizionario vuoto
            return {}
    
    def get_available_languages(self) -> Dict[str, str]:
        """
        Ottieni lista delle lingue disponibili con nomi user-friendly.
        
        Returns:
            Dizionario {codice_lingua: nome_display}
            
        Example:
            >>> manager.get_available_languages()
            {'it': 'Italiano', 'en': 'English', 'es': 'Español', 'fr': 'Français', 'zh': '中文'}
        """
        display_names = {
            'it': 'Italiano',
            'en': 'English', 
            'es': 'Español',
            'fr': 'Français',
            'zh': '中文'
        }
        
        available = {}
        for code in self.supported_languages:
            file_path = self.locales_dir / f"{code}.json"
            if file_path.exists():
                available[code] = display_names.get(code, code.upper())
                
        return available
    
    def format_text(self, key: str, language: str = None, **kwargs) -> str:
        """
        Ottieni traduzione formattata con parametri.
        
        Args:
            key: Chiave della traduzione
            language: Lingua target
            **kwargs: Parametri per string formatting
            
        Returns:
            Testo tradotto e formattato
            
        Example:
            >>> manager.format_text('units_remaining', 'en', count=5)
            '5 units remaining'
        """
        template = self.get(key, language)
        
        try:
            return template.format(**kwargs)
        except (KeyError, ValueError):
            # Se il formatting fallisce, ritorna il template originale
            return template


# Istanza globale singleton per accesso semplificato
_global_translator: Optional[TranslationManager] = None


def get_translator() -> TranslationManager:
    """
    Ottieni istanza globale del gestore traduzioni (singleton pattern).
    
    Returns:
        Istanza TranslationManager configurata
    """
    global _global_translator
    if _global_translator is None:
        _global_translator = TranslationManager()
    return _global_translator


def translate(key: str, language: str = None, fallback: str = None) -> str:
    """
    Funzione di utilità per traduzione rapida.
    
    Args:
        key: Chiave della traduzione
        language: Lingua target
        fallback: Valore di fallback
        
    Returns:
        Testo tradotto
        
    Example:
        >>> from arima_forecaster.utils.translations import translate as _
        >>> title = _('title', 'en')
    """
    return get_translator().get(key, language, fallback)


def get_all_translations(language: str = None) -> Dict[str, str]:
    """
    Funzione di utilità per ottenere tutte le traduzioni.
    
    Args:
        language: Lingua target
        
    Returns:
        Dizionario completo delle traduzioni
    """
    return get_translator().get_all(language)


# Alias per compatibilità con sistemi esistenti
def get_translations_dict(language: str) -> Dict[str, str]:
    """
    Compatibilità con dashboard Moretti esistente.
    
    Args:
        language: Lingua in formato user-friendly (es. 'Italiano', 'English')
        
    Returns:
        Dizionario traduzioni compatibile con TRANSLATIONS originale
    """
    return get_all_translations(language)
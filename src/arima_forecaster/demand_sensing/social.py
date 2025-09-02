"""
Analisi sentiment social media per Demand Sensing.

Analizza sentiment da Twitter, Instagram, Reddit per prevedere domanda.
"""

import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import pandas as pd
import numpy as np
from pydantic import BaseModel, Field
from enum import Enum
import re

from .demand_sensor import ExternalFactor, FactorType
from ..utils.logger import setup_logger

logger = setup_logger(__name__)


class SentimentScore(str, Enum):
    """Livelli di sentiment."""
    
    VERY_NEGATIVE = "very_negative"  # < -0.6
    NEGATIVE = "negative"  # -0.6 to -0.2
    NEUTRAL = "neutral"  # -0.2 to 0.2
    POSITIVE = "positive"  # 0.2 to 0.6
    VERY_POSITIVE = "very_positive"  # > 0.6


class SocialPost(BaseModel):
    """Post social media."""
    
    platform: str  # twitter, instagram, reddit, facebook
    text: str
    author: str
    timestamp: datetime
    likes: int = 0
    shares: int = 0
    comments: int = 0
    sentiment_score: float = 0.0  # -1 to 1
    reach: int = 0  # Stima persone raggiunte
    is_influencer: bool = False
    hashtags: List[str] = Field(default_factory=list)


class SentimentImpact(BaseModel):
    """Configurazione impatto sentiment su domanda."""
    
    # Moltiplicatori per livello sentiment
    sentiment_multipliers: Dict[str, float] = Field(
        default_factory=lambda: {
            'very_positive': 0.25,  # +25% domanda
            'positive': 0.10,  # +10%
            'neutral': 0.0,
            'negative': -0.10,  # -10%
            'very_negative': -0.25  # -25%
        }
    )
    
    # Peso per piattaforma
    platform_weights: Dict[str, float] = Field(
        default_factory=lambda: {
            'twitter': 1.0,
            'instagram': 1.2,  # Più visual, più impatto
            'reddit': 0.8,
            'facebook': 0.7,
            'tiktok': 1.5  # Alto impatto virale
        }
    )
    
    # Fattore viralità
    virality_threshold: int = Field(
        10000,
        description="Soglia interazioni per considerare virale"
    )
    virality_multiplier: float = Field(
        2.0,
        description="Moltiplicatore per post virali"
    )
    
    # Influencer impact
    influencer_multiplier: float = Field(
        1.5,
        description="Moltiplicatore per post influencer"
    )
    
    # Decay temporale
    impact_decay_days: int = Field(
        7,
        description="Giorni per dimezzare impatto"
    )


class SocialSentimentAnalyzer:
    """
    Analizzatore sentiment social per demand sensing.
    """
    
    def __init__(
        self,
        brand_keywords: List[str],
        product_keywords: List[str],
        impact_config: Optional[SentimentImpact] = None,
        languages: List[str] = None
    ):
        """
        Inizializza analizzatore sentiment.
        
        Args:
            brand_keywords: Keywords brand da monitorare
            product_keywords: Keywords prodotto
            impact_config: Configurazione impatti
            languages: Lingue da analizzare
        """
        self.brand_keywords = brand_keywords
        self.product_keywords = product_keywords
        self.all_keywords = brand_keywords + product_keywords
        self.impact_config = impact_config or SentimentImpact()
        self.languages = languages or ['it', 'en']
        
        # Pattern sentiment (semplificato)
        self.positive_words = {
            'it': ['ottimo', 'fantastico', 'eccellente', 'super', 'adoro', 
                   'perfetto', 'migliore', 'consiglio', 'felice', 'soddisfatto'],
            'en': ['great', 'awesome', 'excellent', 'love', 'perfect', 
                   'best', 'recommend', 'happy', 'satisfied', 'amazing']
        }
        
        self.negative_words = {
            'it': ['pessimo', 'terribile', 'orribile', 'deluso', 'sconsiglio',
                   'problemi', 'difetto', 'rotto', 'male', 'peggiore'],
            'en': ['terrible', 'horrible', 'awful', 'disappointed', 'worst',
                   'problem', 'broken', 'bad', 'hate', 'avoid']
        }
        
        # Emoji sentiment
        self.positive_emojis = ['😍', '❤️', '👍', '🔥', '💯', '⭐', '🎉', '😊']
        self.negative_emojis = ['😡', '😢', '👎', '💔', '😤', '🤮', '😠', '😞']
        
        logger.info(f"SocialSentimentAnalyzer inizializzato per {len(self.all_keywords)} keywords")
    
    def fetch_social_posts(
        self,
        platforms: List[str] = None,
        days_back: int = 7,
        use_demo_data: bool = True
    ) -> List[SocialPost]:
        """
        Recupera post social recenti.
        
        Args:
            platforms: Piattaforme da analizzare
            days_back: Giorni di storia
            use_demo_data: Usa dati demo
            
        Returns:
            Lista di post social
        """
        platforms = platforms or ['twitter', 'instagram', 'reddit']
        
        # In produzione qui ci sarebbero chiamate API a:
        # - Twitter API v2
        # - Instagram Basic Display API
        # - Reddit API
        # - Facebook Graph API
        
        if use_demo_data:
            return self._generate_demo_posts(platforms, days_back)
        
        # Placeholder per API reali
        posts = []
        logger.warning("API social non configurate. Usando dati demo.")
        return self._generate_demo_posts(platforms, days_back)
    
    def _generate_demo_posts(
        self,
        platforms: List[str],
        days_back: int
    ) -> List[SocialPost]:
        """
        Genera post social demo realistici.
        
        Args:
            platforms: Piattaforme
            days_back: Giorni storia
            
        Returns:
            Lista post demo
        """
        posts = []
        np.random.seed(42)
        
        # Template post per sentiment
        templates = {
            'very_positive': [
                "Incredibile {product}! Il migliore che abbia mai provato! 😍",
                "Sono innamorato del nuovo {product}! Consigliatissimo! ⭐⭐⭐⭐⭐",
                "WOW! {brand} ha superato ogni aspettativa con {product}! 🔥💯"
            ],
            'positive': [
                "Molto soddisfatto del {product}, buon rapporto qualità prezzo 👍",
                "{brand} non delude mai, ottimo {product}",
                "Consiglio {product}, funziona bene 😊"
            ],
            'neutral': [
                "Ho provato il {product} di {brand}, nella media",
                "{product} fa il suo dovere, niente di speciale",
                "Per ora {product} sembra ok, vedremo nel tempo"
            ],
            'negative': [
                "Deluso dal {product}, mi aspettavo di più da {brand} 😞",
                "Problemi con {product} dopo pochi giorni 👎",
                "Non vale il prezzo, {product} ha diversi difetti"
            ],
            'very_negative': [
                "PESSIMO! {product} si è rotto subito! Mai più {brand}! 😡",
                "Totalmente deluso, {product} è terribile! Sconsiglio! 🤮",
                "{brand} è peggiorata, {product} è un disastro totale! 💔"
            ]
        }
        
        # Genera post per ogni giorno e piattaforma
        for day in range(days_back):
            date = datetime.now() - timedelta(days=day)
            
            # Numero post giornalieri (varia per piattaforma)
            daily_posts = {
                'twitter': np.random.poisson(15),
                'instagram': np.random.poisson(8),
                'reddit': np.random.poisson(5),
                'facebook': np.random.poisson(6),
                'tiktok': np.random.poisson(10)
            }
            
            for platform in platforms:
                n_posts = daily_posts.get(platform, 5)
                
                for _ in range(n_posts):
                    # Determina sentiment (distribuzione realistica)
                    rand = np.random.random()
                    if rand < 0.1:
                        sentiment = 'very_negative'
                        score = np.random.uniform(-1.0, -0.6)
                    elif rand < 0.25:
                        sentiment = 'negative'
                        score = np.random.uniform(-0.6, -0.2)
                    elif rand < 0.60:
                        sentiment = 'neutral'
                        score = np.random.uniform(-0.2, 0.2)
                    elif rand < 0.85:
                        sentiment = 'positive'
                        score = np.random.uniform(0.2, 0.6)
                    else:
                        sentiment = 'very_positive'
                        score = np.random.uniform(0.6, 1.0)
                    
                    # Seleziona template
                    template = np.random.choice(templates[sentiment])
                    text = template.format(
                        product=np.random.choice(self.product_keywords or ['prodotto']),
                        brand=np.random.choice(self.brand_keywords or ['brand'])
                    )
                    
                    # Engagement (correlato a sentiment estremi)
                    if sentiment in ['very_positive', 'very_negative']:
                        engagement_mult = 3
                    elif sentiment in ['positive', 'negative']:
                        engagement_mult = 1.5
                    else:
                        engagement_mult = 1
                    
                    # Occasionale post virale
                    is_viral = np.random.random() < 0.02
                    if is_viral:
                        engagement_mult *= 10
                    
                    # Occasionale influencer
                    is_influencer = np.random.random() < 0.05
                    if is_influencer:
                        engagement_mult *= 5
                    
                    likes = int(np.random.gamma(2, 50) * engagement_mult)
                    shares = int(likes * np.random.uniform(0.05, 0.3))
                    comments = int(likes * np.random.uniform(0.02, 0.15))
                    
                    post = SocialPost(
                        platform=platform,
                        text=text,
                        author=f"user_{np.random.randint(1000, 9999)}",
                        timestamp=date + timedelta(
                            hours=np.random.randint(0, 24),
                            minutes=np.random.randint(0, 60)
                        ),
                        likes=likes,
                        shares=shares,
                        comments=comments,
                        sentiment_score=score,
                        reach=likes * 10 + shares * 50,  # Stima reach
                        is_influencer=is_influencer,
                        hashtags=self._generate_hashtags()
                    )
                    
                    posts.append(post)
        
        logger.info(f"Generati {len(posts)} post social demo")
        return posts
    
    def _generate_hashtags(self) -> List[str]:
        """Genera hashtag casuali."""
        base_tags = ['#' + kw.replace(' ', '') for kw in self.all_keywords]
        extra_tags = ['#review', '#opinioni', '#consigli', '#shopping', '#nuovo']
        
        n_tags = np.random.randint(1, 4)
        tags = np.random.choice(base_tags + extra_tags, n_tags, replace=False)
        
        return list(tags)
    
    def analyze_sentiment(self, text: str) -> float:
        """
        Analizza sentiment di un testo.
        
        Args:
            text: Testo da analizzare
            
        Returns:
            Score sentiment (-1 a 1)
        """
        text_lower = text.lower()
        
        # Conta parole positive/negative
        positive_count = 0
        negative_count = 0
        
        for lang in self.languages:
            if lang in self.positive_words:
                positive_count += sum(1 for word in self.positive_words[lang] 
                                     if word in text_lower)
            if lang in self.negative_words:
                negative_count += sum(1 for word in self.negative_words[lang] 
                                    if word in text_lower)
        
        # Conta emoji
        positive_emoji = sum(1 for emoji in self.positive_emojis if emoji in text)
        negative_emoji = sum(1 for emoji in self.negative_emojis if emoji in text)
        
        # Calcola score
        total_signals = (positive_count + negative_count + 
                        positive_emoji + negative_emoji)
        
        if total_signals == 0:
            return 0.0  # Neutro
        
        positive_total = positive_count + positive_emoji * 1.5  # Emoji peso maggiore
        negative_total = negative_count + negative_emoji * 1.5
        
        score = (positive_total - negative_total) / total_signals
        
        # Normalizza tra -1 e 1
        return max(-1.0, min(1.0, score))
    
    def calculate_social_impact(
        self,
        posts: List[SocialPost],
        forecast_horizon: int = 7
    ) -> List[ExternalFactor]:
        """
        Calcola impatto social sentiment sulla domanda.
        
        Args:
            posts: Post social da analizzare
            forecast_horizon: Giorni previsione
            
        Returns:
            Lista fattori esterni social
        """
        factors = []
        
        # Aggrega sentiment per giorno
        daily_sentiment = {}
        daily_engagement = {}
        
        for post in posts:
            date_key = post.timestamp.date()
            
            if date_key not in daily_sentiment:
                daily_sentiment[date_key] = []
                daily_engagement[date_key] = 0
            
            # Peso sentiment per engagement
            engagement = post.likes + post.shares * 2 + post.comments
            weighted_sentiment = post.sentiment_score * engagement
            
            daily_sentiment[date_key].append(weighted_sentiment)
            daily_engagement[date_key] += engagement
        
        # Calcola sentiment medio pesato per giorno
        avg_sentiments = {}
        for date, sentiments in daily_sentiment.items():
            if daily_engagement[date] > 0:
                avg_sentiments[date] = sum(sentiments) / daily_engagement[date]
            else:
                avg_sentiments[date] = 0
        
        # Genera fattori per forecast horizon
        for day in range(forecast_horizon):
            forecast_date = datetime.now() + timedelta(days=day)
            
            # Usa media mobile ultimi 7 giorni
            recent_dates = [
                forecast_date.date() - timedelta(days=i) 
                for i in range(1, 8)
            ]
            
            recent_sentiments = [
                avg_sentiments.get(d, 0) for d in recent_dates
            ]
            
            if recent_sentiments:
                avg_sentiment = np.mean(recent_sentiments)
            else:
                avg_sentiment = 0
            
            # Determina livello sentiment
            if avg_sentiment < -0.6:
                sentiment_level = SentimentScore.VERY_NEGATIVE
            elif avg_sentiment < -0.2:
                sentiment_level = SentimentScore.NEGATIVE
            elif avg_sentiment < 0.2:
                sentiment_level = SentimentScore.NEUTRAL
            elif avg_sentiment < 0.6:
                sentiment_level = SentimentScore.POSITIVE
            else:
                sentiment_level = SentimentScore.VERY_POSITIVE
            
            # Calcola impatto
            base_impact = self.impact_config.sentiment_multipliers[sentiment_level.value]
            
            # Aggiusta per viralità recente
            recent_viral = sum(1 for p in posts 
                             if p.timestamp.date() in recent_dates and
                             (p.likes + p.shares) > self.impact_config.virality_threshold)
            
            if recent_viral > 0:
                base_impact *= (1 + (recent_viral * 0.1))  # +10% per post virale
            
            # Decay temporale
            decay_factor = 0.5 ** (day / self.impact_config.impact_decay_days)
            impact = base_impact * decay_factor
            
            # Confidenza basata su volume dati
            n_recent_posts = sum(1 for p in posts 
                               if p.timestamp.date() in recent_dates)
            
            if n_recent_posts > 50:
                confidence = 0.8
            elif n_recent_posts > 20:
                confidence = 0.6
            elif n_recent_posts > 5:
                confidence = 0.4
            else:
                confidence = 0.2
            
            factor = ExternalFactor(
                name=f"Social_{forecast_date.strftime('%Y-%m-%d')}",
                type=FactorType.SOCIAL,
                value=avg_sentiment,
                impact=impact,
                confidence=confidence,
                timestamp=forecast_date,
                metadata={
                    'sentiment_level': sentiment_level.value,
                    'posts_analyzed': n_recent_posts,
                    'viral_posts': recent_viral,
                    'avg_engagement': np.mean([p.likes + p.shares 
                                              for p in posts[-50:]]) if posts else 0
                }
            )
            
            factors.append(factor)
            
            logger.debug(
                f"Social {forecast_date}: sentiment={avg_sentiment:.2f}, "
                f"impact={impact:.2%}, posts={n_recent_posts}"
            )
        
        return factors
    
    def get_trending_hashtags(
        self,
        posts: List[SocialPost],
        top_n: int = 10
    ) -> List[Tuple[str, int]]:
        """
        Identifica hashtag trending.
        
        Args:
            posts: Post da analizzare
            top_n: Numero top hashtag
            
        Returns:
            Lista (hashtag, occorrenze)
        """
        hashtag_counts = {}
        
        for post in posts:
            for tag in post.hashtags:
                hashtag_counts[tag] = hashtag_counts.get(tag, 0) + 1
        
        # Ordina per frequenza
        sorted_tags = sorted(
            hashtag_counts.items(),
            key=lambda x: x[1],
            reverse=True
        )
        
        return sorted_tags[:top_n]
    
    def identify_influencers(
        self,
        posts: List[SocialPost],
        min_engagement: int = 5000
    ) -> List[str]:
        """
        Identifica influencer rilevanti.
        
        Args:
            posts: Post da analizzare
            min_engagement: Engagement minimo
            
        Returns:
            Lista autori influencer
        """
        author_engagement = {}
        
        for post in posts:
            if post.is_influencer or (post.likes + post.shares) > min_engagement:
                engagement = post.likes + post.shares * 2
                if post.author not in author_engagement:
                    author_engagement[post.author] = 0
                author_engagement[post.author] += engagement
        
        # Filtra per engagement totale
        influencers = [
            author for author, total_eng in author_engagement.items()
            if total_eng > min_engagement * 3
        ]
        
        return influencers
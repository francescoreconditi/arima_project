"""
Test per modulo demand sensing.

Test completi per tutti i sensori di domanda e ensemble.
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, MagicMock
import json

from arima_forecaster.demand_sensing.demand_sensor import DemandSensor
from arima_forecaster.demand_sensing.weather import WeatherSensor
from arima_forecaster.demand_sensing.trends import TrendsSensor
from arima_forecaster.demand_sensing.social import SocialSensor
from arima_forecaster.demand_sensing.economic import EconomicSensor
from arima_forecaster.demand_sensing.calendar_events import CalendarEventsSensor
from arima_forecaster.demand_sensing.ensemble import EnsembleSensor


class TestDemandSensor:
    """Test per sensore domanda base."""
    
    @pytest.fixture
    def sensor(self):
        """Crea istanza sensor base."""
        return DemandSensor(name="test_sensor")
    
    def test_initialization(self, sensor):
        """Test inizializzazione sensor."""
        assert sensor.name == "test_sensor"
        assert sensor.weight == 1.0
        assert sensor.enabled is True
        assert sensor.config == {}
    
    def test_validate_data(self, sensor):
        """Test validazione dati input."""
        # Dati validi
        valid_data = pd.Series(np.random.randn(100))
        assert sensor.validate_data(valid_data) is True
        
        # Dati non validi
        invalid_data = pd.Series([])
        assert sensor.validate_data(invalid_data) is False
        
        # Dati con NaN
        nan_data = pd.Series([1, 2, np.nan, 4, 5])
        assert sensor.validate_data(nan_data) is False
    
    def test_normalize_impact(self, sensor):
        """Test normalizzazione impatto."""
        impacts = np.array([1, 2, 3, 4, 5])
        normalized = sensor.normalize_impact(impacts)
        
        assert normalized.min() >= -1
        assert normalized.max() <= 1
        assert len(normalized) == len(impacts)


class TestWeatherSensor:
    """Test per sensore meteo."""
    
    @pytest.fixture
    def sensor(self):
        """Crea istanza weather sensor."""
        return WeatherSensor(
            api_key="test_key",
            location="Milano",
            factors=['temperature', 'precipitation']
        )
    
    @patch('requests.get')
    def test_fetch_weather_data(self, mock_get, sensor):
        """Test recupero dati meteo."""
        # Mock response API
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            'current': {
                'temp': 25.5,
                'humidity': 60,
                'wind_speed': 10
            },
            'daily': [
                {'temp': {'day': 26}, 'rain': 0},
                {'temp': {'day': 24}, 'rain': 2.5}
            ]
        }
        mock_get.return_value = mock_response
        
        data = sensor.fetch_weather_data(
            start_date=datetime.now(),
            end_date=datetime.now() + timedelta(days=1)
        )
        
        assert 'temperature' in data
        assert 'precipitation' in data
        assert len(data['temperature']) > 0
    
    def test_calculate_weather_impact(self, sensor):
        """Test calcolo impatto meteo."""
        weather_data = {
            'temperature': [15, 20, 25, 30, 35],  # Temperature crescenti
            'precipitation': [0, 5, 10, 15, 20]   # Pioggia crescente
        }
        
        # Prodotto che vende di più con caldo e poco con pioggia
        product_profile = {
            'weather_sensitive': True,
            'optimal_temp': 28,
            'rain_negative': True
        }
        
        impact = sensor.calculate_impact(weather_data, product_profile)
        
        assert 'temperature_impact' in impact
        assert 'precipitation_impact' in impact
        assert 'combined_impact' in impact
        
        # Temperature vicine a 28 dovrebbero avere impatto positivo
        assert impact['temperature_impact'][3] > impact['temperature_impact'][0]
    
    def test_extreme_weather_detection(self, sensor):
        """Test rilevamento eventi meteo estremi."""
        weather_data = {
            'temperature': [10, 15, 45, 20, 25],  # 45 è estremo
            'wind_speed': [5, 10, 80, 15, 10]     # 80 è estremo
        }
        
        extremes = sensor.detect_extreme_events(weather_data)
        
        assert len(extremes) > 0
        assert any(e['type'] == 'extreme_temperature' for e in extremes)
        assert any(e['type'] == 'extreme_wind' for e in extremes)


class TestTrendsSensor:
    """Test per sensore trend di ricerca."""
    
    @pytest.fixture
    def sensor(self):
        """Crea istanza trends sensor."""
        return TrendsSensor(
            keywords=['smartphone', 'iphone', 'samsung'],
            geo='IT'
        )
    
    @patch('pytrends.request.TrendReq.interest_over_time')
    def test_fetch_trends_data(self, mock_trends, sensor):
        """Test recupero dati Google Trends."""
        # Mock dati trends
        dates = pd.date_range(start='2024-01-01', periods=30, freq='D')
        mock_data = pd.DataFrame({
            'smartphone': np.random.randint(50, 100, 30),
            'iphone': np.random.randint(40, 90, 30),
            'samsung': np.random.randint(30, 80, 30),
            'isPartial': [False] * 30
        }, index=dates)
        
        mock_trends.return_value = mock_data
        
        trends = sensor.fetch_trends_data(
            start_date=datetime(2024, 1, 1),
            end_date=datetime(2024, 1, 30)
        )
        
        assert 'smartphone' in trends.columns
        assert len(trends) == 30
        assert trends['smartphone'].max() <= 100
    
    def test_calculate_trend_momentum(self, sensor):
        """Test calcolo momentum trend."""
        # Serie con trend crescente
        trend_data = pd.Series(
            [10, 15, 20, 25, 30, 35, 40, 45, 50],
            index=pd.date_range('2024-01-01', periods=9, freq='D')
        )
        
        momentum = sensor.calculate_momentum(trend_data, window=3)
        
        assert len(momentum) == len(trend_data)
        assert momentum.iloc[-1] > momentum.iloc[0]  # Momentum crescente
    
    def test_detect_trend_breakout(self, sensor):
        """Test rilevamento breakout trend."""
        # Serie con breakout
        normal = np.random.normal(50, 5, 20)
        breakout = np.random.normal(80, 5, 10)  # Salto improvviso
        data = pd.Series(
            np.concatenate([normal, breakout]),
            index=pd.date_range('2024-01-01', periods=30, freq='D')
        )
        
        breakouts = sensor.detect_breakouts(data, threshold=2.0)
        
        assert len(breakouts) > 0
        assert breakouts[0]['index'] >= 18  # Breakout dopo i primi 20 giorni


class TestSocialSensor:
    """Test per sensore social media."""
    
    @pytest.fixture
    def sensor(self):
        """Crea istanza social sensor."""
        return SocialSensor(
            platforms=['twitter', 'instagram'],
            hashtags=['#product', '#brand']
        )
    
    def test_calculate_sentiment_impact(self, sensor):
        """Test calcolo impatto sentiment."""
        sentiments = {
            'positive': 0.6,
            'neutral': 0.3,
            'negative': 0.1
        }
        
        impact = sensor.calculate_sentiment_impact(sentiments)
        
        assert -1 <= impact <= 1
        assert impact > 0  # Sentiment prevalentemente positivo
    
    def test_viral_detection(self, sensor):
        """Test rilevamento contenuti virali."""
        # Simula metriche post
        posts = [
            {'likes': 100, 'shares': 10, 'comments': 5},
            {'likes': 10000, 'shares': 5000, 'comments': 2000},  # Virale
            {'likes': 200, 'shares': 20, 'comments': 10}
        ]
        
        viral = sensor.detect_viral_content(posts, threshold=1000)
        
        assert len(viral) == 1
        assert viral[0]['likes'] == 10000
    
    def test_engagement_rate_calculation(self, sensor):
        """Test calcolo tasso engagement."""
        metrics = {
            'followers': 10000,
            'likes': 500,
            'comments': 50,
            'shares': 100
        }
        
        rate = sensor.calculate_engagement_rate(metrics)
        
        assert rate == 6.5  # (500 + 50 + 100) / 10000 * 100


class TestEconomicSensor:
    """Test per sensore indicatori economici."""
    
    @pytest.fixture
    def sensor(self):
        """Crea istanza economic sensor."""
        return EconomicSensor(
            indicators=['GDP', 'CPI', 'unemployment']
        )
    
    def test_fetch_economic_data(self, sensor):
        """Test recupero dati economici."""
        # Mock dati economici
        with patch.object(sensor, '_fetch_from_source') as mock_fetch:
            mock_fetch.return_value = pd.DataFrame({
                'GDP': [2.5, 2.6, 2.4],
                'CPI': [101, 102, 103],
                'unemployment': [5.0, 4.9, 5.1]
            })
            
            data = sensor.fetch_economic_indicators()
            
            assert 'GDP' in data.columns
            assert 'CPI' in data.columns
            assert len(data) == 3
    
    def test_calculate_economic_impact(self, sensor):
        """Test calcolo impatto economico."""
        indicators = pd.DataFrame({
            'GDP': [2.0, 2.5, 3.0],  # Crescita
            'unemployment': [6.0, 5.5, 5.0]  # Calo disoccupazione
        })
        
        impact = sensor.calculate_impact(indicators)
        
        assert 'GDP_impact' in impact
        assert 'unemployment_impact' in impact
        assert impact['GDP_impact'] > 0  # Crescita positiva
        assert impact['unemployment_impact'] > 0  # Calo disoccupazione positivo
    
    def test_recession_detection(self, sensor):
        """Test rilevamento recessione."""
        # GDP con due trimestri negativi consecutivi
        gdp_data = pd.Series([2.5, 2.0, -0.5, -1.0, 0.5])
        
        is_recession = sensor.detect_recession(gdp_data)
        
        assert is_recession is True


class TestCalendarEventsSensor:
    """Test per sensore eventi calendario."""
    
    @pytest.fixture
    def sensor(self):
        """Crea istanza calendar sensor."""
        return CalendarEventsSensor(
            country='IT',
            include_holidays=True,
            include_events=True
        )
    
    def test_fetch_holidays(self, sensor):
        """Test recupero festività."""
        holidays = sensor.fetch_holidays(year=2024)
        
        assert len(holidays) > 0
        assert 'Natale' in [h['name'] for h in holidays]
        assert 'Pasqua' in [h['name'] for h in holidays]
    
    def test_calculate_holiday_impact(self, sensor):
        """Test calcolo impatto festività."""
        # Definisci impatto per tipo di festività
        holiday_impacts = {
            'Natale': 2.5,  # +150% vendite
            'Pasqua': 1.5,  # +50% vendite
            'Ferragosto': 0.7  # -30% vendite
        }
        
        dates = pd.date_range('2024-12-20', '2024-12-27', freq='D')
        impact = sensor.calculate_holiday_impact(dates, holiday_impacts)
        
        # Il 25 dicembre dovrebbe avere impatto massimo
        christmas_idx = 5  # 25 dicembre
        assert impact[christmas_idx] == 2.5
    
    def test_promotional_events_detection(self, sensor):
        """Test rilevamento eventi promozionali."""
        events = [
            {'date': '2024-11-29', 'type': 'Black Friday', 'impact': 3.0},
            {'date': '2024-12-26', 'type': 'Boxing Day', 'impact': 2.0}
        ]
        
        sensor.set_promotional_events(events)
        
        date_range = pd.date_range('2024-11-28', '2024-11-30', freq='D')
        impacts = sensor.get_promotional_impact(date_range)
        
        assert impacts[1] == 3.0  # Black Friday


class TestEnsembleSensor:
    """Test per ensemble di sensori."""
    
    @pytest.fixture
    def ensemble(self):
        """Crea ensemble di sensori."""
        sensors = [
            WeatherSensor(api_key="test", weight=0.3),
            TrendsSensor(keywords=['test'], weight=0.3),
            EconomicSensor(indicators=['GDP'], weight=0.4)
        ]
        return EnsembleSensor(sensors=sensors)
    
    def test_ensemble_combination(self, ensemble):
        """Test combinazione impatti sensori."""
        # Mock impatti individuali
        impacts = {
            'weather': np.array([0.1, 0.2, 0.3]),
            'trends': np.array([0.2, 0.3, 0.4]),
            'economic': np.array([0.15, 0.25, 0.35])
        }
        
        with patch.object(ensemble.sensors[0], 'get_impact', return_value=impacts['weather']):
            with patch.object(ensemble.sensors[1], 'get_impact', return_value=impacts['trends']):
                with patch.object(ensemble.sensors[2], 'get_impact', return_value=impacts['economic']):
                    
                    combined = ensemble.get_combined_impact()
                    
                    # Verifica weighted average
                    expected = (impacts['weather'] * 0.3 + 
                              impacts['trends'] * 0.3 + 
                              impacts['economic'] * 0.4)
                    
                    np.testing.assert_array_almost_equal(combined, expected)
    
    def test_adaptive_weights(self, ensemble):
        """Test aggiornamento pesi adattivi."""
        # Simula performance sensori
        performances = {
            ensemble.sensors[0]: 0.8,  # Weather buona performance
            ensemble.sensors[1]: 0.6,  # Trends media performance
            ensemble.sensors[2]: 0.4   # Economic bassa performance
        }
        
        ensemble.update_weights(performances)
        
        # Weather dovrebbe avere peso maggiore
        assert ensemble.sensors[0].weight > ensemble.sensors[2].weight
        
        # La somma dei pesi dovrebbe essere 1
        total_weight = sum(s.weight for s in ensemble.sensors)
        assert abs(total_weight - 1.0) < 0.01
    
    def test_sensor_selection(self, ensemble):
        """Test selezione automatica sensori."""
        # Simula correlazioni con target
        correlations = {
            'weather': 0.7,
            'trends': 0.3,
            'economic': 0.5
        }
        
        selected = ensemble.select_best_sensors(
            correlations, 
            min_correlation=0.4
        )
        
        assert len(selected) == 2  # Solo weather e economic
        assert 'trends' not in [s.name for s in selected]


class TestIntegrationDemandSensing:
    """Test integrazione completa demand sensing."""
    
    def test_full_demand_sensing_pipeline(self):
        """Test pipeline completo demand sensing."""
        # Crea ensemble
        sensors = [
            WeatherSensor(api_key="test", weight=0.25),
            TrendsSensor(keywords=['product'], weight=0.25),
            CalendarEventsSensor(country='IT', weight=0.25),
            EconomicSensor(indicators=['GDP'], weight=0.25)
        ]
        ensemble = EnsembleSensor(sensors)
        
        # Dati storici vendite
        dates = pd.date_range('2024-01-01', '2024-03-31', freq='D')
        sales = pd.Series(
            np.random.poisson(100, len(dates)) * (1 + np.random.normal(0, 0.1, len(dates))),
            index=dates
        )
        
        # Mock sensori
        with patch.object(sensors[0], 'get_impact') as mock_weather:
            with patch.object(sensors[1], 'get_impact') as mock_trends:
                with patch.object(sensors[2], 'get_impact') as mock_calendar:
                    with patch.object(sensors[3], 'get_impact') as mock_economic:
                        
                        # Impatti simulati
                        mock_weather.return_value = np.random.normal(0, 0.1, len(dates))
                        mock_trends.return_value = np.random.normal(0.05, 0.05, len(dates))
                        mock_calendar.return_value = np.zeros(len(dates))
                        mock_calendar.return_value[85] = 0.5  # Pasqua
                        mock_economic.return_value = np.random.normal(0.02, 0.02, len(dates))
                        
                        # Ottieni impatto combinato
                        combined_impact = ensemble.get_combined_impact()
                        
                        # Applica a forecast
                        adjusted_forecast = sales * (1 + combined_impact)
                        
                        assert len(adjusted_forecast) == len(sales)
                        assert adjusted_forecast.iloc[85] > sales.iloc[85]  # Boost Pasqua
    
    def test_realtime_demand_sensing(self):
        """Test demand sensing real-time."""
        sensor = TrendsSensor(keywords=['product'])
        
        # Simula stream dati real-time
        realtime_data = []
        for hour in range(24):
            data_point = {
                'timestamp': datetime.now() + timedelta(hours=hour),
                'search_volume': np.random.randint(50, 150),
                'sentiment': np.random.choice(['positive', 'neutral', 'negative'])
            }
            realtime_data.append(data_point)
        
        # Processa stream
        impacts = []
        for data in realtime_data:
            with patch.object(sensor, 'process_realtime') as mock_process:
                mock_process.return_value = 0.1 if data['sentiment'] == 'positive' else -0.1
                impact = sensor.process_realtime(data)
                impacts.append(impact)
        
        assert len(impacts) == 24
        assert all(-1 <= i <= 1 for i in impacts)
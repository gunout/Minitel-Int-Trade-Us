#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from flask import Flask, render_template, jsonify, request, send_from_directory
from flask_cors import CORS
from datetime import datetime, timedelta
import yfinance as yf
import pandas as pd
import numpy as np
import json
import os
import pytz
import re
import logging
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
import warnings
warnings.filterwarnings('ignore')

# Configuration du logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__,
            static_folder='static',
            template_folder='templates')
CORS(app)

# Fuseau horaire US (New York)
US_TIMEZONE = pytz.timezone('America/New_York')

# Cache avec validation
cache = {}
CACHE_DURATION = 30

# ============================================================
# WATCHLIST US COMPLÈTE AVEC SPACEX
# ============================================================

DEFAULT_WATCHLIST = [
    # Indices US
    '^GSPC',   # S&P 500
    '^IXIC',   # NASDAQ Composite
    '^DJI',    # Dow Jones Industrial Average
    '^RUT',    # Russell 2000
    '^VIX',    # Volatility Index
    
    # SpaceX - Nouvel IPO historique
    'SPCX',    # SpaceX (Nasdaq Global Select Market)
    
    # Magnificent 7
    'AAPL',    # Apple
    'MSFT',    # Microsoft
    'GOOGL',   # Alphabet (Google)
    'AMZN',    # Amazon
    'NVDA',    # NVIDIA
    'META',    # Meta (Facebook)
    'TSLA',    # Tesla
    
    # Grandes valeurs technologiques
    'AMD',     # AMD
    'INTC',    # Intel
    'CRM',     # Salesforce
    'ORCL',    # Oracle
    'IBM',     # IBM
    'CSCO',    # Cisco
    
    # Banques & Finance
    'JPM',     # JPMorgan Chase
    'BAC',     # Bank of America
    'GS',      # Goldman Sachs
    'V',       # Visa
    'MA',      # Mastercard
    'AXP',     # American Express
    
    # Santé
    'JNJ',     # Johnson & Johnson
    'PFE',     # Pfizer
    'UNH',     # UnitedHealth
    'ABBV',    # AbbVie
    'MRK',     # Merck
    
    # Consommation
    'WMT',     # Walmart
    'PG',      # Procter & Gamble
    'KO',      # Coca-Cola
    'PEP',     # PepsiCo
    'MCD',     # McDonald's
    'NKE',     # Nike
    
    # Énergie
    'XOM',     # Exxon Mobil
    'CVX',     # Chevron
    'COP',     # ConocoPhillips
    
    # Autres grandes valeurs
    'BRK-B',   # Berkshire Hathaway
    'DIS',     # Disney
    'NFLX',    # Netflix
    'BA',      # Boeing
    'CAT',     # Caterpillar
    'GE',      # General Electric
    'F',       # Ford
    'GM'       # General Motors
]

# ============================================================
# FONCTIONS UTILITAIRES DE NETTOYAGE
# ============================================================

def sanitize_string(value):
    """Nettoie les chaînes pour JSON"""
    if isinstance(value, str):
        value = re.sub(r'[\x00-\x1f\x7f-\x9f]', '', value)
        value = value.replace('\\', '\\\\').replace('"', '\\"')
        return value
    return value

def sanitize_for_json(obj):
    """Nettoie récursivement un objet pour le JSON"""
    if isinstance(obj, dict):
        return {sanitize_string(k): sanitize_for_json(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [sanitize_for_json(item) for item in obj]
    elif isinstance(obj, str):
        return sanitize_string(obj)
    elif isinstance(obj, (int, float)):
        if pd.isna(obj) or not np.isfinite(obj):
            return None
        return obj
    elif isinstance(obj, (np.integer, np.floating)):
        if pd.isna(obj) or not np.isfinite(obj):
            return None
        return float(obj)
    elif isinstance(obj, pd.Series):
        return sanitize_for_json(obj.tolist())
    elif isinstance(obj, pd.DataFrame):
        return sanitize_for_json(obj.to_dict('records'))
    else:
        return obj

def is_valid_json(data):
    """Vérifie si les données sont sérialisables en JSON"""
    try:
        json.dumps(data)
        return True
    except (TypeError, ValueError) as e:
        logger.error(f"Erreur validation JSON: {e}")
        return False

# ============================================================
# FONCTIONS DE CACHE CORRIGÉES
# ============================================================

def get_cached_data(key, ttl=CACHE_DURATION):
    """Récupère les données du cache avec validation"""
    if key in cache:
        data, timestamp = cache[key]
        if (datetime.now() - timestamp).seconds < ttl:
            if is_valid_json(data):
                return data
            else:
                del cache[key]
                logger.warning(f"Cache corrompu pour {key}, suppression")
    return None

def set_cached_data(key, data):
    """Met en cache après validation"""
    if is_valid_json(data):
        cache[key] = (data, datetime.now())
        logger.info(f"Données mises en cache pour {key}")
    else:
        logger.error(f"Impossible de mettre en cache {key}: données invalides")

# ============================================================
# FONCTIONS MÉTIER US
# ============================================================

def get_exchange(symbol):
    """Détermine la bourse du symbole US"""
    if symbol.endswith('.PA'):
        return 'Euronext Paris'
    elif symbol.endswith('.AS'):
        return 'Euronext Amsterdam'
    elif symbol.startswith('^'):
        if symbol in ['^GSPC', '^DJI', '^IXIC', '^RUT']:
            return 'US Index'
        return 'Indice'
    else:
        try:
            ticker = yf.Ticker(symbol)
            info = ticker.info
            exchange = info.get('exchange', '')
            if 'NASDAQ' in exchange:
                return 'NASDAQ'
            elif 'NYSE' in exchange:
                return 'NYSE'
            elif 'AMEX' in exchange:
                return 'AMEX'
            else:
                return 'US Listed'
        except:
            return 'US Listed'

def get_currency(symbol):
    """Détermine la devise du symbole"""
    if symbol.endswith('.PA') or symbol.endswith('.AS'):
        return 'EUR'
    else:
        return 'USD'

def safe_float(value, default=0.0):
    try:
        if pd.isna(value) or value is None:
            return default
        return float(value)
    except (TypeError, ValueError):
        return default

def safe_int(value, default=0):
    try:
        if pd.isna(value) or value is None:
            return default
        return int(value)
    except (TypeError, ValueError):
        return default

# ============================================================
# ROUTE POUR VIDER LE CACHE
# ============================================================

@app.route('/api/clear-cache')
def clear_cache():
    """Vide le cache"""
    global cache
    cache = {}
    logger.info("Cache vidé")
    return jsonify({'status': 'ok', 'message': 'Cache vidé avec succès'})

# ============================================================
# ROUTE TRADING CORRIGÉE
# ============================================================

@app.route('/api/trading/<symbol>')
def get_trading_data(symbol):
    try:
        cache_key = f"trading_{symbol}"
        cached = get_cached_data(cache_key)
        if cached:
            logger.info(f"Données retournées depuis le cache pour {symbol}")
            return jsonify(cached)

        logger.info(f"Récupération des données pour {symbol}")
        ticker = yf.Ticker(symbol)

        # Récupération sécurisée des infos
        try:
            info = ticker.info
            if 'longName' in info and info['longName']:
                info['longName'] = sanitize_string(str(info['longName']))
            if 'sector' in info and info['sector']:
                info['sector'] = sanitize_string(str(info['sector']))
            if 'industry' in info and info['industry']:
                info['industry'] = sanitize_string(str(info['industry']))
        except Exception as e:
            logger.warning(f"Erreur récupération info pour {symbol}: {e}")
            info = {}

        periods = {
            '1d': '1m',
            '5d': '5m',
            '1mo': '15m',
            '3mo': '1h',
            '6mo': '1d',
            '1y': '1d',
            '2y': '1d',
            '5y': '1wk'
        }

        result = {
            'symbol': symbol,
            'name': info.get('longName', symbol),
            'exchange': get_exchange(symbol),
            'currency': get_currency(symbol),
            'data': {}
        }

        for period, interval in periods.items():
            try:
                hist = ticker.history(period=period, interval=interval)
                if hist.empty:
                    logger.warning(f"Pas de données pour {symbol} - {period}")
                    continue

                # Conversion en heure US
                if hist.index.tz is None:
                    hist.index = hist.index.tz_localize('UTC').tz_convert(US_TIMEZONE)
                else:
                    hist.index = hist.index.tz_convert(US_TIMEZONE)

                close = hist['Close']
                high = hist['High']
                low = hist['Low']
                volume = hist['Volume']

                ma_20 = close.rolling(window=20).mean()
                ma_50 = close.rolling(window=50).mean()
                ma_200 = close.rolling(window=200).mean()

                std = close.rolling(window=20).std()
                bb_upper = ma_20 + 2 * std
                bb_lower = ma_20 - 2 * std

                delta = close.diff()
                gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
                loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
                rs = gain / loss
                rsi = 100 - (100 / (1 + rs))

                exp1 = close.ewm(span=12, adjust=False).mean()
                exp2 = close.ewm(span=26, adjust=False).mean()
                macd = exp1 - exp2
                macd_signal = macd.ewm(span=9, adjust=False).mean()

                candles = []
                for idx, row in hist.iterrows():
                    try:
                        candles.append({
                            'time': int(idx.timestamp()),
                            'open': safe_float(row['Open']),
                            'high': safe_float(row['High']),
                            'low': safe_float(row['Low']),
                            'close': safe_float(row['Close']),
                            'volume': safe_int(row['Volume'])
                        })
                    except Exception as e:
                        logger.warning(f"Erreur bougie pour {symbol} à {idx}: {e}")
                        continue

                if not candles:
                    continue

                def clean_indicator(series):
                    return [safe_float(x) if not pd.isna(x) and np.isfinite(x) else None for x in series]

                result['data'][period] = {
                    'candles': candles,
                    'indicators': {
                        'ma_20': clean_indicator(ma_20),
                        'ma_50': clean_indicator(ma_50),
                        'ma_200': clean_indicator(ma_200),
                        'bb_upper': clean_indicator(bb_upper),
                        'bb_lower': clean_indicator(bb_lower),
                        'rsi': clean_indicator(rsi),
                        'macd': clean_indicator(macd),
                        'macd_signal': clean_indicator(macd_signal)
                    },
                    'stats': {
                        'current_price': safe_float(close.iloc[-1]) if not close.empty else 0,
                        'change': safe_float(close.iloc[-1] - close.iloc[-2]) if len(close) > 1 else 0,
                        'change_percent': safe_float(((close.iloc[-1] - close.iloc[-2]) / close.iloc[-2] * 100)) if len(close) > 1 and close.iloc[-2] != 0 else 0,
                        'high': safe_float(high.max()),
                        'low': safe_float(low.min()),
                        'volume': safe_int(volume.sum()),
                        'rsi_current': safe_float(rsi.iloc[-1]) if not rsi.empty else 50,
                        'macd_current': safe_float(macd.iloc[-1]) if not macd.empty else 0,
                        'macd_signal_current': safe_float(macd_signal.iloc[-1]) if not macd_signal.empty else 0
                    }
                }

            except Exception as e:
                logger.error(f"Erreur pour {symbol} - période {period}: {e}")
                continue

        result['info'] = {
            'sector': info.get('sector', 'N/A'),
            'industry': info.get('industry', 'N/A'),
            'market_cap': safe_int(info.get('marketCap', 0)),
            'pe_ratio': safe_float(info.get('trailingPE', None)),
            'dividend_yield': safe_float(info.get('dividendYield', 0)),
            'beta': safe_float(info.get('beta', None)),
        }

        result = sanitize_for_json(result)
        
        if is_valid_json(result):
            set_cached_data(cache_key, result)
            logger.info(f"Données enregistrées pour {symbol}")
            return jsonify(result)
        else:
            logger.error(f"Données invalides pour {symbol}")
            return jsonify({'error': 'Données invalides', 'symbol': symbol}), 500

    except Exception as e:
        logger.error(f"Erreur générale pour {symbol}: {e}")
        return jsonify({'error': str(e), 'symbol': symbol}), 500

# ============================================================
# ROUTE INSIGHTS CORRIGÉE
# ============================================================

@app.route('/api/insights-advanced/<symbol>')
def get_advanced_insights(symbol):
    try:
        cache_key = f"insights_{symbol}"
        cached = get_cached_data(cache_key)
        if cached:
            return jsonify(cached)

        ticker = yf.Ticker(symbol)
        hist = ticker.history(period='3mo')

        if hist.empty or len(hist) < 50:
            return jsonify({'error': 'Pas assez de données'})

        close = hist['Close'].values
        high = hist['High'].values
        low = hist['Low'].values

        returns = np.diff(close) / close[:-1]
        volatility = safe_float(np.std(returns) * np.sqrt(252) * 100)

        pivot_points = []
        for i in range(2, len(close) - 2):
            try:
                if (high[i-2] < high[i] and high[i-1] < high[i] and
                    high[i+1] < high[i] and high[i+2] < high[i]):
                    pivot_points.append(('resistance', safe_float(high[i])))
                if (low[i-2] > low[i] and low[i-1] > low[i] and
                    low[i+1] > low[i] and low[i+2] > low[i]):
                    pivot_points.append(('support', safe_float(low[i])))
            except:
                continue

        supports = sorted([p[1] for p in pivot_points if p[0] == 'support'], reverse=True)[:3]
        resistances = sorted([p[1] for p in pivot_points if p[0] == 'resistance'], reverse=True)[:3]

        momentum = safe_float((close[-1] - close[-20]) / close[-20] * 100) if len(close) >= 20 else 0

        try:
            x = np.arange(len(close)).reshape(-1, 1)
            y = close.reshape(-1, 1)
            model = make_pipeline(PolynomialFeatures(degree=3), LinearRegression())
            model.fit(x, y)
            future_days = np.arange(len(close), len(close) + 5).reshape(-1, 1)
            predictions = model.predict(future_days).flatten()
            predictions = [safe_float(p) for p in predictions]
        except Exception as e:
            logger.warning(f"Erreur prédiction pour {symbol}: {e}")
            predictions = [safe_float(close[-1])] * 5

        signals = []
        rsi = safe_float(100 - (100 / (1 + np.mean(returns[:14]) / max(np.mean(np.abs(returns[:14])), 0.0001))))

        if rsi > 70:
            signals.append({'type': 'sell', 'indicator': 'RSI', 'value': f'{rsi:.1f}',
                           'message': 'Zone de surachat (>70)'})
        elif rsi < 30:
            signals.append({'type': 'buy', 'indicator': 'RSI', 'value': f'{rsi:.1f}',
                           'message': 'Zone de survente (<30)'})

        macd = safe_float(np.mean(returns[-12:]) - np.mean(returns[-26:]))
        macd_signal = safe_float(np.mean(returns[-9:]))
        if macd > macd_signal:
            signals.append({'type': 'buy', 'indicator': 'MACD', 'value': f'{macd:.4f}',
                           'message': 'MACD au-dessus du signal'})
        elif macd < macd_signal:
            signals.append({'type': 'sell', 'indicator': 'MACD', 'value': f'{macd:.4f}',
                           'message': 'MACD en-dessous du signal'})

        current_price = safe_float(close[-1])
        if supports and abs(current_price - supports[0]) / current_price < 0.01:
            signals.append({'type': 'buy', 'indicator': 'Support', 'value': f'{supports[0]:.2f}',
                           'message': f'Prix proche du support'})
        if resistances and abs(current_price - resistances[0]) / current_price < 0.01:
            signals.append({'type': 'sell', 'indicator': 'Résistance', 'value': f'{resistances[0]:.2f}',
                           'message': f'Prix proche de la résistance'})

        buy_signals = sum(1 for s in signals if s['type'] == 'buy')
        sell_signals = sum(1 for s in signals if s['type'] == 'sell')

        if buy_signals > sell_signals:
            recommendation = 'ACHAT'
            confidence = min(90, 60 + buy_signals * 10)
        elif sell_signals > buy_signals:
            recommendation = 'VENTE'
            confidence = min(90, 60 + sell_signals * 10)
        else:
            recommendation = 'NEUTRE'
            confidence = 50

        stop_loss = safe_float(current_price * 0.975)
        take_profit = safe_float(current_price * 1.05)

        result = {
            'current_price': current_price,
            'volatility': volatility,
            'momentum': momentum,
            'supports': [safe_float(s) for s in supports],
            'resistances': [safe_float(r) for r in resistances],
            'predictions': predictions,
            'signals': signals,
            'recommendation': recommendation,
            'confidence': confidence,
            'stop_loss': stop_loss,
            'take_profit': take_profit,
            'rsi': rsi,
            'macd': macd
        }

        result = sanitize_for_json(result)
        set_cached_data(cache_key, result)
        return jsonify(result)

    except Exception as e:
        logger.error(f"Erreur insights pour {symbol}: {e}")
        return jsonify({'error': str(e)}), 500

# ============================================================
# ROUTE WATCHLIST CORRIGÉE
# ============================================================

@app.route('/api/watchlist')
def get_watchlist():
    try:
        watchlist = request.args.get('symbols', ','.join(DEFAULT_WATCHLIST)).split(',')
        results = []

        for symbol in watchlist:
            symbol = symbol.strip()
            if not symbol:
                continue

            cache_key = f"watchlist_{symbol}"
            cached = get_cached_data(cache_key)

            if cached:
                results.append(cached)
                continue

            try:
                ticker = yf.Ticker(symbol)
                
                try:
                    info = ticker.info
                except Exception as e:
                    logger.warning(f"Erreur info pour {symbol}: {e}")
                    info = {}

                try:
                    hist = ticker.history(period='1d')
                except Exception as e:
                    logger.warning(f"Erreur historique pour {symbol}: {e}")
                    hist = pd.DataFrame()

                current_price = safe_float(info.get('regularMarketPrice', 0))
                if current_price == 0 and not hist.empty:
                    current_price = safe_float(hist['Close'].iloc[-1])

                prev_close = safe_float(info.get('regularMarketPreviousClose', 0))
                if prev_close == 0 and len(hist) > 1:
                    prev_close = safe_float(hist['Close'].iloc[-2])

                change = current_price - prev_close if prev_close else 0
                change_percent = (change / prev_close * 100) if prev_close else 0

                data = {
                    'symbol': symbol,
                    'name': sanitize_string(info.get('longName', symbol)),
                    'price': current_price,
                    'change': change,
                    'changePercent': change_percent,
                    'currency': get_currency(symbol)
                }

                data = sanitize_for_json(data)
                set_cached_data(cache_key, data)
                results.append(data)

            except Exception as e:
                logger.error(f"Erreur watchlist pour {symbol}: {e}")
                results.append({
                    'symbol': symbol,
                    'error': str(e)
                })

        return jsonify(results)

    except Exception as e:
        logger.error(f"Erreur watchlist générale: {e}")
        return jsonify({'error': str(e)}), 500

# ============================================================
# ROUTE TOP PERFORMERS US
# ============================================================

@app.route('/api/top-performers')
def get_top_performers():
    try:
        # Sélection des plus grandes valeurs US + SpaceX
        symbols = ['SPCX', 'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'META', 'TSLA',
                  'JPM', 'V', 'PG', 'JNJ', 'UNH', 'WMT', 'BAC', 'DIS',
                  'NFLX', 'AMD', 'INTC', 'CRM', 'ORCL', 'IBM', 'CSCO',
                  'KO', 'PEP', 'MCD', 'NKE', 'XOM', 'CVX', 'GE', 'CAT']

        performers = []
        for symbol in symbols:
            try:
                cache_key = f"performer_{symbol}"
                cached = get_cached_data(cache_key)
                
                if cached:
                    performers.append(cached)
                    continue

                ticker = yf.Ticker(symbol)
                
                try:
                    info = ticker.info
                except:
                    info = {}

                try:
                    hist = ticker.history(period='1d')
                except:
                    hist = pd.DataFrame()

                current = safe_float(info.get('regularMarketPrice', 0))
                if current == 0 and not hist.empty:
                    current = safe_float(hist['Close'].iloc[-1])

                prev = safe_float(info.get('regularMarketPreviousClose', 0))
                if prev == 0 and len(hist) > 1:
                    prev = safe_float(hist['Close'].iloc[-2])

                change_pct = ((current - prev) / prev * 100) if prev else 0

                data = {
                    'symbol': symbol,
                    'name': sanitize_string(info.get('longName', symbol)),
                    'price': current,
                    'changePercent': change_pct,
                    'currency': get_currency(symbol)
                }

                data = sanitize_for_json(data)
                set_cached_data(cache_key, data)
                performers.append(data)

            except Exception as e:
                logger.warning(f"Erreur performer {symbol}: {e}")
                continue

        performers.sort(key=lambda x: x['changePercent'], reverse=True)
        return jsonify(performers[:20])

    except Exception as e:
        logger.error(f"Erreur top performers: {e}")
        return jsonify({'error': str(e)}), 500

# ============================================================
# ROUTE MARKET STATUS US
# ============================================================

@app.route('/api/market-status')
def get_market_status():
    try:
        us_now = datetime.now(US_TIMEZONE)
        hour = us_now.hour
        minute = us_now.minute
        weekday = us_now.weekday()

        if weekday >= 5:
            return jsonify({
                'status': 'closed',
                'label': 'Fermé (weekend)',
                'icon': '🔴'
            })

        # US Market hours: 9:30 AM - 4:00 PM ET
        is_open = (hour >= 9 and (hour < 16 or (hour == 16 and minute <= 0)))

        return jsonify({
            'status': 'open' if is_open else 'closed',
            'label': 'Ouvert' if is_open else 'Fermé',
            'icon': '🟢' if is_open else '🔴',
            'time': us_now.strftime('%H:%M:%S')
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500

# ============================================================
# ROUTES STATIQUES
# ============================================================

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/static/<path:filename>')
def static_files(filename):
    return send_from_directory('static', filename)

# ============================================================
# MAIN
# ============================================================

if __name__ == '__main__':
    os.makedirs('templates', exist_ok=True)
    os.makedirs('static/css', exist_ok=True)

    print("=" * 60)
    print("🚀 US STOCK TRADER - Version SpaceX incluse")
    print("=" * 60)
    print(f"🌐 http://localhost:5001")
    print(f"📊 Cache activé ({CACHE_DURATION}s)")
    print("=" * 60)
    print("📈 Indices US disponibles:")
    print("   ^GSPC - S&P 500")
    print("   ^IXIC - NASDAQ")
    print("   ^DJI  - Dow Jones")
    print("   ^RUT  - Russell 2000")
    print("   ^VIX  - Volatility Index")
    print("=" * 60)
    print("🚀 Nouveau ticker:")
    print("   SPCX  - SpaceX (Nasdaq Global Select Market)")
    print("=" * 60)
    print("💡 Pour vider le cache: http://localhost:5001/api/clear-cache")
    print("=" * 60)
    
    app.run(host='0.0.0.0', port=5001, debug=True)
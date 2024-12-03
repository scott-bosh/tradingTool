import numpy as np
import pandas as pd
import yfinance as yf
import ta
from typing import Dict, List, Tuple
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class MultiTimeframeAnalyzer:
    """
    Comprehensive multi-timeframe stock analysis tool
    Identifies key levels across different timeframes
    """
    def __init__(self, symbol: str):
        """
        Initialize the analyzer with a stock symbol
        
        Args:
            symbol (str): Stock ticker symbol
        """
        self.symbol = symbol
        self.timeframes = {
            '1D': '1d',   # 1 day
            '1H': '1h',   # 1 hour
            '15M': '15m', # 15 minutes
            '1M': '1m'    # 1 minute
        }
    
    def fetch_historical_data(self, timeframe: str, period: str = '30d') -> pd.DataFrame:
        """
        Fetch historical price data for a specific timeframe
        
        Args:
            timeframe (str): Timeframe interval
            period (str): Historical data lookback period
        
        Returns:
            pd.DataFrame: Historical price data with technical indicators
        """
        try:
            # Fetch data using yfinance
            data = yf.download(self.symbol, period=period, interval=timeframe)
            
            # Add technical indicators
            data = self._add_technical_indicators(data)
            
            return data
        except Exception as e:
            logger.error(f"Error fetching data for {self.symbol} on {timeframe} timeframe: {e}")
            raise
    
    def _add_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add technical indicators to the price data
        
        Args:
            df (pd.DataFrame): Raw price data
        
        Returns:
            pd.DataFrame: Data with additional technical indicators
        """
        if len(df) == 0:
            return df
        
        # Moving Averages
        df['MA_20'] = df['Close'].rolling(window=20).mean()
        df['MA_50'] = df['Close'].rolling(window=50).mean()
        df['MA_200'] = df['Close'].rolling(window=200).mean()
        
        # Identify swing highs and lows
        df['Swing_High'] = df['High'].rolling(window=5, center=True).max()
        df['Swing_Low'] = df['Low'].rolling(window=5, center=True).min()
        
        return df
    
    def identify_key_levels(self, df: pd.DataFrame) -> Dict[str, List[float]]:
        """
        Identify key price levels for trading
        
        Args:
            df (pd.DataFrame): Price data with technical indicators
        
        Returns:
            Dict with different types of key levels
        """
        key_levels = {
            'support_levels': self._find_support_levels(df),
            'resistance_levels': self._find_resistance_levels(df),
            'moving_averages': [
                df['MA_20'].iloc[-1],
                df['MA_50'].iloc[-1],
                df['MA_200'].iloc[-1]
            ],
            'recent_swing_high': df['Swing_High'].iloc[-5:].max(),
            'recent_swing_low': df['Swing_Low'].iloc[-5:].min()
        }
        
        return key_levels
    
    def _find_support_levels(self, df: pd.DataFrame, num_levels: int = 3) -> List[float]:
        """
        Identify key support levels
        
        Args:
            df (pd.DataFrame): Price data
            num_levels (int): Number of support levels to identify
        
        Returns:
            List of support levels
        """
        # Use local minima as support levels
        local_lows = df['Low'].rolling(window=5, center=True).min()
        
        # Remove duplicates and sort
        support_levels = sorted(set(local_lows.dropna().tail(10)))
        
        return support_levels[:num_levels]
    
    def _find_resistance_levels(self, df: pd.DataFrame, num_levels: int = 3) -> List[float]:
        """
        Identify key resistance levels
        
        Args:
            df (pd.DataFrame): Price data
            num_levels (int): Number of resistance levels to identify
        
        Returns:
            List of resistance levels
        """
        # Use local maxima as resistance levels
        local_highs = df['High'].rolling(window=5, center=True).max()
        
        # Remove duplicates and sort
        resistance_levels = sorted(set(local_highs.dropna().tail(10)), reverse=True)
        
        return resistance_levels[:num_levels]
    
    def analyze_stock(self) -> Dict[str, Dict[str, List[float]]]:
        """
        Perform comprehensive multi-timeframe analysis
        
        Returns:
            Dict with key levels for each timeframe
        """
        analysis_results = {}
        
        for timeframe_name, timeframe_code in self.timeframes.items():
            try:
                # Fetch data for the specific timeframe
                data = self.fetch_historical_data(timeframe_code)
                
                # Identify key levels
                key_levels = self.identify_key_levels(data)
                
                analysis_results[timeframe_name] = key_levels
            except Exception as e:
                logger.error(f"Analysis failed for {timeframe_name} timeframe: {e}")
                analysis_results[timeframe_name] = {}
        
        return analysis_results
    
    def print_analysis_results(self, results: Dict[str, Dict[str, List[float]]]):
        """
        Print the analysis results in a readable format
        
        Args:
            results (Dict): Multi-timeframe analysis results
        """
        print(f"\n--- Multi-Timeframe Analysis for {self.symbol} ---")
        
        for timeframe, levels in results.items():
            print(f"\n{timeframe} Timeframe:")
            
            # Print Support Levels
            print("Support Levels:")
            for support in levels.get('support_levels', []):
                print(f"  - {support:.2f}")
            
            # Print Resistance Levels
            print("Resistance Levels:")
            for resistance in levels.get('resistance_levels', []):
                print(f"  - {resistance:.2f}")
            
            # Print Moving Averages
            print("Moving Averages:")
            for ma in levels.get('moving_averages', []):
                print(f"  - {ma:.2f}")
            
            # Print Swing High/Low
            print(f"Recent Swing High: {levels.get('recent_swing_high', 'N/A'):.2f}")
            print(f"Recent Swing Low: {levels.get('recent_swing_low', 'N/A'):.2f}")

def main():
    """
    Main function to run the multi-timeframe stock analysis
    """
    try:
        # Get stock symbol from user
        symbol = input("Enter the stock symbol (e.g., AAPL): ").upper()
        
        # Create analyzer
        analyzer = MultiTimeframeAnalyzer(symbol)
        
        # Perform analysis
        analysis_results = analyzer.analyze_stock()
        
        # Print results
        analyzer.print_analysis_results(analysis_results)
    
    except Exception as e:
        logger.error(f"Analysis failed: {e}")

if __name__ == "__main__":
    main()

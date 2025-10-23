"""
Polygon.io flat files downloader for crypto minute aggregates.
"""

import os
import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Optional, Dict, Any
import gzip
import logging
from tqdm import tqdm
import time

logger = logging.getLogger(__name__)


class PolygonFlatFilesDownloader:
    """
    Download and cache Polygon.io crypto flat files.
    """
    
    def __init__(self, api_key: str, data_root: str = "./app/data/cache"):
        self.api_key = api_key
        self.data_root = Path(data_root)
        self.data_root.mkdir(parents=True, exist_ok=True)
        
        # Base URL for Polygon flat files
        self.base_url = "https://files.polygon.io/flatfiles/global_crypto/minute_aggs_v1"
        
        # Rate limiting
        self.last_request_time = 0
        self.min_request_interval = 0.1  # 100ms between requests
        
    def _rate_limit(self):
        """Apply rate limiting between requests."""
        current_time = time.time()
        time_since_last = current_time - self.last_request_time
        
        if time_since_last < self.min_request_interval:
            time.sleep(self.min_request_interval - time_since_last)
            
        self.last_request_time = time.time()
    
    def _get_file_url(self, date: datetime) -> str:
        """
        Generate Polygon flat file URL for a given date.
        
        Args:
            date: Date to download data for
            
        Returns:
            URL string
        """
        year = date.strftime("%Y")
        month = date.strftime("%m")
        date_str = date.strftime("%Y-%m-%d")
        
        return f"{self.base_url}/{year}/{month}/{date_str}.csv.gz"
    
    def _get_cache_path(self, date: datetime) -> Path:
        """
        Get local cache path for a given date.
        
        Args:
            date: Date to get cache path for
            
        Returns:
            Path object for cached file
        """
        date_str = date.strftime("%Y-%m-%d")
        return self.data_root / f"{date_str}.csv.gz"
    
    def _download_file(self, url: str, cache_path: Path) -> bool:
        """
        Download a single file from Polygon.io.
        
        Args:
            url: URL to download from
            cache_path: Local path to save file
            
        Returns:
            True if successful, False otherwise
        """
        try:
            self._rate_limit()
            
            headers = {
                'Authorization': f'Bearer {self.api_key}',
                'User-Agent': 'ML-Trading-System/1.0'
            }
            
            response = requests.get(url, headers=headers, timeout=30)
            response.raise_for_status()
            
            # Save compressed file
            with open(cache_path, 'wb') as f:
                f.write(response.content)
                
            logger.info(f"Downloaded: {cache_path.name}")
            return True
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Failed to download {url}: {e}")
            return False
        except Exception as e:
            logger.error(f"Unexpected error downloading {url}: {e}")
            return False
    
    def download_date_range(
        self,
        start_date: datetime,
        end_date: datetime,
        symbols: Optional[List[str]] = None,
        force_redownload: bool = False
    ) -> Dict[str, Any]:
        """
        Download Polygon flat files for a date range.
        
        Args:
            start_date: Start date for download
            end_date: End date for download
            symbols: List of symbols to filter (None for all)
            force_redownload: Force re-download of existing files
            
        Returns:
            Dictionary with download statistics
        """
        stats = {
            'total_files': 0,
            'downloaded': 0,
            'skipped': 0,
            'failed': 0,
            'start_date': start_date,
            'end_date': end_date
        }
        
        # Generate date range
        current_date = start_date
        date_list = []
        
        while current_date <= end_date:
            date_list.append(current_date)
            current_date += timedelta(days=1)
        
        stats['total_files'] = len(date_list)
        
        logger.info(f"Downloading {len(date_list)} files from {start_date.date()} to {end_date.date()}")
        
        # Download files with progress bar
        for date in tqdm(date_list, desc="Downloading files"):
            url = self._get_file_url(date)
            cache_path = self._get_cache_path(date)
            
            # Skip if file exists and not forcing redownload
            if cache_path.exists() and not force_redownload:
                stats['skipped'] += 1
                continue
            
            # Download file
            if self._download_file(url, cache_path):
                stats['downloaded'] += 1
            else:
                stats['failed'] += 1
        
        logger.info(f"Download complete: {stats['downloaded']} downloaded, {stats['skipped']} skipped, {stats['failed']} failed")
        return stats
    
    def get_available_dates(self) -> List[datetime]:
        """
        Get list of available cached dates.
        
        Returns:
            List of datetime objects for cached files
        """
        dates = []
        
        for file_path in self.data_root.glob("*.csv.gz"):
            try:
                # Extract date from filename
                date_str = file_path.stem.replace('.csv', '')
                date = datetime.strptime(date_str, "%Y-%m-%d")
                dates.append(date)
            except ValueError:
                continue
        
        return sorted(dates)
    
    def cleanup_old_files(self, days_to_keep: int = 30) -> int:
        """
        Remove old cached files to save disk space.
        
        Args:
            days_to_keep: Number of days of files to keep
            
        Returns:
            Number of files removed
        """
        cutoff_date = datetime.now() - timedelta(days=days_to_keep)
        removed_count = 0
        
        for file_path in self.data_root.glob("*.csv.gz"):
            try:
                # Extract date from filename
                date_str = file_path.stem.replace('.csv', '')
                file_date = datetime.strptime(date_str, "%Y-%m-%d")
                
                if file_date < cutoff_date:
                    file_path.unlink()
                    removed_count += 1
                    logger.info(f"Removed old file: {file_path.name}")
                    
            except ValueError:
                continue
        
        logger.info(f"Cleaned up {removed_count} old files")
        return removed_count

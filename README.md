# EcoPulse-Analytics
Transforming OpenAQ big data into actionable urban intelligence. This engine provides a scientifically audited view of city-wide air quality through high-resolution spatial-temporal visualizations and multi-regional regression analysis.

---

## Technical Overview: Data Pipeline
To fetch and process the data for the 100 stations, follow these steps in your terminal:

### 1. Start the Data Ingestion
The ingestion script fetches hourly data for all of 2025. It is designed to be "smart"—it tracks progress locally, so if it stops, it will pick up right where it left off.

```powershell
python ingest_data.py
```
> [!IMPORTANT]
> **Only run ONE instance of this command.** Running multiple instances can lead to API blocks and data corruption.

### 2. Monitor Progress
*   **Terminal**: The script prints the current station (e.g., `Processing location 3/100`).
*   **Progress File**: Check `ingestion_progress.json` to see the current index.

### 3. Consolidate and Clean
Once all 100 locations are fetched and the script finishes, run:

```powershell
python clean_data.py
```
This will merge the 100 individual files into a single master file named `urban_air_quality_2025_master.parquet` and handle any missing data.

---

### Key Project Components:
- **`dashboard.py`**: The Streamlit interface for visualization.
- **`ingest_data.py`**: Handles API rate limits, monthly chunks, and retries.
- **`clean_data.py`**: Perpetual pivoting, interpolation, and consolidation.
- **`air_quality_data/`**: Raw data repository (excluded from Git).

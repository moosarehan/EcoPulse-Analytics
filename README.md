# Urban Environmental Intelligence Challenge: Data Pipeline

To fetch and process the data for the 100 stations, follow these steps in your terminal:

## 1. Start the Data Ingestion
The ingestion script fetches hourly data for all of 2025. It is designed to be "smart"—it tracks progress locally, so if it stops, it will pick up right where it left off.

```powershell
python ingest_data.py
```
> [!IMPORTANT]
> **Only run ONE instance of this command.** I noticed multiple instances were running, which can lead to API blocks and data corruption. I have stopped the extra ones for you.

## 2. Monitor Progress
*   **Terminal**: The script prints the current station (e.g., `Processing location 3/100`).
*   **Progress File**: Check `ingestion_progress.json` to see the current index.

## 3. Consolidate and Clean
Once all 100 locations are fetched and the script finishes, run:

```powershell
python clean_data.py
```
This will merge the 100 individual files into a single master file named `urban_air_quality_2025_master.parquet` and handle any missing data.

---

### Key Project Components:
- **`ingest_data.py`**: The "Backend Engineer"—handles rate limits, monthly chunks, and retries.
- **`clean_data.py`**: The "Data Scientist"—performs pivoting, interpolation, and consolidation.
- **`air_quality_data/`**: Your raw data repository.

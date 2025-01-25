# NBA Predictions Bot

An AI-powered NBA predictions and betting insights bot that provides game predictions, betting odds, and analysis.

## Features
- Real-time NBA game predictions
- Betting odds and analysis
- Team-specific insights
- Historical game data analysis
- Interactive web interface

## Tech Stack
- Backend: FastAPI, Python
- Frontend: HTML, CSS, JavaScript
- Database: Supabase
- APIs: BallDontLie API, OpenAI API

## Setup
1. Clone the repository
2. Install dependencies: `pip install -r requirements.txt`
3. Set up environment variables in `.env`:
   - BALLDONTLIE_API_KEY
   - OPENAI_API_KEY
   - API_BEARER_TOKEN
   - SUPABASE_URL
   - SUPABASE_SERVICE_KEY

4. Run the application:
```python
uvicorn nba_agent:app --reload
```

## Usage
Visit `http://localhost:8000` in your browser to interact with the bot.

## API Endpoints
- POST `/api/nba_agent`: Main prediction endpoint

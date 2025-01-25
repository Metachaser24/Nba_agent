from typing import List, Optional, Dict, Any
from fastapi import FastAPI, HTTPException, Security, Depends, Request
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from fastapi.middleware.cors import CORSMiddleware
from supabase import create_client, Client
from pydantic import BaseModel
from dotenv import load_dotenv
from pathlib import Path
import sys
import os
import logging
from openai import OpenAI
import asyncio
import requests
from datetime import datetime, timedelta
import dateparser
import httpx
import re
from fastapi.responses import HTMLResponse

# At the top of the file, after imports
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# After loading environment variables
load_dotenv()

# Check all required environment variables
required_vars = [
    "BALLDONTLIE_API_KEY",
    "OPENAI_API_KEY",
    "API_BEARER_TOKEN",
    "SUPABASE_URL",
    "SUPABASE_SERVICE_KEY"
]

missing_vars = [var for var in required_vars if not os.getenv(var)]
if missing_vars:
    raise ValueError(f"Missing required environment variables: {', '.join(missing_vars)}")

logger.info("All required environment variables are set")

# Initialize FastAPI app
app = FastAPI()
security = HTTPBearer()

# Supabase setup
supabase: Client = create_client(
    os.getenv("SUPABASE_URL"),
    os.getenv("SUPABASE_SERVICE_KEY")
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Request/Response Models
class AgentRequest(BaseModel):
    query: str
    user_id: str
    request_id: str
    session_id: str

class AgentResponse(BaseModel):
    success: bool

class NBAPredictor:
    def __init__(self):
        """Initialize the NBA predictor with API configuration"""
        self.base_url = "https://api.balldontlie.io/v1"
        self.api_key = os.getenv("BALLDONTLIE_API_KEY")
        self.openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    async def get_games(self, date: str) -> List[Dict]:
        """Fetch games for a specific date"""
        logger.info(f"Fetching games for date: {date}")
        url = f"{self.base_url}/games"
        headers = {'Authorization': self.api_key}
        params = {'dates[]': date}
        
        try:
            response = requests.get(url, headers=headers, params=params)
            response.raise_for_status()
            games = response.json()['data']
            logger.info(f"Found {len(games)} games for {date}")
            return games
        except Exception as e:
            logger.error(f"Error fetching games: {str(e)}")
            raise

    async def get_team_injuries(self, team_id: int) -> List[Dict]:
        """Fetch current injuries for a team"""
        url = f"{self.base_url}/player_injuries"
        headers = {'Authorization': self.api_key}
        params = {'team_ids[]': [team_id]}
        
        try:
            response = requests.get(url, headers=headers, params=params)
            response.raise_for_status()
            return response.json()['data']
        except Exception as e:
            logger.error(f"Error fetching injuries: {str(e)}")
            return []

    async def get_standings(self, season: int) -> Dict[int, Dict]:
        """Fetch standings for the current season"""
        url = f"{self.base_url}/standings"
        headers = {'Authorization': self.api_key}
        params = {'season': season}
        
        try:
            response = requests.get(url, headers=headers, params=params)
            response.raise_for_status()
            standings_data = response.json()['data']
            
            return {
                standing['team']['id']: {
                    'wins': standing['wins'],
                    'losses': standing['losses'],
                    'home_record': standing['home_record'],
                    'road_record': standing['road_record']
                }
                for standing in standings_data
            }
        except Exception as e:
            logger.error(f"Error fetching standings: {str(e)}")
            return {}

    async def get_betting_odds(self, game_id: int = None, game_date: str = None) -> List[Dict]:
        """Fetch betting odds for a game"""
        try:
            async with httpx.AsyncClient() as client:
                url = "https://api.balldontlie.io/v1/odds"
                params = {}
                if game_id:
                    params['game_id'] = game_id
                if game_date:
                    params['date'] = game_date
                    
                response = await client.get(
                    url,
                    params=params,
                    headers={"Authorization": self.api_key}
                )
                
                if response.status_code != 200:
                    logger.error(f"Odds API error: {response.status_code} - {response.text}")
                    return []
                
                data = response.json()
                logger.info(f"Odds data received: {data}")
                return data.get('data', [])
                
        except Exception as e:
            logger.error(f"Error fetching betting odds: {str(e)}")
            return []

    def _generate_prediction(self, home_team: Dict, away_team: Dict, 
                           standings: Dict, home_injuries: List, away_injuries: List,
                           odds_data: List = None) -> str:
        """Generate prediction with betting odds"""
        # Basic prediction logic
        home_standing = standings.get(home_team['id'], {})
        away_standing = standings.get(away_team['id'], {})
        
        # Calculate win probability based on records
        home_wins = home_standing.get('wins', 0)
        home_losses = home_standing.get('losses', 0)
        away_wins = away_standing.get('wins', 0)
        away_losses = away_standing.get('losses', 0)
        
        # Determine winner and probability
        home_win_pct = home_wins / (home_wins + home_losses) if (home_wins + home_losses) > 0 else 0.5
        away_win_pct = away_wins / (away_wins + away_losses) if (away_wins + away_losses) > 0 else 0.5
        
        if home_win_pct > away_win_pct:
            winner = home_team['full_name']
            probability = int(home_win_pct * 100)
        else:
            winner = away_team['full_name']
            probability = int(away_win_pct * 100)
        
        # Generate analysis
        reasoning = (
            f"The {winner} have a better overall record at "
            f"{home_wins if winner == home_team['full_name'] else away_wins}-"
            f"{home_losses if winner == home_team['full_name'] else away_losses}. "
            f"The {home_team['full_name']} are {len(home_injuries)} players down, while "
            f"the {away_team['full_name']} have {len(away_injuries)} players out."
        )
        
        # Determine confidence
        confidence = "High" if abs(home_win_pct - away_win_pct) > 0.2 else "Medium"
        
        # Format betting lines
        betting_lines = "\n\nBetting Lines:"
        if odds_data:
            latest_spread = None
            latest_over_under = None
            
            # Add logging to debug odds data
            logger.info(f"Processing odds data for {away_team['full_name']} @ {home_team['full_name']}")
            logger.info(f"Odds data: {odds_data}")
            
            for odds in odds_data:
                if not odds:
                    continue
                
                if odds.get('type') == 'spread':
                    if not latest_spread or odds.get('last_update', '') > latest_spread.get('last_update', ''):
                        latest_spread = odds
                elif odds.get('type') == 'over/under':
                    if not latest_over_under or odds.get('last_update', '') > latest_over_under.get('last_update', ''):
                        latest_over_under = odds

            # Add error handling and logging for spread
            if latest_spread:
                try:
                    away_spread = latest_spread.get('away_spread')
                    if away_spread is not None:
                        spread_value = float(away_spread)
                        betting_lines += f"\n📊 {away_team['full_name']} {spread_value:+.1f}"
                    else:
                        logger.warning(f"No away_spread found in latest_spread: {latest_spread}")
                except (ValueError, TypeError) as e:
                    logger.error(f"Error processing spread: {str(e)}")
                    logger.error(f"Latest spread data: {latest_spread}")

            # Add error handling and logging for over/under
            if latest_over_under:
                try:
                    total = latest_over_under.get('over_under')
                    if total is not None:
                        total_value = float(total)
                        betting_lines += f"\n📈 O {total_value:.1f}"
                    else:
                        logger.warning(f"No over_under found in latest_over_under: {latest_over_under}")
                except (ValueError, TypeError) as e:
                    logger.error(f"Error processing over/under: {str(e)}")
                    logger.error(f"Latest over/under data: {latest_over_under}")

        # Construct final prediction
        prediction = f"""🏀 {away_team['full_name']} (Away) @ {home_team['full_name']} (Home)

Winner: {winner} ({probability}%)

Analysis: {reasoning}

Confidence: {confidence}{betting_lines}"""

        return prediction

    def _analyze_over_under(self, total: float, home_team: Dict, away_team: Dict, standings: Dict) -> str:
        """Analyze over/under based on team statistics"""
        try:
            home_stats = standings.get(home_team['id'], {})
            away_stats = standings.get(away_team['id'], {})
            
            # Simple analysis based on team scoring averages
            home_avg_points = home_stats.get('points_per_game', 0)
            away_avg_points = away_stats.get('points_per_game', 0)
            combined_avg = home_avg_points + away_avg_points
            
            if combined_avg > total:
                return f"Prediction: OVER {total}\nReasoning: Teams combine for {combined_avg:.1f} points per game on average"
            else:
                return f"Prediction: UNDER {total}\nReasoning: Teams combine for {combined_avg:.1f} points per game on average"
        except Exception:
            return "Unable to analyze over/under"

    async def analyze_matchup(self, game: Dict) -> Dict:
        """Analyze matchup and generate prediction"""
        try:
            home_team = game['home_team']
            away_team = game['visitor_team']
            
            tasks = [
                self.get_team_injuries(home_team['id']),
                self.get_team_injuries(away_team['id']),
                self.get_standings(2024)
            ]
            
            results = await asyncio.gather(*tasks)
            home_injuries, away_injuries, standings = results

            # Get betting odds
            odds_data = await self.get_betting_odds(game_id=game['id'])
            
            # Generate prediction with odds
            prediction = self._generate_prediction(
                home_team, 
                away_team,
                standings,
                home_injuries,
                away_injuries,
                odds_data
            )

            return {
                "matchup": f"{away_team['full_name']} (Away) @ {home_team['full_name']} (Home)",
                "prediction": prediction,
                "data": {
                    "teams": {
                        "away": {
                            "name": away_team['full_name'],
                            "record": f"{standings.get(away_team['id'], {}).get('wins', 0)}-{standings.get(away_team['id'], {}).get('losses', 0)}",
                            "road_record": standings.get(away_team['id'], {}).get('road_record'),
                            "injuries": len(away_injuries)
                        },
                        "home": {
                            "name": home_team['full_name'],
                            "record": f"{standings.get(home_team['id'], {}).get('wins', 0)}-{standings.get(home_team['id'], {}).get('losses', 0)}",
                            "home_record": standings.get(home_team['id'], {}).get('home_record'),
                            "injuries": len(home_injuries)
                        }
                    }
                }
            }
        except Exception as e:
            logger.error(f"Error in analyze_matchup: {str(e)}")
            raise

    async def parse_game_date(self, query: str) -> str:
        """Parse date from query and return in YYYY-MM-DD format"""
        query_lower = query.lower()
        
        try:
            # Handle specific date mentions like "Jan 27" or "January 27"
            if any(month in query_lower for month in ['jan', 'january']):
                # Extract the date using a more specific pattern
                date_pattern = r'jan(?:uary)?\s+(\d{1,2})'
                match = re.search(date_pattern, query_lower)
                if match:
                    day = int(match.group(1))
                    # Use current year, but handle year boundary cases
                    current_date = datetime.now()
                    year = current_date.year
                    # If the specified date is earlier than today and we're in a new year,
                    # assume it's for the next year
                    if current_date.month == 1 and day < current_date.day:
                        year += 1
                    return f"{year}-01-{day:02d}"

            # Handle relative dates
            if 'tomorrow' in query_lower:
                return (datetime.now() + timedelta(days=1)).strftime('%Y-%m-%d')
            elif 'yesterday' in query_lower:
                return (datetime.now() - timedelta(days=1)).strftime('%Y-%m-%d')
            elif 'today' in query_lower or 'tonight' in query_lower:
                return datetime.now().strftime('%Y-%m-%d')
            
            # Try to parse any other date format
            parsed_date = dateparser.parse(query)
            if parsed_date:
                return parsed_date.strftime('%Y-%m-%d')
            
            # Default to today if no date is specified
            return datetime.now().strftime('%Y-%m-%d')
            
        except Exception as e:
            logger.error(f"Error parsing date from query: {str(e)}")
            # Default to today's date if parsing fails
            return datetime.now().strftime('%Y-%m-%d')

def verify_token(credentials: HTTPAuthorizationCredentials = Security(security)) -> bool:
    """Verify the bearer token against environment variable."""
    expected_token = os.getenv("API_BEARER_TOKEN")
    if not expected_token:
        raise HTTPException(
            status_code=500,
            detail="API_BEARER_TOKEN environment variable not set"
        )
    if credentials.credentials != expected_token:
        raise HTTPException(
            status_code=401,
            detail="Invalid authentication token"
        )
    return True

async def fetch_conversation_history(session_id: str, limit: int = 10) -> List[Dict[str, Any]]:
    """Fetch the most recent conversation history for a session."""
    try:
        response = supabase.table("messages") \
            .select("*") \
            .eq("session_id", session_id) \
            .order("created_at", desc=True) \
            .limit(limit) \
            .execute()
        
        # Convert to list and reverse to get chronological order
        messages = response.data[::-1]
        return messages
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to fetch conversation history: {str(e)}")

async def store_message(session_id: str, message_type: str, content: str, data: Optional[Dict] = None):
    """Store a message in the Supabase messages table."""
    message_obj = {
        "type": message_type,
        "content": content
    }
    if data:
        message_obj["data"] = data

    try:
        supabase.table("messages").insert({
            "session_id": session_id,
            "message": message_obj
        }).execute()
    except Exception as e:
        logger.error(f"Failed to store message: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to store message: {str(e)}")

@app.post("/api/nba_agent", response_model=AgentResponse)
async def nba_agent(
    request: AgentRequest,
    authenticated: bool = Depends(verify_token)
):
    try:
        logger.info(f"Received request: {request.query}")
        
        # Store user's message first
        await store_message(
            session_id=request.session_id,
            message_type="human",
            content=request.query,
            data={"request_id": request.request_id}
        )

        predictor = NBAPredictor()
        game_date = await predictor.parse_game_date(request.query)
        logger.info(f"Parsed date for games: {game_date}")
        
        games = await predictor.get_games(game_date)
        logger.info(f"Found {len(games)} games for {game_date}")
        
        if not games:
            agent_response = f"I couldn't find any NBA games scheduled for {game_date}."
            response_data = None
        else:
            # Check if query is about a specific team
            query_lower = request.query.lower()
            team_specific_games = []
            
            # List of team names and their common variations
            team_names = {
                "celtics": ["boston", "celtics"],
                "nets": ["brooklyn", "nets"],
                "knicks": ["new york", "ny", "knicks"],
                "sixers": ["philadelphia", "philly", "76ers", "sixers"],
                "raptors": ["toronto", "raptors"],
                "bulls": ["chicago", "bulls"],
                "cavaliers": ["cleveland", "cavs", "cavaliers"],
                "pistons": ["detroit", "pistons"],
                "pacers": ["indiana", "pacers"],
                "bucks": ["milwaukee", "bucks"],
                "hawks": ["atlanta", "hawks"],
                "hornets": ["charlotte", "hornets"],
                "heat": ["miami", "heat"],
                "magic": ["orlando", "magic"],
                "wizards": ["washington", "wizards"],
                "nuggets": ["denver", "nuggets"],
                "timberwolves": ["minnesota", "wolves", "timberwolves"],
                "thunder": ["oklahoma", "okc", "thunder"],
                "blazers": ["portland", "blazers", "trail blazers"],
                "jazz": ["utah", "jazz"],
                "warriors": ["golden state", "gsw", "warriors"],
                "clippers": ["la clippers", "lac", "clippers"],
                "lakers": ["la lakers", "lal", "lakers"],
                "suns": ["phoenix", "suns"],
                "kings": ["sacramento", "kings"],
                "mavericks": ["dallas", "mavs", "mavericks"],
                "rockets": ["houston", "rockets"],
                "grizzlies": ["memphis", "grizzlies"],
                "pelicans": ["new orleans", "pels", "pelicans"],
                "spurs": ["san antonio", "spurs"]
            }
            
            # Check if query contains any team names
            requested_team = None
            for team, variations in team_names.items():
                if any(variation in query_lower for variation in variations):
                    requested_team = team
                    break
            
            if requested_team:
                # Filter games for the requested team
                team_specific_games = [
                    game for game in games 
                    if any(variation in game['home_team']['full_name'].lower() for variation in team_names[requested_team])
                    or any(variation in game['visitor_team']['full_name'].lower() for variation in team_names[requested_team])
                ]
                games = team_specific_games
            
            # Analyze filtered games
            all_predictions = []
            for game in games:
                prediction_data = await predictor.analyze_matchup(game)
                all_predictions.append({
                    "matchup": prediction_data["matchup"],
                    "prediction": prediction_data["prediction"],
                    "stats": prediction_data["data"]
                })
            
            # Create appropriate response based on query type
            if requested_team and not team_specific_games:
                agent_response = f"I couldn't find any games scheduled for {requested_team.title()} on {game_date}."
            elif requested_team:
                agent_response = f"Here's my prediction for the {requested_team.title()} game on {game_date}:\n\n"
                for pred in all_predictions:
                    agent_response += f"🏀 {pred['matchup']}\n"
                    agent_response += f"{pred['prediction']}\n\n"
            else:
                agent_response = f"I found {len(games)} games scheduled for {game_date}. Here are my predictions:\n\n"
                for pred in all_predictions:
                    agent_response += f"🏀 {pred['matchup']}\n"
                    agent_response += f"{pred['prediction']}\n\n"
            
            response_data = {
                "date": game_date,
                "games_count": len(games),
                "predictions": all_predictions,
                "team_specific": requested_team is not None
            }

        # Store AI's response
        await store_message(
            session_id=request.session_id,
            message_type="ai",
            content=agent_response,
            data={
                "request_id": request.request_id,
                **(response_data or {})
            }
        )

        return AgentResponse(success=True)

    except Exception as e:
        logger.error(f"Error processing request: {str(e)}", exc_info=True)
        # Store error message in conversation
        await store_message(
            session_id=request.session_id,
            message_type="ai",
            content="I apologize, but I encountered an error processing your request.",
            data={"error": str(e), "request_id": request.request_id}
        )
        return AgentResponse(success=False)

@app.get("/", response_class=HTMLResponse)
async def root(request: Request):
    with open("templates/index.html") as f:
        return HTMLResponse(content=f.read())

if __name__ == "__main__":
    import uvicorn
    # Feel free to change the port here if you need
    uvicorn.run(app, host="0.0.0.0", port=8001)
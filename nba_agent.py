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
import pytz

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
    allow_origins=["https://nbaagent-production.up.railway.app", "http://localhost:8001"],
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

    def _get_current_nba_season(self) -> int:
        """
        Get the NBA season based on the game date.
        For the 2024-25 season, use 2024.
        """
        return 2024  # Hardcode to 2024 for now since that's what the API expects

    async def _is_notable_player(self, player: Dict) -> bool:
        """Determine if a player is notable based on various factors."""
        try:
            # Get player's season averages
            season_stats = await self._get_season_averages(player['id'])
            
            if season_stats:
                # Consider a player notable if they meet any of these criteria
                return any([
                    season_stats.get('pts', 0) >= 10,  # Scores 10+ PPG
                    season_stats.get('reb', 0) >= 5,   # 5+ RPG
                    season_stats.get('ast', 0) >= 4,   # 4+ APG
                    season_stats.get('min', '0') >= '20'  # Plays 20+ minutes
                ])
            
            return False
        except Exception as e:
            logger.error(f"Error checking if player is notable: {str(e)}")
            return False

    async def _get_season_averages(self, player_id: int) -> Dict:
        """Get player's season averages for the current season."""
        current_season = 2024  # NBA season 2024-25
        url = f"{self.base_url}/season_averages"
        params = {
            "season": current_season,
            "player_ids[]": [player_id]  # API expects array of player IDs
        }
        headers = {"Authorization": self.api_key}
        
        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(url, params=params, headers=headers)
                response.raise_for_status()
                data = response.json()
                return data['data'][0] if data.get('data') else {}
        except Exception as e:
            logger.error(f"Error fetching season averages: {str(e)}")
            return {}

    async def _get_advanced_stats(self, player_id: int, season: int) -> Dict:
        """Get player's advanced stats."""
        url = f"{self.base_url}/stats/advanced"
        params = {
            "player_ids[]": [player_id],
            "seasons[]": [season],
            "per_page": 100
        }
        headers = {"Authorization": self.api_key}
        
        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(url, params=params, headers=headers)
                response.raise_for_status()
                return response.json()
        except Exception as e:
            logger.error(f"Error fetching advanced stats: {str(e)}")
            return {}

    async def _get_team_standings(self, player_id: int) -> Dict:
        """Get current team standings."""
        url = f"{self.base_url}/standings"
        params = {"season": 2024}
        headers = {"Authorization": self.api_key}
        
        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(url, params=params, headers=headers)
                response.raise_for_status()
                data = response.json()
                
                # Convert list to dictionary with team_id as key
                standings_dict = {}
                for team in data.get('data', []):
                    if team['team']['id'] == player_id:
                        standings_dict[player_id] = {
                            'wins': team.get('wins', 0),
                            'losses': team.get('losses', 0),
                            'conference': team.get('conference', 'N/A'),
                            'conference_rank': team.get('conference_rank', 'N/A'),
                            'home_record': f"{team.get('home_wins', 0)}-{team.get('home_losses', 0)}",
                            'road_record': f"{team.get('road_wins', 0)}-{team.get('road_losses', 0)}",
                            'last_ten': f"{team.get('last_ten_wins', 0)}-{team.get('last_ten_losses', 0)}",
                            'streak': f"{'W' if team.get('streak_type') == 'win' else 'L'}{team.get('streak', 0)}"
                        }
                return standings_dict
                
        except Exception as e:
            logger.error(f"Error fetching standings: {str(e)}")
            return {}

    async def _get_team_leaders(self, team_id: int, season: int) -> Dict:
        """Get team statistical leaders."""
        url = f"{self.base_url}/leaders"
        stats = ['pts', 'reb', 'ast', 'stl', 'blk']
        leaders = {}
        
        for stat in stats:
            params = {
                "season": season,
                "stat_type": stat
            }
            headers = {"Authorization": self.api_key}
            
            try:
                async with httpx.AsyncClient() as client:
                    response = await client.get(url, params=params, headers=headers)
                    response.raise_for_status()
                    data = response.json()
                    # Filter for team's leaders
                    team_leaders = [p for p in data['data'] if p['player']['team_id'] == team_id]
                    if team_leaders:
                        leaders[stat] = team_leaders[0]
            except Exception as e:
                logger.error(f"Error fetching {stat} leaders: {str(e)}")
        
        return leaders

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

    async def get_standings(self, season: int = 2024) -> Dict:
        """Get current standings."""
        url = f"{self.base_url}/standings"
        params = {"season": season}
        headers = {"Authorization": self.api_key}
        
        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(url, params=params, headers=headers)
                response.raise_for_status()
                data = response.json()
                
                # Convert list to dictionary with team_id as key
                standings_dict = {}
                for team in data.get('data', []):
                    standings_dict[team['team']['id']] = {
                        'wins': team.get('wins', 0),
                        'losses': team.get('losses', 0),
                        'conference': team.get('conference', 'N/A'),
                        'conference_rank': team.get('conference_rank', 'N/A'),
                        'home_record': f"{team.get('home_wins', 0)}-{team.get('home_losses', 0)}",
                        'road_record': f"{team.get('road_wins', 0)}-{team.get('road_losses', 0)}",
                        'last_ten': f"{team.get('last_ten_wins', 0)}-{team.get('last_ten_losses', 0)}",
                        'streak': f"{'W' if team.get('streak_type') == 'win' else 'L'}{team.get('streak', 0)}"
                    }
                return standings_dict
                
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

    def _parse_odds_data(self, odds_data: List[Dict]) -> Dict:
        """Parse odds data to get spread and over/under."""
        if not odds_data:
            return {}
        
        parsed_odds = {
            'spread': None,
            'over_under': None
        }
        
        for odds in odds_data:
            if odds.get('type') == 'spread' and odds.get('live'):
                parsed_odds['spread'] = odds.get('away_spread')
            elif odds.get('type') == 'over/under' and odds.get('live'):
                parsed_odds['over_under'] = odds.get('over_under')
        
        return parsed_odds

    async def _generate_prediction(self, home_team: Dict, away_team: Dict, 
                                 standings: Dict, home_injuries: List, away_injuries: List,
                                 odds_data: List = None) -> str:
        """Generate prediction with consistent format."""
        try:
            # Analysis prompt for more detailed but focused analysis
            analysis_prompt = f"""
            Analyze this NBA matchup between {away_team['full_name']} and {home_team['full_name']}. 
            Consider their records, injuries, and recent performance.

            Provide your response in exactly this format:
            Winner: [Team Name] ([Win Probability]%)
            Analysis: 3-4 sentences analyzing key factors including records, matchup advantages, and injury impact
            Confidence: High/Medium/Low

            Current records:
            {away_team['full_name']}: {standings.get(away_team['id'], {}).get('wins', 0)}-{standings.get(away_team['id'], {}).get('losses', 0)}
            {home_team['full_name']}: {standings.get(home_team['id'], {}).get('wins', 0)}-{standings.get(home_team['id'], {}).get('losses', 0)}
            Injuries: {home_team['full_name']} ({len(home_injuries)} players out), {away_team['full_name']} ({len(away_injuries)} players out)
            """

            response = await asyncio.to_thread(
                self.openai_client.chat.completions.create,
                model="gpt-4-turbo-preview",
                messages=[
                    {"role": "system", "content": "You are an expert NBA analyst. Provide predictions in the exact format requested."},
                    {"role": "user", "content": analysis_prompt}
                ],
                temperature=0.7,
                max_tokens=200
            )
            
            ai_analysis = response.choices[0].message.content.strip()
            
            # Format the prediction
            prediction = f"🏀 {away_team['full_name']} (Away) @ {home_team['full_name']} (Home)\n\n"
            
            # Split the AI response into components
            lines = ai_analysis.split('\n')
            winner_line = next((line for line in lines if line.startswith('Winner:')), '')
            analysis_line = next((line for line in lines if line.startswith('Analysis:')), '')
            confidence_line = next((line for line in lines if line.startswith('Confidence:')), '')
            
            prediction += f"{winner_line}\n"
            prediction += f"{analysis_line}\n"
            prediction += f"{confidence_line}\n"
            
            # Format betting lines
            betting_lines = "\nBetting Lines:"  # Note: only one newline here
            if odds_data:
                latest_spread = None
                latest_over_under = None
                
                for odds in odds_data:
                    if not odds:
                        continue
                    
                    if odds.get('type') == 'spread':
                        if not latest_spread or odds.get('last_update', '') > latest_spread.get('last_update', ''):
                            latest_spread = odds
                    elif odds.get('type') == 'over/under':
                        if not latest_over_under or odds.get('last_update', '') > latest_over_under.get('last_update', ''):
                            latest_over_under = odds

                if latest_spread:
                    try:
                        away_spread = latest_spread.get('away_spread')
                        if away_spread is not None:
                            betting_lines += f"\n{away_team['full_name']} {away_spread}"
                    except (ValueError, TypeError) as e:
                        logger.error(f"Error processing spread: {str(e)}")

                if latest_over_under:
                    try:
                        total = latest_over_under.get('over_under')
                        if total is not None:
                            betting_lines += f"\nO {total}"
                    except (ValueError, TypeError) as e:
                        logger.error(f"Error processing over/under: {str(e)}")
            
            prediction += betting_lines
            
            return prediction

        except Exception as e:
            logger.error(f"Error generating prediction: {str(e)}")
            raise

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
        """Analyze a matchup and generate prediction."""
        try:
            # Get current season
            current_season = self._get_current_nba_season()
            
            # Get additional data needed for analysis
            home_injuries = await self.get_team_injuries(game['home_team']['id'])
            away_injuries = await self.get_team_injuries(game['visitor_team']['id'])
            standings = await self.get_standings(current_season)
            
            # Get odds data for the game
            odds_data = await self.get_betting_odds(game_id=game['id'])
            
            # Generate prediction
            prediction = await self._generate_prediction(
                game['home_team'],
                game['visitor_team'],
                standings,
                home_injuries,
                away_injuries,
                odds_data
            )
            
            return {
                "matchup": f"{game['visitor_team']['full_name']} @ {game['home_team']['full_name']}",
                "prediction": prediction,
                "data": {
                    "home_team": game['home_team'],
                    "away_team": game['visitor_team'],
                    "standings": standings,
                    "injuries": {
                        "home": home_injuries,
                        "away": away_injuries
                    },
                    "odds": odds_data
                }
            }
        except Exception as e:
            logger.error(f"Error in analyze_matchup: {str(e)}")
            raise

    async def parse_game_date(self, query: str) -> str:
        """Parse date from query with timezone handling."""
        try:
            query_lower = query.lower()
            
            # Get current time in EST/ET (NBA's timezone)
            est_tz = pytz.timezone('US/Eastern')
            current_date = datetime.now(est_tz)
            
            # Handle relative dates
            if 'tomorrow' in query_lower:
                target_date = current_date + timedelta(days=1)
            elif 'yesterday' in query_lower:
                target_date = current_date - timedelta(days=1)
            elif 'today' in query_lower or 'tonight' in query_lower:
                target_date = current_date
            else:
                # Convert common date formats to standard format
                # First, try to find date patterns in the query
                date_pattern = r'(?i)(jan|january|feb|february|mar|march|apr|april|may|jun|june|jul|july|aug|august|sep|september|oct|october|nov|november|dec|december)\s+\d{1,2}'
                match = re.search(date_pattern, query_lower)
                
                if match:
                    date_str = match.group(0)
                    # Try to parse the extracted date
                    parsed_date = dateparser.parse(
                        date_str,
                        settings={
                            'TIMEZONE': 'US/Eastern',
                            'RETURN_AS_TIMEZONE_AWARE': True,
                            'PREFER_DATES_FROM': 'future'
                        }
                    )
                    if parsed_date:
                        target_date = parsed_date
                    else:
                        raise ValueError(f"Could not parse date from: {date_str}")
                else:
                    # If no date pattern found, try parsing the entire query
                    parsed_date = dateparser.parse(
                        query_lower,
                        settings={
                            'TIMEZONE': 'US/Eastern',
                            'RETURN_AS_TIMEZONE_AWARE': True,
                            'PREFER_DATES_FROM': 'future'
                        }
                    )
                    if parsed_date:
                        target_date = parsed_date
                    else:
                        raise ValueError(f"Could not parse date from query: {query}")
            
            # Format the date in YYYY-MM-DD
            return target_date.strftime('%Y-%m-%d')
            
        except Exception as e:
            logger.error(f"Error parsing date from query: {str(e)}")
            raise ValueError(f"Unable to determine game date from query. Please specify a date like 'Jan 29' or 'January 29'")

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
        try:
            game_date = await predictor.parse_game_date(request.query)
        except ValueError as e:
            # Handle date parsing error
            agent_response = str(e)
            await store_message(
                session_id=request.session_id,
                message_type="ai",
                content=agent_response,
                data={"request_id": request.request_id}
            )
            return AgentResponse(success=True)

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
        raise HTTPException(
            status_code=500,
            detail=str(e)
        )

@app.get("/", response_class=HTMLResponse)
async def root(request: Request):
    with open("templates/index.html") as f:
        return HTMLResponse(content=f.read())

if __name__ == "__main__":
    import uvicorn
    # Feel free to change the port here if you need
    uvicorn.run(app, host="0.0.0.0", port=8001)
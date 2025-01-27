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
from openai import AsyncOpenAI
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
        self.openai_client = AsyncOpenAI(
            api_key=os.getenv("OPENAI_API_KEY")
        )

    def _get_current_nba_season(self) -> int:
        """
        Get the NBA season based on the game date.
        For the 2023-24 season, use 2023.
        """
        return 2023  # Hardcode to 2023 for now since that's what the API expects

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
        params = {"season": 2023}
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

    async def get_standings(self, season: int = 2023) -> Dict:
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

    async def get_season_averages(self, player_ids: List[int]) -> List[Dict]:
        """Fetch season averages for multiple players."""
        if not player_ids:
            return []

        season_averages = []
        
        # Create tasks for each player
        async with httpx.AsyncClient() as client:
            tasks = []
            for player_id in player_ids:
                url = f"{self.base_url}/season_averages"
                params = {
                    'season': 2024,
                    'player_id': player_id  # Changed from player_ids[] to player_id
                }
                tasks.append(
                    client.get(url, headers={'Authorization': self.api_key}, params=params)
                )
            
            # Execute all requests concurrently
            responses = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Process responses
            for response in responses:
                try:
                    if isinstance(response, Exception):
                        logger.error(f"Error fetching season averages: {str(response)}")
                        continue
                        
                    response.raise_for_status()
                    data = response.json()
                    if data.get('data'):
                        season_averages.extend(data['data'])
                except Exception as e:
                    logger.error(f"Error processing season averages response: {str(e)}")
                    continue

        return season_averages

    async def _generate_prediction(self, home_team: Dict, away_team: Dict, 
                                 standings: Dict, home_injuries: List, away_injuries: List,
                                 odds_data: List = None, home_stats: Dict = None, 
                                 away_stats: Dict = None) -> str:
        """Generate prediction using AI analysis of comprehensive team and player data."""
        try:
            # Get player IDs and stats
            home_player_ids = [int(k) for k in home_stats.keys() if str(k).isdigit()] if home_stats else []
            away_player_ids = [int(k) for k in away_stats.keys() if str(k).isdigit()] if away_stats else []
            
            # Get season averages
            home_season_stats = await self.get_season_averages(home_player_ids)
            away_season_stats = await self.get_season_averages(away_player_ids)

            # Format team records
            home_record = standings.get(home_team['id'], {})
            away_record = standings.get(away_team['id'], {})

            # Create detailed prompt with all available data
            prompt = f"""Analyze this NBA matchup and provide a detailed prediction:

Game: {away_team['full_name']} (Away) @ {home_team['full_name']} (Home)

HOME TEAM ({home_team['full_name']}):
Record: {home_record.get('wins', 0)}-{home_record.get('losses', 0)}
Home Record: {home_record.get('home_record', '0-0')}
Conference Rank: {home_record.get('conference_rank', 'N/A')}
Last 10 Games: {home_record.get('last_ten', 'N/A')}
Current Streak: {home_record.get('streak', 'N/A')}

Key Players Stats:
{self._format_season_stats(home_season_stats)}

Injuries: {', '.join([inj.get('player', {}).get('full_name', '') for inj in home_injuries]) if home_injuries else 'None reported'}

AWAY TEAM ({away_team['full_name']}):
Record: {away_record.get('wins', 0)}-{away_record.get('losses', 0)}
Road Record: {away_record.get('road_record', '0-0')}
Conference Rank: {away_record.get('conference_rank', 'N/A')}
Last 10 Games: {away_record.get('last_ten', 'N/A')}
Current Streak: {away_record.get('streak', 'N/A')}

Key Players Stats:
{self._format_season_stats(away_season_stats)}

Injuries: {', '.join([inj.get('player', {}).get('full_name', '') for inj in away_injuries]) if away_injuries else 'None reported'}

Betting Lines:
{self._format_betting_lines(odds_data) if odds_data else 'Not available'}

Based on this comprehensive data, provide:
1. A predicted winner with win probability
2. A detailed analysis considering:
   - Team records and current form
   - Home/Away performance
   - Key player matchups and their season averages
   - Impact of injuries
   - Conference standings impact
   - Recent performance trends
3. A confidence level (High/Medium/Low) with explanation

Format your response exactly as:
Winner: [Team] ([X]%)
Analysis: [Your detailed analysis]
Confidence: [Level] - [Brief explanation]"""

            try:
                # Get AI analysis
                response = await self.openai_client.chat.completions.create(
                    model="gpt-4",
                    messages=[
                        {
                            "role": "system", 
                            "content": "You are an expert NBA analyst. Provide detailed analysis based on current season data, team performance, and player statistics. Be specific and reference actual statistics in your analysis."
                        },
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.7
                )
                
                ai_analysis = response.choices[0].message.content
                
                # Validate AI response format
                if not all(section in ai_analysis for section in ['Winner:', 'Analysis:', 'Confidence:']):
                    logger.warning("AI response missing required sections")
                    ai_analysis = self._generate_fallback_analysis(home_team, away_team, home_record, away_record)
            
            except Exception as e:
                logger.error(f"Error getting AI analysis: {str(e)}")
                ai_analysis = self._generate_fallback_analysis(home_team, away_team, home_record, away_record)

            # Format the final prediction
            matchup = f"NBA: {away_team['full_name']} (Away) @ {home_team['full_name']} (Home)"
            return f"{matchup}\n\n{ai_analysis}"

        except Exception as e:
            logger.error(f"Error generating prediction: {str(e)}")
            raise

    def _generate_fallback_analysis(self, home_team: Dict, away_team: Dict, 
                                  home_record: Dict, away_record: Dict) -> str:
        """Generate a basic analysis when AI analysis fails."""
        home_wins = home_record.get('wins', 0)
        home_losses = home_record.get('losses', 0)
        away_wins = away_record.get('wins', 0)
        away_losses = away_record.get('losses', 0)
        
        home_win_pct = home_wins / (home_wins + home_losses) if (home_wins + home_losses) > 0 else 0.5
        away_win_pct = away_wins / (away_wins + away_losses) if (away_wins + away_losses) > 0 else 0.5
        
        winner = home_team['full_name'] if home_win_pct >= away_win_pct else away_team['full_name']
        probability = int(max(home_win_pct, away_win_pct) * 100)
        
        return (f"Winner: {winner} ({probability}%)\n"
                f"Analysis: Based on current season records, {winner} has a better win percentage "
                f"({home_win_pct:.3f} vs {away_win_pct:.3f}). "
                f"Home team record: {home_wins}-{home_losses}, Away team record: {away_wins}-{away_losses}.\n"
                f"Confidence: Medium - Based on win-loss records only")

    def _format_season_stats(self, season_stats: List[Dict]) -> str:
        """Format season averages for display."""
        if not season_stats:
            return "No season averages available"
        
        formatted_stats = []
        for stat in season_stats:
            if stat:  # Only format if we have stats
                formatted_stats.append(
                    f"- {stat.get('player_name', 'Unknown')}: "
                    f"{stat.get('pts', 0):.1f} PPG, "
                    f"{stat.get('reb', 0):.1f} RPG, "
                    f"{stat.get('ast', 0):.1f} APG, "
                    f"{stat.get('min', '0')} MPG, "
                    f"FG%: {stat.get('fg_pct', 0):.1f}, "
                    f"3P%: {stat.get('fg3_pct', 0):.1f}"
                )
        
        return "\n".join(formatted_stats) if formatted_stats else "No season averages available"

    def _format_betting_lines(self, odds_data: List[Dict]) -> str:
        """Format betting lines for display."""
        formatted_lines = []
        for odds in odds_data:
            if odds.get('type') == 'spread':
                formatted_lines.append(f"{odds.get('away_team', {}).get('full_name')} {odds.get('away_spread')}")
            elif odds.get('type') == 'over/under':
                formatted_lines.append(f"O {odds.get('over_under')}")
        
        return "\n".join(formatted_lines) if formatted_lines else "No betting lines available"

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

    def _generate_parlay_prediction(self, predictions: List[Dict]) -> str:
        """Generate a parlay prediction based on highest confidence picks."""
        try:
            # Filter predictions with high confidence
            high_confidence_picks = []
            
            for pred in predictions:
                # Extract win probability from prediction string
                winner_line = [line for line in pred['prediction'].split('\n') if 'Winner:' in line][0]
                prob_str = winner_line.split('(')[1].split('%')[0]
                confidence_line = [line for line in pred['prediction'].split('\n') if 'Confidence:' in line][0]
                confidence = confidence_line.split(': ')[1]
                
                # Get betting lines
                betting_lines = pred['prediction'].split('Betting Lines:')[1].strip().split('\n') if 'Betting Lines:' in pred['prediction'] else []
                
                if confidence == "High" and float(prob_str) > 65:
                    high_confidence_picks.append({
                        'matchup': pred['matchup'],
                        'winner': winner_line.split(': ')[1].split(' (')[0],
                        'probability': float(prob_str),
                        'betting_lines': betting_lines
                    })

            if not high_confidence_picks:
                return "I don't have enough high-confidence picks to recommend a parlay today."

            # Sort by probability
            high_confidence_picks.sort(key=lambda x: x['probability'], reverse=True)
            
            # Take top 2-3 picks
            parlay_picks = high_confidence_picks[:min(3, len(high_confidence_picks))]
            
            # Calculate combined probability
            combined_prob = 100
            for pick in parlay_picks:
                combined_prob *= (pick['probability'] / 100)
            
            # Generate parlay prediction
            parlay = "🎲 Recommended Parlay:\n\n"
            for i, pick in enumerate(parlay_picks, 1):
                parlay += f"{i}. {pick['winner']}"
                # Add betting line if available
                if pick['betting_lines']:
                    spread_line = next((line for line in pick['betting_lines'] if pick['winner'] in line), '')
                    if spread_line:
                        parlay += f" ({spread_line.split(pick['winner'])[1].strip()})"
                parlay += f" ({pick['probability']:.0f}% probability)\n"
            
            parlay += f"\nCombined Probability: {combined_prob:.1f}%\n"
            parlay += "Note: This parlay combines our highest confidence picks based on team performance, injuries, and betting lines."
            
            return parlay
            
        except Exception as e:
            logger.error(f"Error generating parlay: {str(e)}")
            return "Sorry, I couldn't generate a parlay prediction at this time."

    async def get_parlay_prediction(self, date: str) -> str:
        """Get parlay prediction for games on a specific date."""
        try:
            games = await self.get_games(date)
            if not games:
                return f"No games found for {date}"

            all_predictions = []
            for game in games:
                prediction_data = await self._generate_prediction(
                    home_team=game['home_team'],
                    away_team=game['visitor_team'],
                    standings=await self.get_standings(),
                    home_injuries=await self.get_team_injuries(game['home_team']['id']),
                    away_injuries=await self.get_team_injuries(game['visitor_team']['id']),
                    odds_data=await self.get_betting_odds(game_id=game['id']),
                    home_stats=await self._get_team_leaders(game['home_team']['id'], 2024),
                    away_stats=await self._get_team_leaders(game['visitor_team']['id'], 2024)
                )
                all_predictions.append(prediction_data)

            return self._generate_parlay_prediction(all_predictions)

        except Exception as e:
            logger.error(f"Error getting parlay prediction: {str(e)}")
            raise

    async def analyze_matchup(self, game: Dict) -> Dict:
        """Analyze NBA matchup with complete data."""
        try:
            home_team = game['home_team']
            away_team = game['visitor_team']
            
            # Get all required data concurrently
            tasks = [
                self.get_team_injuries(home_team['id']),
                self.get_team_injuries(away_team['id']),
                self.get_standings(2024),
                self.get_team_stats(home_team['id']),
                self.get_team_stats(away_team['id'])
            ]
            
            # Execute first batch of tasks
            home_injuries, away_injuries, standings, home_team_stats, away_team_stats = await asyncio.gather(*tasks)
            
            # Get players for both teams
            home_players = await self.get_team_players(home_team['id'])
            away_players = await self.get_team_players(away_team['id'])
            
            # Get player IDs
            home_player_ids = [p['id'] for p in home_players]
            away_player_ids = [p['id'] for p in away_players]
            
            # Second batch of tasks for player-specific data
            player_tasks = [
                self.get_player_stats(home_player_ids),
                self.get_player_stats(away_player_ids),
                self.get_season_averages(home_player_ids),
                self.get_season_averages(away_player_ids),
                self.get_betting_odds(game['id'])
            ]
            
            # Execute second batch of tasks
            home_stats, away_stats, home_season_avgs, away_season_avgs, odds_data = await asyncio.gather(*player_tasks)

            # Generate prediction with complete dataset
            prediction = await self._generate_prediction(
                home_team=home_team,
                away_team=away_team,
                standings=standings,
                home_injuries=home_injuries,
                away_injuries=away_injuries,
                odds_data=odds_data,
                home_stats=home_stats,
                away_stats=away_stats,
                home_season_avgs=home_season_avgs,
                away_season_avgs=away_season_avgs,
                home_team_stats=home_team_stats,
                away_team_stats=away_team_stats
            )

            return {
                "matchup": f"{away_team['full_name']} (Away) @ {home_team['full_name']} (Home)",
                "prediction": prediction,
                "data": {
                    "teams": {
                        "home": {
                            "name": home_team['full_name'],
                            "record": f"{standings.get(home_team['id'], {}).get('wins', 0)}-{standings.get(home_team['id'], {}).get('losses', 0)}",
                            "home_record": standings.get(home_team['id'], {}).get('home_record'),
                            "team_stats": home_team_stats,
                            "injuries": home_injuries,
                            "player_stats": home_stats,
                            "season_averages": home_season_avgs
                        },
                        "away": {
                            "name": away_team['full_name'],
                            "record": f"{standings.get(away_team['id'], {}).get('wins', 0)}-{standings.get(away_team['id'], {}).get('losses', 0)}",
                            "road_record": standings.get(away_team['id'], {}).get('road_record'),
                            "team_stats": away_team_stats,
                            "injuries": away_injuries,
                            "player_stats": away_stats,
                            "season_averages": away_season_avgs
                        }
                    },
                    "odds": odds_data
                }
            }

        except Exception as e:
            logger.error(f"Error in analyze_matchup: {str(e)}")
            raise

    async def get_team_stats(self, team_id: int) -> Dict:
        """Fetch team statistics for the current season."""
        url = f"{self.base_url}/teams/{team_id}/stats"
        
        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(
                    url, 
                    headers={'Authorization': self.api_key},
                    params={'season': 2024}
                )
                response.raise_for_status()
                data = response.json()
                
                # Extract relevant team stats
                team_stats = data.get('data', {})
                if team_stats:
                    return {
                        'ppg': team_stats.get('pts_per_game', 0),
                        'rpg': team_stats.get('reb_per_game', 0),
                        'apg': team_stats.get('ast_per_game', 0),
                        'spg': team_stats.get('stl_per_game', 0),
                        'bpg': team_stats.get('blk_per_game', 0),
                        'fg_pct': team_stats.get('fg_pct', 0),
                        'fg3_pct': team_stats.get('fg3_pct', 0),
                        'ft_pct': team_stats.get('ft_pct', 0),
                        'off_rtg': team_stats.get('off_rating', 0),
                        'def_rtg': team_stats.get('def_rating', 0)
                    }
                return {}
                
        except Exception as e:
            logger.error(f"Error fetching team stats for team {team_id}: {str(e)}")
            return {}

    async def get_team_players(self, team_id: int) -> List[Dict]:
        """Fetch all players for a team."""
        url = f"{self.base_url}/players"
        params = {
            'team_ids[]': team_id,
            'per_page': 100,  # Get all players
            'season': 2024
        }
        
        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(url, headers={'Authorization': self.api_key}, params=params)
                response.raise_for_status()
                data = response.json()
                return data.get('data', [])
        except Exception as e:
            logger.error(f"Error fetching team players for team {team_id}: {str(e)}")
            return []

    async def get_player_stats(self, player_ids: List[int]) -> Dict[int, Dict]:
        """Fetch stats for multiple players."""
        if not player_ids:
            return {}
        
        stats_dict = {}
        for player_id in player_ids:
            url = f"{self.base_url}/stats"
            params = {
                'player_ids[]': player_id,
                'per_page': 100,
                'seasons[]': 2024
            }
            
            try:
                async with httpx.AsyncClient() as client:
                    response = await client.get(url, headers={'Authorization': self.api_key}, params=params)
                    response.raise_for_status()
                    data = response.json()
                    if data.get('data'):
                        # Get the most recent stats
                        stats_dict[player_id] = data['data'][0]
            except Exception as e:
                logger.error(f"Error fetching stats for player {player_id}: {str(e)}")
                continue
            
        return stats_dict

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
        
        # Store user's message
        await store_message(
            session_id=request.session_id,
            message_type="human",
            content=request.query,
            data={"request_id": request.request_id}
        )

        predictor = NBAPredictor()
        game_date = await predictor.parse_game_date(request.query)
        logger.info(f"Parsed date for games: {game_date}")
        
        # Initialize response_data
        response_data = {}
        
        # Check if user is asking for a parlay
        if 'parlay' in request.query.lower():
            agent_response = await predictor.get_parlay_prediction(game_date)
            response_data = {
                "date": game_date,
                "type": "parlay",
                "prediction": agent_response
            }
        else:
            games = await predictor.get_games(game_date)
            logger.info(f"Found {len(games)} games for {game_date}")
            
            if not games:
                agent_response = f"I couldn't find any NBA games scheduled for {game_date}."
                response_data = {
                    "date": game_date,
                    "games_count": 0
                }
            else:
                # Process each game and await the results
                all_predictions = []
                for game in games:
                    prediction_data = await predictor.analyze_matchup(game)
                    all_predictions.append(prediction_data)

                # Format the response
                if len(all_predictions) == 1:
                    agent_response = all_predictions[0]
                else:
                    agent_response = "\n\n".join(pred for pred in all_predictions)
                
                response_data = {
                    "date": game_date,
                    "games_count": len(games),
                    "predictions": all_predictions
                }

        # Store AI's response with the awaited results
        await store_message(
            session_id=request.session_id,
            message_type="ai",
            content=agent_response,
            data={
                "request_id": request.request_id,
                **response_data
            }
        )

        return AgentResponse(success=True)

    except Exception as e:
        logger.error(f"Error processing request: {str(e)}")
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
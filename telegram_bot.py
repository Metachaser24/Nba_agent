from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes
from nba_agent import NBAPredictor, team_names
from dotenv import load_dotenv
import os
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Check for Telegram token
if not os.getenv("TELEGRAM_BOT_TOKEN"):
    raise ValueError("TELEGRAM_BOT_TOKEN environment variable is not set")

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Send a message when the command /start is issued."""
    await update.message.reply_text('Hi! I am your NBA Predictions Bot. Ask me about any NBA games!')

async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Send a message when the command /help is issued."""
    help_text = """
Here's how to use me:
- Ask about today's games: "What are today's games?"
- Ask about a specific team: "What's the Hornets game today?"
- Ask about tomorrow's games: "What are tomorrow's games?"
    """
    await update.message.reply_text(help_text)

async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle the user message."""
    try:
        predictor = NBAPredictor()
        query = update.message.text
        
        # Show typing indicator
        await update.message.chat.send_action(action="typing")
        
        game_date = await predictor.parse_game_date(query)
        games = await predictor.get_games(game_date)
        
        if not games:
            await update.message.reply_text(f"I couldn't find any NBA games scheduled for {game_date}.")
            return

        # Use existing team filtering logic
        query_lower = query.lower()
        requested_team = None
        for team, variations in team_names.items():
            if any(variation in query_lower for variation in variations):
                requested_team = team
                break
        
        if requested_team:
            team_specific_games = [
                game for game in games 
                if any(variation in game['home_team']['full_name'].lower() for variation in team_names[requested_team])
                or any(variation in game['visitor_team']['full_name'].lower() for variation in team_names[requested_team])
            ]
            games = team_specific_games

        # Generate predictions
        all_predictions = []
        for game in games:
            prediction_data = await predictor.analyze_matchup(game)
            all_predictions.append(prediction_data)

        # Format response
        if requested_team and not team_specific_games:
            response = f"I couldn't find any games scheduled for {requested_team.title()} on {game_date}."
        elif requested_team:
            response = f"Here's my prediction for the {requested_team.title()} game on {game_date}:\n\n"
            for pred in all_predictions:
                response += pred["prediction"] + "\n\n"
        else:
            response = f"I found {len(games)} games scheduled for {game_date}. Here are my predictions:\n\n"
            for pred in all_predictions:
                response += pred["prediction"] + "\n\n"

        await update.message.reply_text(response)

    except Exception as e:
        logger.error(f"Error processing message: {str(e)}", exc_info=True)
        await update.message.reply_text("Sorry, I encountered an error processing your request.")

def main():
    """Start the bot."""
    # Create the Application and pass it your bot's token
    application = Application.builder().token(os.getenv("TELEGRAM_BOT_TOKEN")).build()

    # Add handlers
    application.add_handler(CommandHandler("start", start))
    application.add_handler(CommandHandler("help", help_command))
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))

    # Run the bot
    application.run_polling()

if __name__ == "__main__":
    main()
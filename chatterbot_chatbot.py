from chatterbot import ChatBot
from chatterbot.trainers import ChatterBotCorpusTrainer

# Create chatbot instance
chatbot = ChatBot('MyBot')

# Create a new trainer for the chatbot
trainer = ChatterBotCorpusTrainer(chatbot)

# Train the chatbot based on the English corpus data
trainer.train("chatterbot.corpus.english")

print("Chatbot is ready to talk! Type 'quit' to exit.")

while True:
    try:
        user_input = input("You: ")
        if user_input.lower() == 'quit':
            print("Goodbye!")
            break

        bot_response = chatbot.get_response(user_input)
        print("Bot:", bot_response)

    except (KeyboardInterrupt, EOFError, SystemExit):
        break
